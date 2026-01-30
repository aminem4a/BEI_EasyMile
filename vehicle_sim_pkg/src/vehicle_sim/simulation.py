import os
import sys
import numpy as np

# Path Hack
current_dir = os.path.dirname(os.path.abspath(__file__))
pkg_dir = os.path.dirname(os.path.dirname(current_dir))
if pkg_dir not in sys.path: sys.path.append(pkg_dir)

try:
    from src.vehicle_sim.utils.data_loader import DataLoader
    from src.vehicle_sim.control.allocation import TorqueAllocator
except ImportError:
    from src.vehicle_sim.utils.data_loader import DataLoader
    from src.vehicle_sim.control.allocation import TorqueAllocator

class Simulation:
    def __init__(self, sim_cfg, veh_cfg, data_dir=None):
        self.sim_cfg = sim_cfg
        self.veh_cfg = veh_cfg
        
        if data_dir is None: data_dir = os.path.join(pkg_dir, "data")
        
        map_path = os.path.join(data_dir, "efficiency_map_clean.csv")
        if not os.path.exists(map_path): map_path = os.path.join(data_dir, "efficiency_map.csv")

        self.loader = DataLoader(map_path)
        self.allocator = TorqueAllocator(self.loader)
        self.strategies = ["Inverse", "Piecewise", "Smooth", "Quadratic"]

    def _val(self, obj, key, default):
        if isinstance(obj, dict): return obj.get(key, default)
        return getattr(obj, key, default)

    def run_open_loop(self, t, trq_profile, v_profile=None):
        wheel_r = self._val(self.veh_cfg, 'wheel_radius', 0.3)
        
        if v_profile is None:
            rpm_target = self._val(self.sim_cfg, 'rpm', 500.0)
            v_const = (rpm_target * 2 * np.pi * wheel_r) / 60
            v_profile = np.full_like(t, v_const)

        global_res = {}
        print(f"▶️ Simulation Open Loop ({len(t)} points)...")

        for strat in self.strategies:
            r_ratio = []
            r_power = []
            
            # Nouvelles données demandées
            r_cos_phi_f = []
            r_cos_phi_r = []
            r_trq_achieved = [] # Suivi de consigne
            
            prev_ratio = 0.5
            
            for i in range(len(t)):
                T_req = trq_profile[i]
                v = v_profile[i]
                rpm = (v * 60) / (2 * np.pi * wheel_r)
                
                # Optimisation
                res = self.allocator.optimize(strat, T_req, rpm, prev_front_ratio=prev_ratio)
                
                Tf = res['T_front']
                Tr = res['T_rear']
                
                # Mise à jour Ratio
                if abs(T_req) < 0.1:
                    ratio = prev_ratio 
                else:
                    ratio = Tf / T_req
                    prev_ratio = ratio
                
                # --- CALCULS AVANCÉS ---
                # 1. Pertes recalculées via la MAP (Vérité terrain)
                loss_f = self.loader.get_loss(Tf, rpm)
                loss_r = self.loader.get_loss(Tr, rpm)
                
                # 2. Puissances Méca
                w = rpm * 2 * np.pi / 60
                Pm_f = Tf * w
                Pm_r = Tr * w
                
                # 3. Puissances Elec
                Pe_f = Pm_f + loss_f
                Pe_r = Pm_r + loss_r
                
                # 4. Cos Phi (Efficacité ~ Pm / Pe)
                # On évite la div par 0 et on borne entre 0 et 1
                def calc_cos(pm, pe):
                    if abs(pe) < 1.0: return 0.0
                    val = pm / pe
                    return min(max(val, 0.0), 1.0) # Clamp 0-1

                cos_f = calc_cos(Pm_f, Pe_f)
                cos_r = calc_cos(Pm_r, Pe_r)
                
                # Stockage
                r_ratio.append(ratio)
                r_power.append(Pe_f + Pe_r)
                r_cos_phi_f.append(cos_f)
                r_cos_phi_r.append(cos_r)
                r_trq_achieved.append(Tf + Tr) # Couple réel total
                
            global_res[strat] = {
                'time': t, 
                'front_ratio': r_ratio, 
                'power': r_power,
                'cos_phi_f': r_cos_phi_f,
                'cos_phi_r': r_cos_phi_r,
                'trq_achieved': r_trq_achieved
            }
            
        return global_res