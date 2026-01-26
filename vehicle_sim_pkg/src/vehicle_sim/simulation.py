# -*- coding: utf-8 -*-
from .models import Vehicle
from .control import SpeedController, TorqueAllocator
from .config import SimulationConfig, VehicleConfig
from .utils import DataLoader

class Simulation:
    def __init__(self, sim_cfg: SimulationConfig, veh_cfg: VehicleConfig, data_dir: str = None):
        self.sim_cfg = sim_cfg
        
        self.loader = None
        if data_dir:
            self.loader = DataLoader(data_dir)
            if self.loader.torque_max_interp:
                veh_cfg.max_motor_torque = self.loader.get_max_torque(0)

        self.vehicle = Vehicle(veh_cfg, dt=sim_cfg.dt)
        self.controller = SpeedController(kp=12000.0, ki=4000.0)
        self.allocator = TorqueAllocator(mode="smooth", data_loader=self.loader)
        
        self.history = {
            "time": [], "velocity": [], 
            "torque_fl": [], "torque_rl": [], 
            "cosphi_av": [], "cosphi_ar": []
        }

    def run(self, target_speed_profile=None):
        t = 0.0
        steps = int(self.sim_cfg.duration / self.sim_cfg.dt)
        
        # --- INTERVALLE D'AFFICHAGE ---
        print_interval = max(1, steps // 10) # Affiche tous les 10%
        
        for i in range(steps):
            if i % print_interval == 0:
                print(f"   ... Calcul : {(i/steps)*100:.0f}%")

            # Consigne
            if hasattr(target_speed_profile, '__call__'):
                tgt = float(target_speed_profile(t))
            else:
                tgt = float(target_speed_profile or 0.0)

            # Boucle
            v = self.vehicle.velocity
            v_rpm = (v / self.vehicle.config.wheel_radius) * self.vehicle.config.ratio_reduction * 9.55
            
            t_roues = self.controller.compute_command(tgt, v, self.sim_cfg.dt)
            t_moteurs_total = t_roues / self.vehicle.config.ratio_reduction
            
            cmds = self.allocator.allocate(t_moteurs_total, v_rpm, self.sim_cfg.dt)
            self.vehicle.update_dynamics(cmds, self.sim_cfg.dt)
            
            # Logs
            self.history["time"].append(t)
            self.history["velocity"].append(v)
            self.history["torque_fl"].append(cmds[0])
            self.history["torque_rl"].append(cmds[2])
            
            rads = v_rpm * 0.1047
            self.history["cosphi_av"].append(self.allocator._estimate_cosphi(cmds[0], rads))
            self.history["cosphi_ar"].append(self.allocator._estimate_cosphi(cmds[2], rads))
            
            t += self.sim_cfg.dt
            
        return self.history