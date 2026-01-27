# -*- coding: utf-8 -*-
from .models import Vehicle
from .control import SpeedController, TorqueAllocator
from .config import SimulationConfig, VehicleConfig
from .utils import DataLoader

class Simulation:
    def __init__(self, sim_cfg: SimulationConfig, veh_cfg: VehicleConfig, data_dir: str = None):
        self.sim_cfg = sim_cfg
        
        # Chargement des données (Map + Limites)
        self.loader = None
        if data_dir:
            self.loader = DataLoader(data_dir)
            if self.loader.torque_max_interp:
                veh_cfg.max_motor_torque = self.loader.get_max_torque(0)

        self.vehicle = Vehicle(veh_cfg, dt=sim_cfg.dt)
        
        # Le PID est créé mais ne servira que pour run() (Boucle Fermée)
        self.controller = SpeedController(kp=12000.0, ki=4000.0)
        
        # L'allocateur sert pour les deux modes
        self.allocator = TorqueAllocator(mode="smooth", data_loader=self.loader)
        
        self.history = {
            "time": [], "velocity": [], 
            "torque_fl": [], "torque_rl": [], 
            "cosphi_av": [], "cosphi_ar": []
        }

    def run(self, target_speed_profile=None):
        """
        MODE 1 : BOUCLE FERMÉE (Closed Loop)
        Le PID adapte le couple pour suivre la vitesse cible.
        Utilisé par : run_real_scenario.py
        """
        self._reset_history()
        t = 0.0
        steps = int(self.sim_cfg.duration / self.sim_cfg.dt)
        print_interval = max(1, steps // 10)
        
        for i in range(steps):
            if i % print_interval == 0:
                print(f"   [BF] Calcul : {(i/steps)*100:.0f}%")

            # 1. Consigne de Vitesse
            if hasattr(target_speed_profile, '__call__'):
                tgt = float(target_speed_profile(t))
            else:
                tgt = float(target_speed_profile or 0.0)

            # 2. Vitesse actuelle
            v = self.vehicle.velocity
            v_rpm = (v / self.vehicle.config.wheel_radius) * self.vehicle.config.ratio_reduction * 9.55
            
            # 3. PID : Calcul du Couple nécessaire pour atteindre la vitesse
            t_roues = self.controller.compute_command(tgt, v, self.sim_cfg.dt)
            t_moteurs_total = t_roues / self.vehicle.config.ratio_reduction
            
            # 4. Allocation & Physique
            self._step_physics(t_moteurs_total, v_rpm, t, v)
            t += self.sim_cfg.dt
            
        return self.history

    def run_open_loop(self, torque_profile_func):
        """
        MODE 2 : BOUCLE OUVERTE (Open Loop)
        On injecte directement un profil de couple (ex: issu d'un fichier).
        La vitesse résulte de la physique (F=ma).
        Utilisé par : run_open_loop.py
        """
        self._reset_history()
        t = 0.0
        steps = int(self.sim_cfg.duration / self.sim_cfg.dt)
        print_interval = max(1, steps // 10)
        
        for i in range(steps):
            if i % print_interval == 0:
                print(f"   [BO] Calcul : {(i/steps)*100:.0f}%")

            # 1. Vitesse actuelle (pour info et calcul CosPhi)
            v = self.vehicle.velocity
            v_rpm = (v / self.vehicle.config.wheel_radius) * self.vehicle.config.ratio_reduction * 9.55
            
            # 2. Lecture Directe du Couple (Pas de PID)
            t_roues_total = float(torque_profile_func(t))
            t_moteurs_total = t_roues_total / self.vehicle.config.ratio_reduction
            
            # 3. Allocation & Physique
            self._step_physics(t_moteurs_total, v_rpm, t, v)
            t += self.sim_cfg.dt
            
        return self.history

    def _step_physics(self, t_moteurs_total, v_rpm, t, v):
        """Méthode interne partagée pour éviter de copier-coller le code"""
        # Allocation
        cmds = self.allocator.allocate(t_moteurs_total, v_rpm, self.sim_cfg.dt)
        
        # Application Physique
        self.vehicle.update_dynamics(cmds, self.sim_cfg.dt)
        
        # Enregistrement
        self.history["time"].append(t)
        self.history["velocity"].append(v)
        self.history["torque_fl"].append(cmds[0])
        self.history["torque_rl"].append(cmds[2])
        
        rads = v_rpm * 0.1047
        self.history["cosphi_av"].append(self.allocator._estimate_cosphi(cmds[0], rads))
        self.history["cosphi_ar"].append(self.allocator._estimate_cosphi(cmds[2], rads))

    def _reset_history(self):
        self.history = {
            "time": [], "velocity": [], 
            "torque_fl": [], "torque_rl": [], 
            "cosphi_av": [], "cosphi_ar": []
        }