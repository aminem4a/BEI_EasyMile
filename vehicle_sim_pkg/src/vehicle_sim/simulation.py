# -*- coding: utf-8 -*-
import numpy as np
from .models.vehicle import Vehicle
from .control.controllers import SpeedController
from .control.allocations import TorqueAllocator
from .config import SimulationConfig, VehicleConfig

class Simulation:
    def __init__(self, sim_config: SimulationConfig, vehicle_config: VehicleConfig):
        self.sim_config = sim_config
        self.veh_config = vehicle_config
        
        # Initialisation des sous-modules
        # On passe le dt au véhicule car il en a besoin pour ses moteurs internes
        self.vehicle = Vehicle(vehicle_config, dt=sim_config.dt)
        
        # Le contrôleur (on utilise les gains définis ou par défaut)
        # Note: kd=0 car nous utilisons principalement un PI pour la vitesse
        self.controller = SpeedController(kp=12000.0, ki=4000.0, kd=0.0)
        
        # L'allocateur intelligent (Smooth + Rate Limiter)
        self.allocator = TorqueAllocator(alpha=0.005, max_rate=300.0)
        
        # Data logging : C'est ici qu'on définit tout ce qu'on veut tracer
        self.history = {
            "time": [],
            "velocity": [],      # Vitesse réelle (m/s)
            "target_speed": [],  # Consigne de vitesse (m/s)
            "cmd_torque": [],    # Couple global demandé par le PI
            "torque_fl": [],     # Couple moteur Avant-Gauche
            "torque_rl": [],     # Couple moteur Arrière-Gauche
            "cosphi_av": [],     # Efficacité moteur Avant
            "cosphi_ar": []      # Efficacité moteur Arrière
        }

    def run(self, target_speed: float):
        """
        Exécute la boucle de simulation.
        target_speed est en m/s.
        """
        t = 0.0
        dt = self.sim_config.dt
        steps = int(self.sim_config.duration / dt)

        print(f"Simulation lancée pour {self.sim_config.duration}s...")

        for _ in range(steps):
            # 1. Capture de l'état actuel (Vitesse)
            current_v = self.vehicle.velocity
            
            # 2. Calcul du besoin en couple global (PI)
            # Sortie en Couple aux ROUES
            total_torque_roues = self.controller.compute_command(target_speed, current_v, dt)
            
            # Conversion en couple global MOTEUR pour l'allocateur
            total_torque_moteurs = total_torque_roues / self.veh_config.ratio_reduction
            
            # 3. Allocation (Répartition intelligente)
            # On calcule les RPM moteurs pour que l'allocateur trouve le bon CosPhi
            v_rpm = (current_v / self.veh_config.wheel_radius) * self.veh_config.ratio_reduction * 9.549
            
            wheel_cmds = self.allocator.allocate(total_torque_moteurs, speed_rpm=v_rpm, dt=dt)
            
            # 4. Mise à jour de la physique (Moteurs + Dynamique Véhicule)
            self.vehicle.update_dynamics(wheel_cmds, dt)
            
            # 5. Enregistrement des données (Logging)
            self.history["time"].append(t)
            self.history["velocity"].append(current_v)
            self.history["target_speed"].append(target_speed)
            self.history["cmd_torque"].append(total_torque_moteurs)
            
            # On logue les couples individuels (moteurs)
            self.history["torque_fl"].append(wheel_cmds[0])
            self.history["torque_rl"].append(wheel_cmds[2])
            
            # On logue l'efficacité calculée
            cp_av = self.allocator._get_cp(wheel_cmds[0], v_rpm)
            cp_ar = self.allocator._get_cp(wheel_cmds[2], v_rpm)
            self.history["cosphi_av"].append(cp_av)
            self.history["cosphi_ar"].append(cp_ar)
            
            t += dt

        return self.history