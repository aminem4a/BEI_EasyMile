import numpy as np
from .models.vehicle import Vehicle
from .control.controllers import SpeedController
from .control.allocations import TorqueAllocator
from .config import SimulationConfig, VehicleConfig

class Simulation:
    def __init__(self, sim_config: SimulationConfig, vehicle_config: VehicleConfig):
        self.sim_config = sim_config
        self.veh_config = vehicle_config
        
        # Initialize sub-modules
        # Note: on passe dt au véhicule pour configurer les moteurs Strejc
        self.vehicle = Vehicle(vehicle_config, dt=sim_config.dt)
        self.controller = SpeedController(kp=12000.0, ki=4000.0, kd=0.0)
        self.allocator = TorqueAllocator(alpha=0.005, max_rate=300.0)
        
        # Data logging 
        self.history = {
            "time": [], "velocity": [], "cmd_torque": [],
            "torque_fl": [], "torque_rl": []
        }

    def run(self, target_speed: float):
        """
        Runs the simulation loop.
        """
        t = 0.0
        steps = int(self.sim_config.duration / self.sim_config.dt)

        print(f"Simulation démarrée : Cible {target_speed*3.6:.1f} km/h")

        for _ in range(steps):
            # 1. Sense
            current_v = self.vehicle.velocity
            
            # 2. Control (PI -> Couple Global aux Roues)
            total_torque_roues = self.controller.compute_command(target_speed, current_v, self.sim_config.dt)
            
            # Conversion pour l'allocateur (qui parle en couple moteur)
            total_torque_moteurs = total_torque_roues / self.veh_config.ratio_reduction
            
            # 3. Allocate (Distribute torque)
            # Calcul RPM moteur pour l'efficiency map
            v_rpm = (current_v / self.veh_config.wheel_radius) * self.veh_config.ratio_reduction * 9.549
            
            wheel_cmds = self.allocator.allocate(total_torque_moteurs, speed_rpm=v_rpm, dt=self.sim_config.dt)
            
            # 4. Actuate (Update Physics)
            self.vehicle.update_dynamics(wheel_cmds, self.sim_config.dt)
            
            # 5. Log Data
            self.history["time"].append(t)
            self.history["velocity"].append(current_v)
            self.history["cmd_torque"].append(total_torque_moteurs)
            self.history["torque_fl"].append(wheel_cmds[0]) # Avant Gauche
            self.history["torque_rl"].append(wheel_cmds[2]) # Arrière Gauche
            
            t += self.sim_config.dt

        return self.history