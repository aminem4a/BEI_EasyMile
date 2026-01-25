import numpy as np
from typing import List
from .engine import Engine
from ..config import VehicleConfig

class Vehicle:
    """
    4-Wheel Vehicle Dynamic Model (Longitudinal).
    """
    def __init__(self, config: VehicleConfig, dt: float = 0.01):
        self.config = config
        
        # Instantiate 4 engines with Strejc config
        self.engines: List[Engine] = [
            Engine(
                max_torque=config.motor_max_torque,
                k=config.strejc_k, n=config.strejc_n, 
                ta=config.strejc_ta, tu=config.strejc_tu, 
                dt=dt
            ) for _ in range(4)
        ]
        
        # State variables
        self.velocity = 0.0  # m/s
        self.position = 0.0  # m
        self.acceleration = 0.0 # m/s^2

    def update_dynamics(self, wheel_torques: List[float], dt: float) -> None:
        """
        Integrates vehicle dynamics for one time step.
        Args:
            wheel_torques: List of 4 MOTOR torques targets [FL, FR, RL, RR].
        """
        # 1. Update Engines (Calcul du couple MOTEUR réel)
        actual_motor_torques = []
        for engine, cmd in zip(self.engines, wheel_torques):
            actual_motor_torques.append(engine.step(cmd, dt))

        # 2. Compute Total Force
        # Force = (Sum Moteurs * Ratio) / Rayon
        total_motor_torque = sum(actual_motor_torques)
        total_wheel_torque = total_motor_torque * self.config.ratio_reduction
        total_drive_force = total_wheel_torque / self.config.wheel_radius
        
        # 3. Resistance Forces
        rho, S, Cx = 1.225, self.config.surface_frontale, self.config.cx
        m, g, crr = self.config.mass, 9.81, self.config.crr
        
        # Sens du mouvement (pour frottements)
        sign_v = np.sign(self.velocity) if abs(self.velocity) > 1e-3 else (np.sign(total_drive_force) if abs(total_drive_force) > 10 else 0)

        f_aero = 0.5 * rho * S * Cx * (self.velocity**2) * sign_v
        f_roll = m * g * crr * sign_v
        f_fric = (self.config.friction_torque / self.config.wheel_radius) * sign_v

        resistance = f_aero + f_roll + f_fric

        # 4. Newton's Second Law: F = ma
        # Gestion de l'arrêt complet (si force motrice < frottement statique)
        if abs(self.velocity) < 1e-3 and abs(total_drive_force) <= abs(resistance):
            self.acceleration = 0.0
            self.velocity = 0.0
        else:
            self.acceleration = (total_drive_force - resistance) / self.config.mass
            self.velocity += self.acceleration * dt
            self.position += self.velocity * dt