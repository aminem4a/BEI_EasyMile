from dataclasses import dataclass

@dataclass
class VehicleConfig:
    # --- Paramètres EZDolly ---
    mass: float = 5637.0         # kg
    wheel_radius: float = 0.24   # m
    surface_frontale: float = 4.0 # m2
    cx: float = 0.8              # Coeff aero
    crr: float = 0.015           # Coeff roulement
    ratio_reduction: float = 26.0 # Rapport de réduction (Moteur -> Roue)
    friction_torque: float = 221.0 # Nm (Frottement sec aux roues)
    
    # --- Paramètres Moteurs (Strejc) ---
    motor_max_torque: float = 180.0 # Nm (Nominal moteur ~)
    strejc_k: float = 1.0
    strejc_n: int = 2
    strejc_ta: float = 0.05
    strejc_tu: float = 0.02

@dataclass
class SimulationConfig:
    dt: float = 0.01      # Time step in seconds
    duration: float = 15.0 # Total simulation time