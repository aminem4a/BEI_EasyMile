from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


# -----------------------------
# Simulation (boucle temporelle)
# -----------------------------
@dataclass
class SimulationConfig:
    dt: float = 0.1
    duration: float = 60.0


# -----------------------------
# Véhicule (plateforme)
# -----------------------------
@dataclass
class VehicleConfig:
    """
    Paramètres plateforme EZDolly (longitudinal simplifié).
    Référence : Fiche Véhicule (page 9/12).
    """

    # Masses
    mass_empty_kg: float = 4900.0       # Poids à vide (kg) 
    payload_max_kg: float = 7000.0      # Charge maximale 7T (kg) 
    payload_kg: float = 0.0             # Charge actuelle simulée (kg)

    # Dimensions (utiles surtout pour extension latérale/validation)
    length_m: float = 6.4               
    width_m: float = 3.0                
    height_m: float = 3.0               

    # Châssis
    wheelbase_m: float = 4.8            # Empattement
    track_m: float = 2.2                # Voie 

    # Roues
    wheel_radius_m: float = 0.24        # Rayon roue 

    # Limites opérationnelles
    max_speed_kmh: float = 15.0         

    # Résistances 
    rolling_resistance_coeff: float = 0.0
    aero_drag_coeff: float = 0.0

    @property
    def mass_total_kg(self) -> float:
        """Masse totale simulée."""
        return self.mass_empty_kg + max(0.0, self.payload_kg)

    @property
    def max_speed_mps(self) -> float:
        """15 km/h -> 4.166... m/s"""
        return self.max_speed_kmh / 3.6


# -----------------------------
# Traction / moteurs / réducteur
# -----------------------------
@dataclass
class DrivetrainConfig:
    """
    Paramètres traction (4 moteurs asynchrones, un par roue).
    Référence : Traction + plaque signalétique 
    """

    n_motors: int = 4

    # Rapport de réduction moteur/essieu
    gear_ratio: float = 26.0            

    # Plaque signalétique (valeurs nominales)
    motor_nominal_power_w: float = 8000.0   
    motor_nominal_speed_rpm: float = 2200.0 
    motor_nominal_freq_hz: float = 55.0     
    motor_nominal_torque_nm: float = 34.3   
    motor_nominal_current_a: float = 109.0  
    motor_nominal_cosphi: float = 0.91      
    motor_poles: int = 4                    

    # Modèle simplifié de saturation couple
    motor_peak_torque_nm: float = 120.0

    def get_motor_peak_torque(self) -> float:
        if self.motor_peak_torque_nm is not None:
            return float(self.motor_peak_torque_nm)
        return float(self.motor_nominal_torque_nm)


# -----------------------------
# Régulateur vitesse -> couple total
# -----------------------------
@dataclass
class ControllerConfig:
    kp: float = 120.0
    ki: float = 2.0
    kd: float = 0.0
    integral_limit: float = 1e6


# -----------------------------
# Allocation couple total -> couples roues
# -----------------------------
@dataclass
class AllocationConfig:
    # pondérations (front/rear)
    a1: float = 0.7
    a2: float = 0.3
    allow_regen: bool = True

    # lissage
    lambda1: float = 5e-3
    lambda2: float = 5e-4
    dC_max: float = 5.0

    # mapping cos(phi)
    clip_01: bool = False

    # limite couple par roue (si None, dérivé des données map cosphi)
    Cmax_per_wheel: float = 5
