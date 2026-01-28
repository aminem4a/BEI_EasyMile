import numpy as np

class Vehicle:
    """
    Modèle physique calibré (EZDolly).
    Logique : Couple (Input) -> Vitesse (Output).
    """
    def __init__(self):
        # --- PARAMÈTRES CALIBRÉS (Issus de Class_vehiculetest.py) ---
        self.m = 4900.0        # Masse (kg)
        self.r = 0.24           # Rayon roue (m)
        self.S = 4.0            # Surface frontale (m^2)
        self.Cx = 0.8           # Coeff aéro
        self.Crr = 0.015        # Coeff roulement
        self.rho_air = 1.225    # Densité air
        self.g = 9.81           
        
        # Couple de frottement sec constant (Pertes mécaniques transmission)
        self.friction_torque = 221.0 

        # État initial du véhicule
        self.v = 0.0 # Vitesse (m/s)
        self.x = 0.0 # Position (m)
        self.a = 0.0 # Accélération (m/s^2)

    def reset(self):
        """Réinitialise l'état du véhicule."""
        self.v = 0.0
        self.x = 0.0
        self.a = 0.0

    def update(self, total_torque_at_wheels, dt, slope_rad=0.0):
        """
        Calcule la nouvelle vitesse en fonction du couple appliqué.
        
        Args:
            total_torque_at_wheels (float): Somme des couples des 4 roues (Nm).
            dt (float): Pas de temps (s).
            slope_rad (float): Pente en radians.
        
        Returns:
            v (float): Nouvelle vitesse (m/s)
            a (float): Accélération instantanée (m/s^2)
            x (float): Position (m)
        """
        # 1. Force de Traction (F = C / r)
        f_tract = total_torque_at_wheels / self.r
        
        # 2. Détermination du sens du mouvement
        # Si on roule déjà, on prend le signe de la vitesse
        if abs(self.v) > 1e-3:
            sign_v = np.sign(self.v)
        # Sinon, si on est à l'arrêt, on regarde si la force de traction veut nous faire bouger
        else:
            sign_v = np.sign(f_tract) if abs(f_tract) > 0 else 0

        # 3. Calcul des Forces Résistives
        # Aéro : 0.5 * rho * S * Cx * v^2
        f_aero = 0.5 * self.rho_air * self.S * self.Cx * (self.v**2) * sign_v
        
        # Roulement : m * g * Crr * cos(pente)
        f_roll = self.m * self.g * self.Crr * np.cos(slope_rad) * sign_v
        
        # Pente : m * g * sin(pente)
        f_slope = self.m * self.g * np.sin(slope_rad)
        
        # Frottement Mécanique (Transmission) : Constant tant qu'on bouge
        f_mech = (self.friction_torque / self.r) * sign_v

        # Somme des résistances
        f_resist = f_aero + f_roll + f_slope + f_mech

        # 4. Principe Fondamental de la Dynamique (PFD) : ma = Somme(F)
        # a = (Force Motrice - Forces Résistantes) / m
        self.a = (f_tract - f_resist) / self.m
        
        # 5. Intégration d'Euler
        self.v += self.a * dt
        self.x += self.v * dt

        # 6. Gestion de l'arrêt (Clamp)
        # Si la vitesse est très faible et que la traction n'est pas suffisante pour vaincre les résistances statiques
        if abs(self.v) < 0.01 and abs(f_tract) < abs(f_resist):
            self.v = 0.0
            self.a = 0.0

        return self.v, self.a, self.x