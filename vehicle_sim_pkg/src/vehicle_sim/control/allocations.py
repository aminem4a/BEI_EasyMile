import numpy as np
from typing import List
from scipy.optimize import minimize_scalar

# Données embarquées (Efficiency Map pour l'optimisation)
T_DATA = np.array([16.5, 18.9, 29.4, 7.6, 32.5, 38.4, 117.1, 47.7, 2.5, 61.8, -10.0, -30.0])
S_DATA = np.array([2852, 1468, 1230, 2347, 956, 1260, 1183, 1467, 4425, 1729, 2000, 2000])
Z_DATA = np.array([0.20, 0.42, 0.55, 0.59, 0.57, 0.59, 0.70, 0.72, 0.29, 0.82, 0.20, 0.40])

class TorqueAllocator:
    """
    Maps Total Torque Command -> Individual Wheel Torques using Optimization.
    """
    def __init__(self, alpha: float = 0.005, max_rate: float = 300.0):
        # Initialisation du modèle d'efficacité (Fit Polynomial)
        A = np.c_[np.ones_like(T_DATA), T_DATA, S_DATA, T_DATA**2, T_DATA*S_DATA, S_DATA**2]
        self.C = np.linalg.lstsq(A, Z_DATA, rcond=None)[0]
        
        # State for Rate Limiter
        self.last_tf = 0.0 # Couple avant précédent
        self.alpha = alpha
        self.max_rate = max_rate

    def _get_cp(self, t, s):
        # Evaluation CosPhi
        val = self.C[0] + self.C[1]*t + self.C[2]*s + self.C[3]*t**2 + self.C[4]*t*s + self.C[5]*s**2
        return max(0.1, min(0.99, val))

    def allocate(self, total_torque_cmd: float, speed_rpm: float = 1000.0, dt: float = 0.01) -> List[float]:
        """
        Splits total torque among 4 wheels optimizing CosPhi with Rate Limiting.
        """
        # Le couple total est en entrée (Moteurs cumulés)
        # On divise par 2 essieux pour l'algo
        half_total = total_torque_cmd / 2.0
        
        # Rate Limiter Boundaries
        d_max = self.max_rate * dt
        low = max(0, self.last_tf - d_max)
        high = min(half_total, self.last_tf + d_max)
        if low > high: low = high # Sécurité

        # Fonction Coût
        def cost(tf):
            tr = half_total - tf
            # Max CosPhi = Min(-CosPhi)
            gain = -(self._get_cp(tf, speed_rpm) + self._get_cp(tr, speed_rpm))
            smooth = self.alpha * (tf - self.last_tf)**2
            return gain + smooth

        # Optimisation
        try:
            res = minimize_scalar(cost, bounds=(low, high), method='bounded')
            tf_opt = res.x
        except:
            tf_opt = total_torque_cmd / 4.0 # Fallback

        self.last_tf = tf_opt
        tr_opt = half_total - tf_opt
        
        # Retourne [AVG, AVD, ARG, ARD]
        # Note: on divise par 2 car tf_opt est par essieu, et on a 2 roues par essieu
        # ATTENTION: Si l'optimiseur travaille en couple MOTEUR par essieu, 
        # et qu'on a 2 moteurs par essieu, le couple par moteur est tf_opt / 2 ?
        # Dans nos codes précédents, on répartissait sur 2 "super-moteurs".
        # Ici on a 4 moteurs. On va supposer Symétrie G/D.
        return [tf_opt/2, tf_opt/2, tr_opt/2, tr_opt/2]