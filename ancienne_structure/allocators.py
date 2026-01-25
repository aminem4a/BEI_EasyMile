import numpy as np
from scipy.optimize import minimize_scalar, minimize
from dataclasses import dataclass

# On reprend les données brutes présentes dans vos fichiers pour l'entraînement des modèles
# (Idéalement, chargez-les depuis le CSV, mais pour l'exemple on utilise les arrays)
# ... [Insérer ici le chargement des données T_data, S_data, Z_data si non fait] ...

# ==============================================================================
# STRATÉGIE 1 : Basée sur allocation_couple.py (Fit Polynomial Unique)
# ==============================================================================
class StrategyPoly:
    """Encapsulation de EVTorqueOptimizerLS"""
    def __init__(self, t_data, s_data, z_data):
        # On reprend la logique de votre fichier: Fit polynomial P2
        # Z ~ p00 + p10*x + p01*y + p20*x^2 + p11*xy + p02*y^2
        self.coeffs = self._fit_surface(t_data, s_data, z_data)
        
    def _fit_surface(self, x, y, z):
        # Construction matrice A pour moindres carrés (Modele quadratique)
        ones = np.ones(x.shape)
        A = np.c_[ones, x, y, x**2, x*y, y**2]
        C, _, _, _ = np.linalg.lstsq(A, z, rcond=None)
        return C

    def predict_cosphi(self, T, S):
        c = self.coeffs
        # Modèle polynomial simple
        return c[0] + c[1]*T + c[2]*S + c[3]*T**2 + c[4]*T*S + c[5]*S**2

    def compute(self, T_global, speed_rpm):
        # Optimisation : trouver T_front qui maximise la somme des CosPhi
        # Contrainte : T_front + T_rear = T_global  => T_rear = T_global - T_front
        
        def objective(t_front):
            t_rear = (T_global/2.0) - t_front # Divisé par 2 car on raisonne par essieu ici?
            # Attention : T_global est souvent pour tout le véhicule. 
            # Si T_global est pour 4 moteurs, T_essieu = T_global / 2
            
            # On maximise le cosphi moyen (donc on minimise l'opposé)
            # Hypothèse: 2 moteurs à l'avant, 2 à l'arrière
            cp_f = self.predict_cosphi(t_front, speed_rpm)
            cp_r = self.predict_cosphi(t_rear, speed_rpm)
            return -(cp_f + cp_r) # On veut maximiser

        # Bornes (0 à T_global par essieu)
        res = minimize_scalar(objective, bounds=(0, T_global/2.0), method='bounded')
        
        t_front_opt = res.x
        t_rear_opt = (T_global/2.0) - t_front_opt
        
        # Retourne les couples pour [AVG, AVD, ARG, ARD]
        # On divise par 2 si t_front_opt est le couple de l'ESSIEU entier
        # Si votre optimiseur travaille en couple par ROUE, retirez la division par 2.
        return [t_front_opt, t_front_opt, t_rear_opt, t_rear_opt]


# ==============================================================================
# STRATÉGIE 2 : Basée sur allocation_2.py (Fit Piecewise + Optimisation)
# ==============================================================================
class StrategyPiecewise:
    """Fit séparé positif/négatif avec collage"""
    def __init__(self, t_data, s_data, z_data):
        # Séparation des données
        mask_pos = t_data >= 0
        self.coeffs_pos = self._fit_poly2(t_data[mask_pos], s_data[mask_pos], z_data[mask_pos])
        # Pour le négatif, on fit sur la valeur absolue du couple (symétrie supposée ou fit spécifique)
        mask_neg = t_data < 0
        if np.any(mask_neg):
            self.coeffs_neg = self._fit_poly2(np.abs(t_data[mask_neg]), s_data[mask_neg], z_data[mask_neg])
        else:
            self.coeffs_neg = self.coeffs_pos # Fallback

    def _fit_poly2(self, x, y, z):
        A = np.c_[np.ones_like(x), x, y, x**2, x*y, y**2]
        C, _, _, _ = np.linalg.lstsq(A, z, rcond=None)
        return C

    def get_cosphi(self, T, S):
        # Fonction piecewise
        T_abs = abs(T)
        c = self.coeffs_pos if T >= 0 else self.coeffs_neg
        val = c[0] + c[1]*T_abs + c[2]*S + c[3]*T_abs**2 + c[4]*T_abs*S + c[5]*S**2
        return val

    def compute(self, T_global, speed_rpm):
        # Même logique : trouver la répartition optimale statique
        def cost_func(t_front):
            t_rear = (T_global/2.0) - t_front
            return -(self.get_cosphi(t_front, speed_rpm) + self.get_cosphi(t_rear, speed_rpm))
        
        res = minimize_scalar(cost_func, bounds=(0, T_global/2.0), method='bounded')
        tf, tr = res.x, (T_global/2.0) - res.x
        return [tf, tf, tr, tr]


# ==============================================================================
# STRATÉGIE 3 : Basée sur allocation_3.py (Lissage Temporel / Rate Limiter)
# ==============================================================================
class StrategySmooth:
    """Optimisation avec mémoire pour lisser les variations"""
    def __init__(self, t_data, s_data, z_data, alpha_smooth=0.1):
        self.base_strategy = StrategyPiecewise(t_data, s_data, z_data)
        self.last_t_front = 0.0
        self.alpha = alpha_smooth # Poids de la pénalité de changement
        
    def compute(self, T_global, speed_rpm):
        # Objectif : Max(CosPhi) - Penalité(Changement)
        def cost_func(t_front):
            t_rear = (T_global/2.0) - t_front
            
            # Gain énergétique (négatif car on minimise)
            e_cost = -(self.base_strategy.get_cosphi(t_front, speed_rpm) + 
                       self.base_strategy.get_cosphi(t_rear, speed_rpm))
            
            # Pénalité de variation brusque (Smoothness)
            s_cost = self.alpha * (t_front - self.last_t_front)**2
            
            return e_cost + s_cost

        # On utilise minimize_scalar ou minimize (BFGS) si on veut être plus fin
        res = minimize_scalar(cost_func, bounds=(0, T_global/2.0), method='bounded')
        
        t_front_opt = res.x
        t_rear_opt = (T_global/2.0) - t_front_opt
        
        # Mise à jour de la mémoire pour le prochain pas de temps
        self.last_t_front = t_front_opt
        
        return [t_front_opt, t_front_opt, t_rear_opt, t_rear_opt]