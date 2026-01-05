import numpy as np
import pandas as pd
from scipy.interpolate import LinearNDInterpolator
from scipy.optimize import minimize_scalar

# ==========================================
# 1. DONNÉES 
# ==========================================
torque_data = np.array([16.583, 18.988, 29.45, 7.644, 32.495, 38.412, 33.2, 15.582,
                   40.85, 49.164, 13.289, 12.272, 60.795, 53.025, 117.075, 47.7,
                   28.035, 2.5, 10.735, 61.824, 23.381, 6.666, 36.181, 57.018,
                   10.094, 19.488, 48.609, 22.698, 7.505, 31.6, 17.85, 28.987, 53.346,
                   53.53, 52.38, 5.05, 9.595, 20.37, 23.808, 27.371, 69.319,
                   14.847, 24.096, 36.865, 72, 24.735, 31.824, 40.788, 44.352,
                   46.272, 62.304, 76.1, 76.538, 87.87, 83.125, 99.716, 28.684,
                   87.138, 13.802, 37.595, 42.33, 42.874, 52, 63.08, 67.32, 81.279,
                   86.632, 87.87, 95.76, 13.39, 20.055, 22.374, 23.816, 45.374,
                   44.175, 51.1, 62.296, 73.632, 83.712, 18.236, 31.556, 45.186,
                   46.53, 58.092, 65.52, 77, 93.66, 98.098, 29.498, 32.11, 38.178,
                   54.136, 61.74, 68.02, 16.463, 48.609, 39.382])
speed_data = np.array([2852.85, 1468.7, 1230, 2347.2, 956.87, 1260.72, 1607.55, 1890.05,
                  894.34, 596.55, 2195.96, 2371.65, 1432.6, 958.65, 1183.35, 1467.61,
                  1919.4, 4425.2, 2630.4, 938.08, 2238.6, 4092.43, 1821, 1187.76,
                  2979.2, 2493.63, 1795, 4122.82, 3636, 2126.05, 2834, 2802.8, 2609.84,
                  2687.36, 1705.25, 4382.65, 4425.2, 4066.02, 3392.64, 3498.66, 2209.66,
                  4309.52, 2710, 2126.05, 1183.05, 2882.88, 3473.91, 3079.65, 2269.44,
                  2558.78, 1783, 887, 1426.56, 1403.15, 1709.73, 1116.47, 2530.5, 1451.38,
                  3573.9, 2469.94, 2470, 2754.05, 2163.84, 1942.75, 1564.16, 1233.44,
                  1819.65, 550.2, 1517.19, 4387.76, 3763.2, 3991.55, 3063.06, 2096, 2922.3,
                  1975.05, 1842.67, 1926.6, 875.67, 4058.48, 2573.76, 2744.56, 2849,
                  2020.76, 2209.66, 2075.84, 1225.12, 1392.96, 3074.55, 2907.66, 2586.99,
                  2271.74, 2309, 1757, 3983.04, 2376.53, 2546.88])
cosphi_data = np.array([0.2037, 0.4264, 0.5562, 0.5916, 0.5723, 0.5917, 0.6014, 0.6534,
                   0.627, 0.6767, 0.735, 0.6745, 0.7056, 0.7154, 0.7008, 0.7178,
                   0.7828, 0.8085, 0.78, 0.8, 0.7872, 0.83, 0.83, 0.8051,
                   0.798, 0.8484, 0.8148, 0.867, 0.8772, 0.8858, 0.8787, 0.87, 0.9135,
                   0.8439, 0.8526, 0.836, 0.88, 0.88, 0.8976, 0.8976, 0.88,
                   0.8633, 0.8633, 0.9167, 0.8544, 0.855, 0.927, 0.855, 0.9,
                   0.927, 0.873, 0.918, 0.918, 0.927, 0.945, 0.945, 0.9464,
                   0.9373, 0.9476, 0.966, 0.966, 0.9292, 0.8832, 0.9108, 0.966, 0.9384,
                   0.8832, 0.9384, 0.92, 0.9021, 0.9021, 0.93, 0.9021, 0.9579,
                   0.9486, 0.9579, 0.9207, 0.9765, 0.8835, 0.893, 0.987, 0.9494,
                   0.893, 0.94, 0.9588, 0.987, 0.9024, 0.893, 0.912, 0.912, 0.9792,
                   0.96, 0.9312, 0.96, 0.9409, 0.9409, 0.9595])




# ==========================================
# 2. CLASSE D'OPTIMISATION
# ==========================================

class EVTorqueOptimizer:
    def __init__(self, T_data, V_data, Phi_data, max_motor_torque=120.0):
        """
        T_data: Vecteur couple (numpy array)
        V_data: Vecteur vitesse (numpy array)
        Phi_data: Vecteur cos(phi) (numpy array)
        max_motor_torque: Limite physique d'un seul moteur
        """
        self.max_torque = max_motor_torque
        
        # Préparation des points pour l'interpolateur (X, Y) -> Z
        # On empile T et V pour faire une matrice de colonnes [[t0, v0], [t1, v1], ...]
        points = np.column_stack((T_data, V_data))
        values = Phi_data
        
        # Création du modèle d'interpolation
        # fill_value=0 assure que si on demande une valeur hors du nuage connu, on a 0
        self.model = LinearNDInterpolator(points, values, fill_value=0)

    def get_efficiency(self, torque, speed):
        """ Retourne le cos(phi) interpolé """
        if torque <= 0.1: return 0.0 # Seuil bas pour éviter le bruit
        val = self.model(torque, speed)
        # L'interpolateur renvoie parfois des nan (not a number) si hors bornes
        return float(val) if not np.isnan(val) else 0.0

    def cost_function(self, c_front, c_global, speed, priority_coeff):
        """
        Fonction de coût à minimiser.
        Objectif : Maximiser (Priority * Eff_Avant + Eff_Arrière)
        """
        # Calcul du couple arrière (Contrainte : Somme totale respectée)
        # C_global = 2*C_front + 2*C_rear  =>  C_rear = (C_global/2) - C_front
        c_rear = (c_global / 2.0) - c_front
        
        # Contraintes physiques (Pénalité si violées)
        if c_rear < 0 or c_rear > self.max_torque:
            return 1e6 # Valeur très haute = solution rejetée
            
        # Récupération des efficacités
        eff_front = self.get_efficiency(c_front, speed)
        eff_rear = self.get_efficiency(c_rear, speed)
        
        # Calcul du score pondéré (Négatif car on minimise)
        # Plus priority_coeff est grand, plus on favorise le rendement avant
        score = - (priority_coeff * eff_front + eff_rear)
        return score

    def compute_optimal_torques(self, global_torque_cmd, current_speed, priority=10):
        """
        Calcule la répartition optimale.
        global_torque_cmd: Consigne totale (pour les 4 roues)
        current_speed: Vitesse actuelle (doit être dans la même unité que les données, ex: RPM)
        priority: Facteur de priorité pour l'avant (défaut 10)
        """
        
        # Définition des bornes de recherche pour UNE roue avant
        # 1. Ne peut pas être < 0
        # 2. Ne peut pas être > Max Moteur
        # 3. Doit laisser assez de couple pour que l'arrière ne dépasse pas son Max
        # 4. Ne peut pas dépasser la moitié du total (si c_arriere = 0)
        
        min_front = max(0, (global_torque_cmd - 2 * self.max_torque) / 2)
        max_front = min(self.max_torque, global_torque_cmd / 2)
        
        if min_front > max_front:
            return {"Erreur": "Couple demandé trop élevé pour les moteurs"}

        # Lancement de l'optimiseur
        res = minimize_scalar(
            self.cost_function,
            bounds=(min_front, max_front),
            args=(global_torque_cmd, current_speed, priority),
            method='bounded'
        )
        
        c_front_opt = res.x
        c_rear_opt = (global_torque_cmd / 2.0) - c_front_opt
        
        return {
            "Consigne_Globale": global_torque_cmd,
            "Vitesse": current_speed ,
            "Couple_Roue_AVANT": round(c_front_opt, 3),
            "Couple_Roue_ARRIERE": round(c_rear_opt, 3),
            "Eff_Avant": round(self.get_efficiency(c_front_opt, current_speed), 4),
            "Eff_Arriere": round(self.get_efficiency(c_rear_opt, current_speed), 4)
        }

# ==========================================
# 3. EXEMPLE D'UTILISATION
# ==========================================

# Initialisation 
optimizer = EVTorqueOptimizer(torque_data, speed_data, cosphi_data, max_motor_torque=120)

# Cas A : Demande faible (40 Nm total) -> On s'attend à tout sur l'avant
print("--- TEST 1 : Faible Charge ---")
resultat_1 = optimizer.compute_optimal_torques(global_torque_cmd=40, current_speed=2000, priority=10)
print(resultat_1)

print("\n--- TEST 2 : Forte Charge ---")
# Cas B : Demande forte (300 Nm total) -> L'avant sature, l'arrière doit aider
resultat_2 = optimizer.compute_optimal_torques(global_torque_cmd=300, current_speed=2000, priority=10)
print(resultat_2)

