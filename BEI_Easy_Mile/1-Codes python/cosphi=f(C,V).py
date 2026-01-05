import numpy as np
import pandas as pd
from scipy.interpolate import LinearNDInterpolator
from scipy.optimize import minimize_scalar

# Données
torque = np.array([16.583, 18.988, 29.45, 7.644, 32.495, 38.412, 33.2, 15.582,
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
speed = np.array([2852.85, 1468.7, 1230, 2347.2, 956.87, 1260.72, 1607.55, 1890.05,
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
cosphi = np.array([0.2037, 0.4264, 0.5562, 0.5916, 0.5723, 0.5917, 0.6014, 0.6534,
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




# A. Vérification de sécurité (dimensions)
assert len(torque) == len(speed) == len(cosphi), "Erreur : Les tableaux n'ont pas la même taille !"







class TorqueAllocator:
    def __init__(self, c_data, v_data, eff_data):
        # Création de la cartographie d'efficacité
        self.efficiency_map = LinearNDInterpolator(list(zip(c_data, v_data)), eff_data, fill_value=0)
        
        # --- RÉGLAGE DE LA PRIORITÉ ---
        # Ce paramètre décide à quel point on déteste utiliser les roues arrières.
        # 0.5 : L'arrière s'active si le gain d'efficacité est visible.
        # 2.0 : L'arrière ne s'active presque jamais (Traction pure).
        self.penalty_rear_usage = 0.5 

    def get_efficiency(self, torque, speed):
        # On évite les valeurs nulles ou négatives pour la physique
        if torque <= 0.1: return 0.01 
        return float(self.efficiency_map(torque, speed))

    def objective_function(self, c_front, c_total_req, speed):
        """
        On cherche à MINIMISER cette fonction.
        """
        # 1. Calcul des couples
        # c_front = couple sur UNE roue avant
        # c_total_req = couple total pour les 4 roues
        c_rear = (c_total_req / 2.0) - c_front
        
        # Contrainte physique : pas de couple négatif ici
        if c_rear < 0: return 1e9

        # 2. Calcul de l'efficacité (Le but principal)
        eff_front = self.get_efficiency(c_front, speed)
        eff_rear = self.get_efficiency(c_rear, speed)
        
        # On maximise la moyenne des efficacités (donc on minimise l'opposé)
        # On multiplie par 10 pour que ce soit l'ordre de grandeur principal
        avg_efficiency_cost = -10.0 * ((eff_front + eff_rear) / 2.0)
        
        # 3. La "Taxe" sur les roues arrières (Priorité Avant)
        # Plus on met de couple à l'arrière, plus le coût augmente.
        # Cela force l'optimiseur à charger l'avant au maximum (c_front grand -> c_rear petit).
        bias_cost = self.penalty_rear_usage * c_rear
        
        # NOTE : Pas de terme "smoothness" ici, comme demandé.
        
        return avg_efficiency_cost + bias_cost

    def compute_allocation(self, total_torque_cmd, speed):
        # On cherche c_front entre 0 et le max possible (qui est total/2)
        max_possible_front = total_torque_cmd / 2.0
        
        result = minimize_scalar(
            self.objective_function,
            bounds=(0, max_possible_front),
            args=(total_torque_cmd, speed),
            method='bounded'
        )
        
        c_front_opt = result.x
        c_rear_opt = max_possible_front - c_front_opt
        
        return {
            "T_fl": c_front_opt, "T_fr": c_front_opt,
            "T_rl": c_rear_opt,  "T_rr": c_rear_opt
        }











# --- EXEMPLE D'UTILISATION ---

# 1. Génération de fausses données (Data Map) pour l'exemple
# Supposons que les moteurs sont plus efficaces à moyen régime
#C_samples = np.random.uniform(0, 100, 500) # Couples entre 0 et 100 Nm
#V_samples = np.random.uniform(0, 100, 500) # Vitesse rad/s
# Fonction bidon pour cos(phi) : eff augmente avec la vitesse, baisse si couple trop haut
#Eff_samples = 0.8 + 0.1*np.sin(V_samples/20) - 0.001*(C_samples-50)**2 
#Eff_samples = np.clip(Eff_samples, 0.5, 0.99) # borné entre 0.5 et 0.99

# 2. Initialisation de l'allocateur
allocator = TorqueAllocator(torque, speed, cosphi)

# 3. Simulation d'un scénario
print(f"{'Temps':<10} | {'Consigne Totale':<15} | {'Vitesse':<10} | {'C_Avant (x2)':<15} | {'C_Arrière (x2)':<15}")
print("-" * 75)

current_speed = 5000 # rad/s
# On fait varier la consigne de couple total
scenarios = [0, 200, 200, 200, 200] # Consigne totale qui change

for t, global_torque_req in enumerate(scenarios):
    res = allocator.compute_allocation(global_torque_req, current_speed)
    
    c_front_total = res['T_fl'] + res['T_fr']
    c_rear_total = res['T_rl'] + res['T_rr']
    
    print(f"t={t:<8} | {global_torque_req:<15.1f} | {current_speed:<10.1f} | {c_front_total:<15.2f} | {c_rear_total:<15.2f}")