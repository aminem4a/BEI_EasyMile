"""
TEST COMPLET : Comparaison des Stratégies d'Allocation
Basé sur un Mapping 'Least Squares' (CosPhi = Polynôme)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar

# =============================================================================
# 1. DONNÉES BRUTES (Issues de allocation_3.py)
# =============================================================================
T_data = np.array([16.583, 18.988, 29.45, 7.644, 32.495, 38.412, 33.2, 15.582, 40.85, 49.164,
                   13.289, 12.272, 60.795, 53.025, 117.075, 47.7, 28.035, 2.5, 10.735, 61.824,
                   23.381, 6.666, 36.181, 57.018, 10.094, 19.488, 48.609, 22.698, 7.505, 31.6,
                   17.85, 28.987, 53.346, 53.53, 52.38, 5.05, 9.595, 20.37, 23.808, 27.371,
                   69.319, 14.847, 24.096, 36.865, 72, 24.735, 31.824, 40.788, 44.352, 46.272,
                   62.304, 76.1, 76.538, 87.87, 83.125, 99.716, 28.684, 87.138, 13.802,
                   37.595, 42.33, 42.874, 52, 63.08, 67.32, 81.279, 86.632, 87.87, 95.76,
                   13.39, 20.055, 22.374, 23.816, 45.374, 44.175, 51.1, 62.296, 73.632,
                   83.712, 18.236, 31.556, 45.186, 46.53, 58.092, 65.52, 77, 93.66, 98.098,
                   29.498, 32.11, 38.178, 54.136, 61.74, 68.02, 16.463, 48.609, 39.382])

S_data = np.array([2852.85, 1468.7, 1230, 2347.2, 956.87, 1260.72, 1607.55, 1890.05, 894.34,
                   596.55, 2195.96, 2371.65, 1432.6, 958.65, 1183.35, 1467.61, 1919.4, 4425.2,
                   2630.4, 938.08, 2238.6, 4092.43, 1821, 1187.76, 2979.2, 2493.63, 1795,
                   4122.82, 3636, 2126.05, 2834, 2802.8, 2609.84, 2687.36, 1705.25, 4382.65,
                   4425.2, 4066.02, 3392.64, 3498.66, 2209.66, 4309.52, 2710, 2126.05,
                   1183.05, 2882.88, 3473.91, 3079.65, 2269.44, 2558.78, 1783, 887, 1426.56,
                   1403.15, 1709.73, 1116.47, 2530.5, 1451.38, 3573.9, 2469.94, 2470,
                   2754.05, 2163.84, 1942.75, 1564.16, 1233.44, 1819.65, 550.2, 1517.19,
                   4387.76, 3763.2, 3991.55, 3063.06, 2096, 2922.3, 1975.05, 1842.67,
                   1926.6, 875.67, 4058.48, 2573.76, 2744.56, 2849, 2020.76, 2209.66,
                   2075.84, 1225.12, 1392.96, 3074.55, 2907.66, 2586.99, 2271.74, 2309,
                   1757, 3983.04, 2376.53, 2546.88])

Z_data = np.array([0.2037, 0.4264, 0.5562, 0.5916, 0.5723, 0.5917, 0.6014, 0.6534, 0.627,
                   0.6767, 0.735, 0.6745, 0.7056, 0.7154, 0.7008, 0.7178, 0.7828, 0.8085,
                   0.78, 0.8, 0.7872, 0.83, 0.83, 0.8051, 0.798, 0.8484, 0.8148, 0.867,
                   0.8772, 0.8858, 0.8787, 0.87, 0.9135, 0.8439, 0.8526, 0.836, 0.88, 0.88,
                   0.8976, 0.8976, 0.88, 0.8633, 0.8633, 0.9167, 0.8544, 0.855, 0.927, 0.855,
                   0.9, 0.927, 0.873, 0.918, 0.918, 0.927, 0.945, 0.945, 0.9464, 0.9373,
                   0.9476, 0.966, 0.966, 0.9292, 0.8832, 0.9108, 0.966, 0.9384, 0.8832,
                   0.9384, 0.92, 0.9021, 0.9021, 0.93, 0.9021, 0.9579, 0.9486, 0.9579,
                   0.9207, 0.9765, 0.8835, 0.893, 0.987, 0.9494, 0.893, 0.94, 0.9588, 0.987,
                   0.9024, 0.893, 0.912, 0.912, 0.9792, 0.96, 0.9312, 0.96, 0.9409, 0.9409,
                   0.9595])

# =============================================================================
# 2. MAPPING : MOINDRES CARRÉS (Fit Polynômial)
# =============================================================================
def build_A_full(T, S):
    """Construit la matrice [T^2, S^2, TS, T, S, 1]"""
    T = np.asarray(T).ravel()
    S = np.asarray(S).ravel()
    return np.column_stack([T**2, S**2, T*S, T, S, np.ones_like(T)])

def fit_full(T, S, Z):
    """Résout Az = Z au sens des moindres carrés"""
    A = build_A_full(T, S)
    coeffs, _, _, _ = np.linalg.lstsq(A, Z, rcond=None)
    return coeffs

def predict_poly_efficiency(coeffs, T, S):
    """Prédiction Cos(phi) = A*T^2 + ..."""
    A = build_A_full(T, S)
    val = (A @ coeffs)
    return np.clip(val, 0.01, 1.0) # Clamp physique

# Calcul des Coefficients (Positif et Négatif)
coeffs_pos = fit_full(T_data, S_data, Z_data)
coeffs_neg = fit_full(-T_data, S_data, Z_data) # Hypothèse symétrique/miroir sur les données

print("✅ Mapping Moindres Carrés effectué.")
print(f"Coeffs POS: {coeffs_pos}")

# =============================================================================
# 3. MODÈLE D'ALLOCATION (Utilisant le Mapping)
# =============================================================================
class AllocatorComparison:
    def __init__(self, coeffs_pos, coeffs_neg):
        self.cp = coeffs_pos
        self.cn = coeffs_neg
        
    def get_efficiency(self, T, rpm):
        """Calcule Cos(phi) avec le polynôme approprié"""
        rpm = abs(rpm)
        if T >= 0:
            return float(predict_poly_efficiency(self.cp, [T], [rpm]))
        else:
            return float(predict_poly_efficiency(self.cn, [T], [rpm]))

    def get_loss(self, T, rpm):
        """
        Calcule la Perte (W) à partir de l'Efficacité prédite.
        Loss = P_meca * (1/eta - 1)
        """
        w = abs(rpm) * 2 * np.pi / 60
        Pm = T * w
        eta = self.get_efficiency(T, rpm)
        
        # Formule pertes moteur
        # P_elec = Pm / eta -> Perte = P_elec - Pm = Pm * (1/eta - 1)
        if Pm >= 0:
            return Pm * (1.0/eta - 1.0)
        else:
            # En géné : P_elec = Pm * eta -> Perte = |Pm| - |P_elec| = |Pm|*(1 - eta)
            return abs(Pm) * (1.0 - eta)

    # --- STRATÉGIES ---
    
    def strategy_inverse(self, T_req, rpm):
        """Répartition 50/50"""
        return T_req * 0.5

    def strategy_quadratic(self, T_req, rpm):
        """
        Optimisation Analytique.
        Si la courbe CosPhi est concave (Pont), les pertes sont 'concaves inverses' -> Tout ou Rien.
        """
        # Test simple : Extrémités vs Centre
        loss_single = self.get_loss(T_req, rpm) + self.get_loss(0, rpm)
        loss_split = self.get_loss(T_req*0.5, rpm) * 2
        
        if loss_single < loss_split:
            return T_req # 100% Avant
        else:
            return T_req * 0.5 # 50/50

    def strategy_piecewise(self, T_req, rpm):
        """Recherche exhaustive (Scan) pour minimiser les pertes"""
        ratios = np.linspace(0, 1, 21)
        best_loss = float('inf')
        best_Tf = T_req * 0.5
        
        for r in ratios:
            Tf = T_req * r
            Tr = T_req * (1 - r)
            loss = self.get_loss(Tf, rpm) + self.get_loss(Tr, rpm)
            if loss < best_loss:
                best_loss = loss
                best_Tf = Tf
        return best_Tf

    def strategy_smooth(self, T_req, rpm, prev_Tf):
        """Minimisation Pertes + Pénalité de variation"""
        ratios = np.linspace(0, 1, 21)
        best_cost = float('inf')
        best_Tf = T_req * 0.5
        
        prev_ratio = prev_Tf / (T_req + 1e-6) if abs(T_req) > 1 else 0.5
        alpha = 100.0 # Facteur de lissage
        
        for r in ratios:
            Tf = T_req * r
            Tr = T_req * (1 - r)
            loss = self.get_loss(Tf, rpm) + self.get_loss(Tr, rpm)
            cost = loss + alpha * (r - prev_ratio)**2
            if cost < best_cost:
                best_cost = cost
                best_Tf = Tf
        return best_Tf

# =============================================================================
# 4. GÉNÉRATION PROFIL TEST (Copie de allocation_3 pour autonomie)
# =============================================================================
def generate_test_reference(duration_s=60.0, dt=0.1, torque_peak_total=360.0):
    t = np.arange(0.0, duration_s + 1e-12, dt)
    speed_min, speed_max = np.min(S_data), np.max(S_data)
    speed = np.empty_like(t)
    torque = np.empty_like(t)
    for i, ti in enumerate(t):
        if ti < 15: speed[i] = speed_min + (speed_max - speed_min) * (ti / 15.0)
        elif ti < 30: speed[i] = speed_max
        elif ti < 45: speed[i] = speed_max - 0.65 * (speed_max - speed_min) * ((ti - 30.0) / 15.0)
        else:
            base = speed_min + 0.35 * (speed_max - speed_min)
            amp = 0.12 * (speed_max - speed_min)
            speed[i] = base + amp * np.sin(2.0 * np.pi * (ti - 45.0) / 6.0)
        if ti < 12: torque[i] = torque_peak_total * (ti / 12.0)
        elif ti < 28: torque[i] = 0.35 * torque_peak_total
        elif ti < 44: torque[i] = 0.35 * torque_peak_total - 0.90 * torque_peak_total * ((ti - 28.0) / 16.0)
        else: torque[i] = 0.10 * torque_peak_total * np.sin(2.0 * np.pi * (ti - 44.0) / 4.0)
    return t, speed, torque

# =============================================================================
# 5. SIMULATION & AFFICHAGE
# =============================================================================
def main():
    # 1. Setup
    t, v_rpm, trq_req = generate_test_reference()
    allocator = AllocatorComparison(coeffs_pos, coeffs_neg)
    
    strategies = ["Inverse", "Quadratic", "Piecewise", "Smooth"]
    results = {s: {'cos_phi_f': [], 'cos_phi_r': [], 'ratio': []} for s in strategies}
    
    print(f"Simulation sur {len(t)} points...")

    # 2. Boucle de Simulation
    for strat in strategies:
        prev_Tf = 0.0
        for i in range(len(t)):
            T_tot = trq_req[i]
            rpm = v_rpm[i]
            
            # Allocation
            if strat == "Inverse":
                Tf = allocator.strategy_inverse(T_tot, rpm)
            elif strat == "Quadratic":
                Tf = allocator.strategy_quadratic(T_tot, rpm)
            elif strat == "Piecewise":
                Tf = allocator.strategy_piecewise(T_tot, rpm)
            elif strat == "Smooth":
                Tf = allocator.strategy_smooth(T_tot, rpm, prev_Tf)
            
            Tr = T_tot - Tf
            prev_Tf = Tf
            
            # Calcul Cos Phi Résultant
            eta_f = allocator.get_efficiency(Tf, rpm)
            eta_r = allocator.get_efficiency(Tr, rpm)
            
            # Si couple nul, on considère cos_phi = 0 pour l'affichage (moteur éteint)
            if abs(Tf) < 0.1: eta_f = 0.0
            if abs(Tr) < 0.1: eta_r = 0.0
            
            results[strat]['cos_phi_f'].append(eta_f)
            results[strat]['cos_phi_r'].append(eta_r)
            
            ratio = Tf / (abs(T_tot)+1e-6) if abs(T_tot) > 1 else 0.5
            results[strat]['ratio'].append(ratio)

    # 3. Affichage Comparatif
    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    
    # Plot 1: Consigne
    axes[0].plot(t, trq_req, 'k-', lw=1.5, label="Consigne Couple Total")
    axes[0].set_title("1. Consigne de Couple")
    axes[0].set_ylabel("Nm")
    axes[0].grid(True)
    axes[0].legend()
    
    # Plot 2: Cos Phi Avant (Comparaison)
    styles = {'Inverse': ':', 'Quadratic': '--', 'Piecewise': '-.', 'Smooth': '-'}
    for s in strategies:
        axes[1].plot(t, results[s]['cos_phi_f'], label=f"Strat {s}", linestyle=styles.get(s, '-'))
    
    axes[1].set_title("2. Évolution du Cos Phi (Moteur Avant)")
    axes[1].set_ylabel("Cos Phi")
    axes[1].legend(loc='lower right')
    axes[1].grid(True)
    
    # Plot 3: Ratio
    for s in strategies:
        axes[2].plot(t, results[s]['ratio'], label=s, linestyle=styles.get(s, '-'))
        
    axes[2].set_title("3. Ratio de Répartition (Avant / Total)")
    axes[2].set_ylabel("Ratio (0=AR, 1=AV)")
    axes[2].set_xlabel("Temps (s)")
    axes[2].legend(loc='center right')
    axes[2].grid(True)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()