import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
from dataclasses import dataclass

# =============================================================================
# 0. DONNÉES & MAPPING (Socle Commun)
# =============================================================================
# Données brutes (reprises de allocation_3.py)
T_data = np.array([16.583, 18.988, 29.45, 7.644, 32.495, 38.412, 33.2, 15.582, 40.85, 49.164, 13.289, 12.272, 60.795, 53.025, 117.075, 47.7, 28.035, 2.5, 10.735, 61.824, 23.381, 6.666, 36.181, 57.018, 10.094, 19.488, 48.609, 22.698, 7.505, 31.6, 17.85, 28.987, 53.346, 53.53, 52.38, 5.05, 9.595, 20.37, 23.808, 27.371, 69.319, 14.847, 24.096, 36.865, 72, 24.735, 31.824, 40.788, 44.352, 46.272, 62.304, 76.1, 76.538, 87.87, 83.125, 99.716, 28.684, 87.138, 13.802, 37.595, 42.33, 42.874, 52, 63.08, 67.32, 81.279, 86.632, 87.87, 95.76, 13.39, 20.055, 22.374, 23.816, 45.374, 44.175, 51.1, 62.296, 73.632, 83.712, 18.236, 31.556, 45.186, 46.53, 58.092, 65.52, 77, 93.66, 98.098, 29.498, 32.11, 38.178, 54.136, 61.74, 68.02, 16.463, 48.609, 39.382])
S_data = np.array([2852.85, 1468.7, 1230, 2347.2, 956.87, 1260.72, 1607.55, 1890.05, 894.34, 596.55, 2195.96, 2371.65, 1432.6, 958.65, 1183.35, 1467.61, 1919.4, 4425.2, 2630.4, 938.08, 2238.6, 4092.43, 1821, 1187.76, 2979.2, 2493.63, 1795, 4122.82, 3636, 2126.05, 2834, 2802.8, 2609.84, 2687.36, 1705.25, 4382.65, 4425.2, 4066.02, 3392.64, 3498.66, 2209.66, 4309.52, 2710, 2126.05, 1183.05, 2882.88, 3473.91, 3079.65, 2269.44, 2558.78, 1783, 887, 1426.56, 1403.15, 1709.73, 1116.47, 2530.5, 1451.38, 3573.9, 2469.94, 2470, 2754.05, 2163.84, 1942.75, 1564.16, 1233.44, 1819.65, 550.2, 1517.19, 4387.76, 3763.2, 3991.55, 3063.06, 2096, 2922.3, 1975.05, 1842.67, 1926.6, 875.67, 4058.48, 2573.76, 2744.56, 2849, 2020.76, 2209.66, 2075.84, 1225.12, 1392.96, 3074.55, 2907.66, 2586.99, 2271.74, 2309, 1757, 3983.04, 2376.53, 2546.88])
Z_data = np.array([0.2037, 0.4264, 0.5562, 0.5916, 0.5723, 0.5917, 0.6014, 0.6534, 0.627, 0.6767, 0.735, 0.6745, 0.7056, 0.7154, 0.7008, 0.7178, 0.7828, 0.8085, 0.78, 0.8, 0.7872, 0.83, 0.83, 0.8051, 0.798, 0.8484, 0.8148, 0.867, 0.8772, 0.8858, 0.8787, 0.87, 0.9135, 0.8439, 0.8526, 0.836, 0.88, 0.88, 0.8976, 0.8976, 0.88, 0.8633, 0.8633, 0.9167, 0.8544, 0.855, 0.927, 0.855, 0.9, 0.927, 0.873, 0.918, 0.918, 0.927, 0.945, 0.945, 0.9464, 0.9373, 0.9476, 0.966, 0.966, 0.9292, 0.8832, 0.9108, 0.966, 0.9384, 0.8832, 0.9384, 0.92, 0.9021, 0.9021, 0.93, 0.9021, 0.9579, 0.9486, 0.9579, 0.9207, 0.9765, 0.8835, 0.893, 0.987, 0.9494, 0.893, 0.94, 0.9588, 0.987, 0.9024, 0.893, 0.912, 0.912, 0.9792, 0.96, 0.9312, 0.96, 0.9409, 0.9409, 0.9595])

def build_A_full(T, S):
    T = np.asarray(T).ravel()
    S = np.asarray(S).ravel()
    return np.column_stack([T**2, S**2, T*S, T, S, np.ones_like(T)])

def fit_full(T, S, Z):
    A = build_A_full(T, S)
    coeffs, _, _, _ = np.linalg.lstsq(A, Z, rcond=None)
    return coeffs

def predict_full(coeffs, T, S):
    A = build_A_full(T, S)
    return (A @ coeffs).reshape(np.asarray(T).shape)

# Construction du mapping (Positif & Négatif)
coeff_pos = fit_full(T_data, S_data, Z_data)
coeff_neg = fit_full(-T_data, S_data, Z_data)

def cosphi_map(T, S):
    """Fonction de mapping commune : CosPhi = f(Couple_Par_Roue, Vitesse)"""
    T = np.asarray(T)
    S = np.asarray(S)
    Zp = predict_full(coeff_pos, T, S)
    Zn = predict_full(coeff_neg, T, S)
    # Clip à [0, 1] pour éviter les valeurs aberrantes du polynôme
    val = np.where(T >= 0, Zp, Zn)
    return np.clip(val, 0.0, 1.0)

# =============================================================================
# 1. CLASSES "ALLOCATION 2 & 3" (Fichiers Camarade)
# =============================================================================

class Allocator2_Impl:
    """Implémentation exacte de allocation_2.py"""
    def allocate(self, C_total, speed_rpm):
        a1, a2 = 0.7, 0.3
        Cmax = 117.0
        
        half = 0.5 * C_total
        lo = max(-Cmax, half - Cmax)
        hi = min(+Cmax, half + Cmax)
        
        def obj(Cav):
            Car = 0.5 * C_total - Cav
            eta_f = float(cosphi_map(Cav, speed_rpm))
            eta_r = float(cosphi_map(Car, speed_rpm))
            return -(a1 * eta_f + a2 * eta_r)

        res = minimize_scalar(obj, bounds=(lo, hi), method="bounded")
        Cav = float(res.x)
        Car = 0.5 * C_total - Cav
        return Cav, Car, cosphi_map(Cav, speed_rpm), cosphi_map(Car, speed_rpm)

class Allocator3_Impl:
    """Implémentation exacte de allocation_3.py (Lissage)"""
    def allocate(self, C_total, speed_rpm, Cav_prev, Cav_prev2):
        a1, a2 = 0.7, 0.3
        Cmax = 117.0
        lambda1, lambda2 = 5e-3, 5e-4
        
        half = 0.5 * C_total
        lo = max(-Cmax, half - Cmax)
        hi = min(+Cmax, half + Cmax)
        
        def obj(Cav):
            Car = 0.5 * C_total - Cav
            eta_f = float(cosphi_map(Cav, speed_rpm))
            eta_r = float(cosphi_map(Car, speed_rpm))
            score = a1 * eta_f + a2 * eta_r
            
            # Pénalités de lissage
            if Cav_prev is not None:
                score -= lambda1 * (Cav - Cav_prev)**2
            if Cav_prev is not None and Cav_prev2 is not None:
                dd = Cav - 2.0*Cav_prev + Cav_prev2
                score -= lambda2 * (dd**2)
            return -score

        res = minimize_scalar(obj, bounds=(lo, hi), method="bounded")
        Cav = float(res.x)
        Car = 0.5 * C_total - Cav
        return Cav, Car, cosphi_map(Cav, speed_rpm), cosphi_map(Car, speed_rpm)


# =============================================================================
# 2. CLASSES "ALLOCATION PACKAGE" (Test Indépendant précédent)
# =============================================================================

class AllocatorComparison_Package:
    """
    Implémentation des stratégies 'Piecewise' et 'Quadratic' telles que définies 
    dans votre package (test_comparaison_cosphi.py).
    NOTE: Ces stratégies décident la répartition du couple TOTAL (Tf, Tr).
    Pour interroger la map (qui est par roue), on divise par 2.
    """
    def allocate(self, strat, C_total, speed_rpm, prev_Tf=0.0):
        # T_req est le couple TOTAL véhicule (ex: 300 Nm)
        T_req = C_total
        rpm = abs(speed_rpm)
        Cmax_roue = 117.0
        
        # 1. Stratégie Quadratic (Analytique / Bang-Bang)
        if strat == "Quadratic":
            # Option 1: 50/50
            # Tf = T_req/2 -> Chaque roue AV prend T_req/4
            c_roue_50 = T_req / 4.0
            score_50 = 0.7*cosphi_map(c_roue_50, rpm) + 0.3*cosphi_map(c_roue_50, rpm)
            
            # Option 2: Tout Avant (si possible physiquement)
            # Tf = T_req -> Chaque roue AV prend T_req/2
            if (T_req/2.0) <= Cmax_roue:
                c_roue_100 = T_req / 2.0
                # On suppose l'arrière éteint (0 Nm)
                score_100 = 0.7*cosphi_map(c_roue_100, rpm) + 0.3*cosphi_map(0, rpm)
            else:
                score_100 = -999 # Impossible
                
            if score_100 > score_50:
                Tf = T_req # Tout sur l'essieu avant
            else:
                Tf = T_req * 0.5 # 50/50
                
        # 2. Stratégie Piecewise (Scan discret)
        elif strat == "Piecewise":
            ratios = np.linspace(0, 1, 41) 
            best_score = -float('inf')
            best_Tf = T_req * 0.5
            
            for r in ratios:
                Tf_try = T_req * r
                Tr_try = T_req * (1 - r)
                
                # Couples par roue correspondants
                c_av_roue = Tf_try / 2.0
                c_ar_roue = Tr_try / 2.0
                
                if abs(c_av_roue) <= Cmax_roue and abs(c_ar_roue) <= Cmax_roue:
                    score = 0.7*cosphi_map(c_av_roue, rpm) + 0.3*cosphi_map(c_ar_roue, rpm)
                    if score > best_score:
                        best_score = score
                        best_Tf = Tf_try
            Tf = best_Tf

        # 3. Stratégie Smooth (Scan + Pénalité)
        elif strat == "Smooth":
            ratios = np.linspace(0, 1, 41)
            best_score = -float('inf')
            best_Tf = T_req * 0.5
            alpha = 0.1 # Facteur lissage package
            
            prev_ratio = prev_Tf / (T_req + 1e-6) if abs(T_req) > 1 else 0.5
            
            for r in ratios:
                Tf_try = T_req * r
                Tr_try = T_req * (1 - r)
                c_av_roue = Tf_try / 2.0
                c_ar_roue = Tr_try / 2.0
                
                if abs(c_av_roue) <= Cmax_roue and abs(c_ar_roue) <= Cmax_roue:
                    raw_score = 0.7*cosphi_map(c_av_roue, rpm) + 0.3*cosphi_map(c_ar_roue, rpm)
                    # Pénalité sur la variation du ratio (comme dans le package)
                    score = raw_score - alpha * (r - prev_ratio)**2
                    if score > best_score:
                        best_score = score
                        best_Tf = Tf_try
            Tf = best_Tf
        else:
            Tf = T_req * 0.5

        # Conversion finales pour affichage (Cav par roue)
        Cav = Tf / 2.0
        Car = (T_req - Tf) / 2.0
        return Cav, Car, cosphi_map(Cav, rpm), cosphi_map(Car, rpm), Tf


# =============================================================================
# 3. EXÉCUTION & PLOTS
# =============================================================================
def main():
    # Génération Scénario
    duration_s = 60.0; dt = 0.1; torque_peak_total = 360.0
    t = np.arange(0.0, duration_s + 1e-12, dt)
    speed_min, speed_max = np.min(S_data), np.max(S_data)
    speed = np.empty_like(t)
    torque = np.empty_like(t)
    
    for i, ti in enumerate(t):
        if ti < 15: speed[i] = speed_min + (speed_max - speed_min) * (ti / 15.0)
        elif ti < 30: speed[i] = speed_max
        elif ti < 45: speed[i] = speed_max - 0.65 * (speed_max - speed_min) * ((ti - 30.0) / 15.0)
        else: speed[i] = speed_min + 0.35*(speed_max-speed_min) + 0.12*(speed_max-speed_min)*np.sin(2.0*np.pi*(ti-45.0)/6.0)
        if ti < 12: torque[i] = torque_peak_total * (ti / 12.0)
        elif ti < 28: torque[i] = 0.35 * torque_peak_total
        elif ti < 44: torque[i] = 0.35 * torque_peak_total - 0.90 * torque_peak_total * ((ti - 28.0) / 16.0)
        else: torque[i] = 0.10 * torque_peak_total * np.sin(2.0 * np.pi * (ti - 44.0) / 4.0)

    # Instance
    alloc2 = Allocator2_Impl()
    alloc3 = Allocator3_Impl()
    allocPkg = AllocatorComparison_Package()
    
    # Stockage
    res = {
        'Alloc 2': {'eta_f': [], 'eta_r': []},
        'Alloc 3': {'eta_f': [], 'eta_r': []},
        'Pkg Quad': {'eta_f': [], 'eta_r': []},
        'Pkg Piecewise': {'eta_f': [], 'eta_r': []},
        'Pkg Smooth': {'eta_f': [], 'eta_r': []}
    }
    
    cav_prev, cav_prev2 = None, None
    prev_Tf_smooth = 0.0
    
    print("Simulation...")
    for k in range(len(t)):
        C = torque[k]
        S = speed[k]
        
        # Groupe A (Fichiers)
        _, _, ef2, er2 = alloc2.allocate(C, S)
        res['Alloc 2']['eta_f'].append(ef2)
        res['Alloc 2']['eta_r'].append(er2)
        
        c3, _, ef3, er3 = alloc3.allocate(C, S, cav_prev, cav_prev2)
        res['Alloc 3']['eta_f'].append(ef3)
        res['Alloc 3']['eta_r'].append(er3)
        cav_prev2 = cav_prev; cav_prev = c3
        
        # Groupe B (Package)
        _, _, efQ, erQ, _ = allocPkg.allocate("Quadratic", C, S)
        res['Pkg Quad']['eta_f'].append(efQ)
        res['Pkg Quad']['eta_r'].append(erQ)
        
        _, _, efP, erP, _ = allocPkg.allocate("Piecewise", C, S)
        res['Pkg Piecewise']['eta_f'].append(efP)
        res['Pkg Piecewise']['eta_r'].append(erP)
        
        _, _, efS, erS, tfS = allocPkg.allocate("Smooth", C, S, prev_Tf_smooth)
        res['Pkg Smooth']['eta_f'].append(efS)
        res['Pkg Smooth']['eta_r'].append(erS)
        prev_Tf_smooth = tfS

    # --- FENÊTRE 1 : Alloc 2 & 3 ---
    fig1, axes1 = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    fig1.canvas.manager.set_window_title('Fenêtre 1 : Allocation Fichiers (2 & 3)')
    
    axes1[0].plot(t, res['Alloc 2']['eta_f'], 'b-', lw=1.5, label='Alloc 2 (Brut)')
    axes1[0].plot(t, res['Alloc 3']['eta_f'], 'c-', lw=2.0, label='Alloc 3 (Lissé)')
    axes1[0].set_title("Alloc 2 & 3 : Cos Phi AVANT")
    axes1[0].grid(True); axes1[0].legend()
    
    axes1[1].plot(t, res['Alloc 2']['eta_r'], 'b-', lw=1.5, label='Alloc 2')
    axes1[1].plot(t, res['Alloc 3']['eta_r'], 'c-', lw=2.0, label='Alloc 3')
    axes1[1].set_title("Alloc 2 & 3 : Cos Phi ARRIERE")
    axes1[1].set_xlabel("Temps (s)"); axes1[1].grid(True); axes1[1].legend()

    # --- FENÊTRE 2 : Package Comparison ---
    fig2, axes2 = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    fig2.canvas.manager.set_window_title('Fenêtre 2 : Allocation Package (Test Précédent)')
    
    axes2[0].plot(t, res['Pkg Quad']['eta_f'], 'r--', lw=1.5, label='Quadratic')
    axes2[0].plot(t, res['Pkg Piecewise']['eta_f'], 'm:', lw=2.0, label='Piecewise')
    axes2[0].plot(t, res['Pkg Smooth']['eta_f'], 'g-', lw=1.5, alpha=0.7, label='Smooth')
    axes2[0].set_title("Package Test : Cos Phi AVANT")
    axes2[0].grid(True); axes2[0].legend()
    
    axes2[1].plot(t, res['Pkg Quad']['eta_r'], 'r--', lw=1.5, label='Quadratic')
    axes2[1].plot(t, res['Pkg Piecewise']['eta_r'], 'm:', lw=2.0, label='Piecewise')
    axes2[1].plot(t, res['Pkg Smooth']['eta_r'], 'g-', lw=1.5, alpha=0.7, label='Smooth')
    axes2[1].set_title("Package Test : Cos Phi ARRIERE")
    axes2[1].set_xlabel("Temps (s)"); axes2[1].grid(True); axes2[1].legend()

    plt.show()

if __name__ == "__main__":
    main()