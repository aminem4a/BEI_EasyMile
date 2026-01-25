# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar

# ============================================================================
# 1. DONNÉES BRUTES (Extraites pour l'entraînement des modèles)
# ============================================================================
T_DATA = np.array([16.5, 18.9, 29.4, 7.6, 32.5, 38.4, 117.1, 47.7, 2.5, 61.8, -10.0, -30.0, -50.0]) # Ajout points négatifs fictifs pour robustesse
S_DATA = np.array([2852, 1468, 1230, 2347, 956, 1260, 1183, 1467, 4425, 1729, 2000, 2000, 2000])
Z_DATA = np.array([0.20, 0.42, 0.55, 0.59, 0.57, 0.59, 0.70, 0.72, 0.29, 0.82, 0.20, 0.40, 0.60])

# ============================================================================
# 2. CLASSES D'ALLOCATION (Les 3 Stratégies)
# ============================================================================

class AllocatorBase:
    def fit_poly2(self, x, y, z):
        # Ajustement surface quadratique : z = p0 + p1*x + p2*y + p3*x^2 + ...
        A = np.c_[np.ones_like(x), x, y, x**2, x*y, y**2]
        C, _, _, _ = np.linalg.lstsq(A, z, rcond=None)
        return C
    
    def eval_poly2(self, C, t, s):
        return C[0] + C[1]*t + C[2]*s + C[3]*t**2 + C[4]*t*s + C[5]*s**2

# --- STRATÉGIE 1 : Polynôme Unique (allocation_couple.py) ---
class Strat1_Poly(AllocatorBase):
    def __init__(self):
        self.C = self.fit_poly2(T_DATA, S_DATA, Z_DATA)
        
    def get_cosphi(self, t, s):
        return self.eval_poly2(self.C, t, s)
        
    def compute(self, T_total, speed):
        # Optimisation simple : Max(CosPhi_Front + CosPhi_Rear)
        def cost(tf):
            tr = (T_total/2.0) - tf # On raisonne par essieu (2 moteurs)
            return -(self.get_cosphi(tf, speed) + self.get_cosphi(tr, speed))
        
        res = minimize_scalar(cost, bounds=(0, T_total/2.0), method='bounded')
        tf = res.x
        return [tf, tf, (T_total/2.0)-tf, (T_total/2.0)-tf] # [AVG, AVD, ARG, ARD]

# --- STRATÉGIE 2 : Piecewise (allocation_2.py) ---
class Strat2_Piecewise(AllocatorBase):
    def __init__(self):
        # Fit séparé positif / négatif (ici simplifié avec les mêmes données)
        self.C_pos = self.fit_poly2(T_DATA, S_DATA, Z_DATA)
        self.C_neg = self.C_pos # Dans le vrai code, fit sur données négatives
        
    def get_cosphi(self, t, s):
        C = self.C_pos if t >= 0 else self.C_neg
        val = self.eval_poly2(C, abs(t), s)
        return max(0.1, min(0.99, val)) # Saturation physique
        
    def compute(self, T_total, speed):
        def cost(tf):
            tr = (T_total/2.0) - tf
            return -(self.get_cosphi(tf, speed) + self.get_cosphi(tr, speed))
        
        res = minimize_scalar(cost, bounds=(0, T_total/2.0), method='bounded')
        tf = res.x
        return [tf, tf, (T_total/2.0)-tf, (T_total/2.0)-tf]

# --- STRATÉGIE 3 : Smooth / Rate Limited (allocation_3.py) ---
class Strat3_Smooth(AllocatorBase):
    def __init__(self, alpha=0.005):
        self.base = Strat2_Piecewise()
        self.last_tf = 0.0
        self.alpha = alpha # Pénalité de variation
        
    def compute(self, T_total, speed):
        def cost(tf):
            tr = (T_total/2.0) - tf
            # 1. Gain énergétique
            gain = -(self.base.get_cosphi(tf, speed) + self.base.get_cosphi(tr, speed))
            # 2. Pénalité de changement brutal (Lissage)
            smooth = self.alpha * (tf - self.last_tf)**2
            return gain + smooth
            
        res = minimize_scalar(cost, bounds=(0, T_total/2.0), method='bounded')
        self.last_tf = res.x
        return [res.x, res.x, (T_total/2.0)-res.x, (T_total/2.0)-res.x]

# ============================================================================
# 3. MODÈLES PHYSIQUES (Véhicule + Moteur)
# ============================================================================
class VehiculeAvance:
    def __init__(self):
        self.m, self.r = 5637.0, 0.24
        self.v, self.friction = 0.0, 221.0
    def update(self, torques, dt):
        f_net = (sum(torques)/self.r) - (0.5*1.225*4*0.8*self.v**2 + self.m*9.81*0.015 + self.friction/self.r)
        self.v += (f_net/self.m)*dt
        return max(0, self.v)

class Motor2ndOrder:
    def __init__(self):
        self.state = np.array([0.0, 0.0])
    def step(self, u, dt): # omega=45, zeta=0.8
        self.state += np.array([self.state[1], 2025*(u - self.state[0]) - 72*self.state[1]]) * dt
        return self.state[0]

# ============================================================================
# 4. SIMULATION COMPARATIVE
# ============================================================================
def run_comparison():
    dt = 0.01
    temps = np.arange(0, 15.0, dt)
    ratio = 26.0
    
    # Scénario de Couple Global (Moteurs)
    # Rampe -> Palier -> Echelon -> Zéro
    T_profil = np.zeros_like(temps)
    for i, t in enumerate(temps):
        if t < 5: T_profil[i] = 100 * (t/5) # Rampe
        elif t < 8: T_profil[i] = 100       # Palier stable
        elif t < 12: T_profil[i] = 300      # Gros échelon (Accélération)
        else: T_profil[i] = 20              # Relâchement
        
    # Initialisation des 3 simulations parallèles
    sims = {
        "1. Poly (Baseline)":    {"algo": Strat1_Poly(),      "veh": VehiculeAvance(), "mot": [Motor2ndOrder() for _ in range(4)], "h_tf": [], "h_tr": [], "h_cp": []},
        "2. Piecewise (Split)":  {"algo": Strat2_Piecewise(), "veh": VehiculeAvance(), "mot": [Motor2ndOrder() for _ in range(4)], "h_tf": [], "h_tr": [], "h_cp": []},
        "3. Smooth (Lissée)":    {"algo": Strat3_Smooth(),    "veh": VehiculeAvance(), "mot": [Motor2ndOrder() for _ in range(4)], "h_tf": [], "h_tr": [], "h_cp": []}
    }
    
    print("Simulation comparative en cours...")
    
    for i, t in enumerate(temps):
        T_req = T_profil[i] # Couple total demandé aux 4 moteurs
        
        for key, sim in sims.items():
            # A. État actuel
            v_rpm = (sim["veh"].v / 0.24) * ratio * 9.549
            
            # B. Allocation
            # On passe T_req * 26 si l'allocateur attend du couple ROUE, 
            # ici on suppose que l'allocateur travaille en couple MOTEUR directement.
            refs = sim["algo"].compute(T_req * 4, v_rpm) # *4 car compute divise par 2 essieux
            
            # C. Dynamique Moteurs
            # refs est [AVG, AVD, ARG, ARD] -> on prend juste AVG (tf) et ARG (tr) pour stockage
            tf_ref = refs[0]
            tr_ref = refs[2]
            
            # On applique aux modèles
            t_reels = [m.step(r, dt) for m, r in zip(sim["mot"], refs)]
            
            # D. Véhicule
            sim["veh"].update([c*ratio for c in t_reels], dt)
            
            # E. Calcul Score CosPhi (Performance instantanée)
            # On utilise le modèle Piecewise comme "Vérité Terrain" pour évaluer le score
            cp_score = Strat2_Piecewise().get_cosphi(t_reels[0], v_rpm)
            
            # Stockage
            sim["h_tf"].append(t_reels[0])
            sim["h_tr"].append(t_reels[2])
            sim["h_cp"].append(cp_score)

    # --- AFFICHAGE ---
    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    
    # Plot 1: Répartition Couple Avant (AVG)
    for name, sim in sims.items():
        axes[0].plot(temps, sim["h_tf"], label=name, linewidth=2 if "Smooth" in name else 1)
    axes[0].plot(temps, T_profil, 'k:', alpha=0.3, label="Consigne Globale / 4")
    axes[0].set_ylabel("Couple Moteur AVANT (Nm)")
    axes[0].set_title("Comparaison : Allocation sur l'Essieu Avant")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Répartition Couple Arrière (ARG)
    for name, sim in sims.items():
        axes[1].plot(temps, sim["h_tr"], label=name)
    axes[1].set_ylabel("Couple Moteur ARRIÈRE (Nm)")
    axes[1].set_title("Comparaison : Allocation sur l'Essieu Arrière")
    axes[1].grid(True, alpha=0.3)
    
    # Plot 3: Performance (CosPhi estimé)
    for name, sim in sims.items():
        axes[2].plot(temps, sim["h_cp"], label=name)
    axes[2].set_ylabel("Cos(phi) Estimé")
    axes[2].set_xlabel("Temps (s)")
    axes[2].set_title("Performance Énergétique (Maximisation CosPhi)")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("comparatif_strategies.png")
    plt.show()

if __name__ == "__main__":
    run_comparison()