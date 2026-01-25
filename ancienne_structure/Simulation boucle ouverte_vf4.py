# -*- coding: utf-8 -*-
"""
SIMULATION EZDolly - CORRIGÉE
-----------------------------
1. Correction ValueError : Gestion robuste des bornes dans Strat3
2. Correction SyntaxWarning : Utilisation de chaînes brutes (r"...") pour LaTeX
3. Affichage : 2 Fenêtres distinctes
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar

# ============================================================================
# 1. DONNÉES
# ============================================================================
T_DATA = np.array([16.5, 18.9, 29.4, 7.6, 32.5, 38.4, 117.1, 47.7, 2.5, 61.8, -10.0, -30.0, -50.0])
S_DATA = np.array([2852, 1468, 1230, 2347, 956, 1260, 1183, 1467, 4425, 1729, 2000, 2000, 2000])
Z_DATA = np.array([0.20, 0.42, 0.55, 0.59, 0.57, 0.59, 0.70, 0.72, 0.29, 0.82, 0.20, 0.40, 0.60])

# ============================================================================
# 2. MODÈLES PHYSIQUES
# ============================================================================
class VehiculeAvance:
    def __init__(self):
        self.m, self.r = 5637.0, 0.24
        self.S, self.Cx, self.Crr = 4.0, 0.8, 0.015
        self.rho_air, self.g, self.friction = 1.225, 9.81, 221.0
        self.v, self.a = 0.0, 0.0

    def update(self, torques, dt):
        f_tract = sum(torques) / self.r
        # Gestion simplifiée du sens pour éviter oscillation à v=0
        if abs(self.v) > 1e-3: sign_v = np.sign(self.v)
        else: sign_v = np.sign(f_tract) if abs(f_tract) > 10 else 0

        f_res = 0.5*self.rho_air*self.S*self.Cx*(self.v**2)*sign_v + self.m*self.g*self.Crr*sign_v + (self.friction/self.r)*sign_v
        
        # PFD
        if abs(self.v) < 1e-3 and abs(f_tract) <= abs(f_res):
            self.a, self.v = 0.0, 0.0
        else:
            self.a = (f_tract - f_res) / self.m
            self.v += self.a * dt
            
        return max(0, self.v)

class MotorStrejc:
    def __init__(self, K=1.0, n=2, Ta=0.05, Tu=0.02, dt=0.01):
        self.K, self.n, self.Ta, self.dt = K, int(n), Ta, dt
        self.buffer = [0.0] * max(1, int(Tu/dt))
        self.states = np.zeros(self.n)
        
    def step(self, u):
        self.buffer.append(u * self.K)
        sig = self.buffer.pop(0)
        for i in range(self.n):
            dy = (sig - self.states[i]) / self.Ta
            self.states[i] += dy * self.dt
            sig = self.states[i]
        return self.states[-1]

# ============================================================================
# 3. STRATÉGIES D'ALLOCATION (CORRIGÉES)
# ============================================================================
class AllocatorBase:
    def fit(self): 
        A = np.c_[np.ones_like(T_DATA), T_DATA, S_DATA, T_DATA**2, T_DATA*S_DATA, S_DATA**2]
        return np.linalg.lstsq(A, Z_DATA, rcond=None)[0]
    def eval(self, C, t, s): 
        return C[0] + C[1]*t + C[2]*s + C[3]*t**2 + C[4]*t*s + C[5]*s**2

class Strat1_Poly(AllocatorBase):
    def __init__(self): self.C = self.fit()
    def compute(self, T, S, dt):
        res = minimize_scalar(lambda tf: -(self.eval(self.C, tf, S) + self.eval(self.C, T/2-tf, S)), 
                              bounds=(0, T/2), method='bounded')
        return [res.x, res.x, T/2-res.x, T/2-res.x]

class Strat2_Piecewise(AllocatorBase):
    def __init__(self): self.C = self.fit()
    def get_cp(self, t, s): 
        return max(0.1, min(0.99, self.eval(self.C, abs(t), s)))
    def compute(self, T, S, dt):
        res = minimize_scalar(lambda tf: -(self.get_cp(tf, S) + self.get_cp(T/2-tf, S)), 
                              bounds=(0, T/2), method='bounded')
        return [res.x, res.x, T/2-res.x, T/2-res.x]

class Strat3_SmoothRL(AllocatorBase):
    """Stratégie avec Rate Limiter (Corrigée pour éviter le crash)"""
    def __init__(self, alpha=0.005, max_rate=300.0):
        self.base = Strat2_Piecewise()
        self.last, self.alpha, self.rate = 0.0, alpha, max_rate
        
    def compute(self, T, S, dt):
        # 1. Bornes Physiques (La somme doit faire T)
        phys_min, phys_max = 0.0, T / 2.0
        
        # 2. Bornes Dynamiques (Rate Limiter)
        d = self.rate * dt
        dyn_min = self.last - d
        dyn_max = self.last + d
        
        # 3. Intersection des contraintes
        low = max(phys_min, dyn_min)
        high = min(phys_max, dyn_max)
        
        # 4. SÉCURITÉ ANTI-CRASH (Si T chute trop vite, phys_max < dyn_min)
        if low > high:
            # On privilégie la contrainte physique (respect de la consigne conducteur)
            low = high
            
        def cost(tf):
            tr = (T/2.0) - tf
            gain = -(self.base.get_cp(tf, S) + self.base.get_cp(tr, S))
            smooth = self.alpha * (tf - self.last)**2
            return gain + smooth
            
        res = minimize_scalar(cost, bounds=(low, high), method='bounded')
        self.last = res.x
        return [res.x, res.x, T/2-res.x, T/2-res.x]

# ============================================================================
# 4. SIMULATION ET AFFICHAGE
# ============================================================================
def run_simulation_split_windows():
    dt = 0.01
    temps = np.arange(0, 15.0, dt)
    ratio = 26.0
    
    # Scénario Couple Global
    T_profil = np.zeros_like(temps)
    for i, t in enumerate(temps):
        if t < 5: T_profil[i] = 200 * (t/5)
        elif t < 8: T_profil[i] = 200
        elif t < 12: T_profil[i] = 500
        else: T_profil[i] = 0 # Chute brutale ici
        
    sims = {
        "1. Poly":      {"algo": Strat1_Poly(),      "veh": VehiculeAvance(), "mot": [MotorStrejc(dt=dt) for _ in range(4)], "h_tf": [], "h_tr": [], "h_cp_av": [], "h_cp_ar": []},
        "2. Piecewise": {"algo": Strat2_Piecewise(), "veh": VehiculeAvance(), "mot": [MotorStrejc(dt=dt) for _ in range(4)], "h_tf": [], "h_tr": [], "h_cp_av": [], "h_cp_ar": []},
        "3. Smooth+RL": {"algo": Strat3_SmoothRL(),  "veh": VehiculeAvance(), "mot": [MotorStrejc(dt=dt) for _ in range(4)], "h_tf": [], "h_tr": [], "h_cp_av": [], "h_cp_ar": []}
    }
    
    evaluator = Strat2_Piecewise() # Référence de mesure

    print("Calcul en cours...")
    for i, t in enumerate(temps):
        T_req = T_profil[i]
        for _, sim in sims.items():
            v_rpm = (sim["veh"].v / 0.24) * ratio * 9.549
            
            # Appel de l'allocateur (Version corrigée)
            refs = sim["algo"].compute(T_req, v_rpm, dt)
            
            t_reels = [m.step(r) for m, r in zip(sim["mot"], refs)]
            sim["veh"].update([c*ratio for c in t_reels], dt)
            
            sim["h_tf"].append(t_reels[0])
            sim["h_tr"].append(t_reels[2])
            sim["h_cp_av"].append(evaluator.get_cp(t_reels[0], v_rpm))
            sim["h_cp_ar"].append(evaluator.get_cp(t_reels[2], v_rpm))

    # --- FIGURE 1 : COUPLES ---
    fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), num="Couples Moteurs")
    
    for name, sim in sims.items():
        ax1.plot(temps, sim["h_tf"], label=name, linewidth=2 if "Smooth" in name else 1)
    ax1.set_title("Couple Moteur AVANT")
    ax1.set_ylabel("Couple (Nm)")
    ax1.grid(True, alpha=0.3); ax1.legend(loc="upper left")
    
    for name, sim in sims.items():
        ax2.plot(temps, sim["h_tr"], label=name)
    ax2.set_title("Couple Moteur ARRIÈRE")
    ax2.set_ylabel("Couple (Nm)")
    ax2.set_xlabel("Temps (s)")
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

    # --- FIGURE 2 : COS PHI ---
    fig2, (ax3, ax4) = plt.subplots(2, 1, figsize=(10, 8), num="Efficacité CosPhi")
    
    for name, sim in sims.items():
        ax3.plot(temps, sim["h_cp_av"], label=name)
    ax3.set_title("Facteur de Puissance AVANT")
    ax3.set_ylabel(r"Cos($\phi$)") # Correction SyntaxWarning
    ax3.set_ylim(0, 1.05)
    ax3.grid(True, alpha=0.3); ax3.legend(loc="lower right")
    
    for name, sim in sims.items():
        ax4.plot(temps, sim["h_cp_ar"], label=name)
    ax4.set_title("Facteur de Puissance ARRIÈRE")
    ax4.set_ylabel(r"Cos($\phi$)") # Correction SyntaxWarning
    ax4.set_ylim(0, 1.05)
    ax4.set_xlabel("Temps (s)")
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
    print("Terminé.")

if __name__ == "__main__":
    run_simulation_split_windows()