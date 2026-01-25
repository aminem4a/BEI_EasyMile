# -*- coding: utf-8 -*-
"""
SIMULATION BOUCLE OUVERTE - EZDolly (Version Complète)
------------------------------------------------------
Intègre :
1. Modèle Véhicule Physique (Masse, Aero, Frottements)
2. Modèle Moteur Strejc (Retard pur + 2nd ordre)
3. Comparaison des 3 Stratégies d'Allocation (Poly, Piecewise, Smooth+RL)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar

# ============================================================================
# 1. DONNÉES BRUTES (Pour entrainement des Allocateurs)
# ============================================================================
# T=Couple, S=Vitesse, Z=CosPhi (Données issues de allocation_couple.py)
T_DATA = np.array([16.5, 18.9, 29.4, 7.6, 32.5, 38.4, 117.1, 47.7, 2.5, 61.8, -10.0, -30.0, -50.0])
S_DATA = np.array([2852, 1468, 1230, 2347, 956, 1260, 1183, 1467, 4425, 1729, 2000, 2000, 2000])
Z_DATA = np.array([0.20, 0.42, 0.55, 0.59, 0.57, 0.59, 0.70, 0.72, 0.29, 0.82, 0.20, 0.40, 0.60])

# ============================================================================
# 2. MODÈLE VÉHICULE (Class_vehiculeoff.py)
# ============================================================================
class VehiculeAvance:
    def __init__(self):
        # Paramètres calibrés
        self.m = 5637.0        # Masse (kg)
        self.r = 0.24          # Rayon roue (m)
        self.S = 4.0           # Surface frontale
        self.Cx = 0.8          # Coeff aero
        self.Crr = 0.015       # Coeff roulement
        self.rho_air = 1.225
        self.g = 9.81
        self.friction_torque = 221.0 # Frottement mécanique (Nm aux roues)

        self.v = 0.0
        self.x = 0.0
        self.a = 0.0

    def update(self, torques_roues, dt, slope_rad=0.0):
        # 1. Traction
        f_tract = sum(torques_roues) / self.r
        
        # 2. Sens
        if abs(self.v) > 1e-3: sign_v = np.sign(self.v)
        else: sign_v = np.sign(f_tract) if abs(f_tract) > 0 else 0

        # 3. Résistances
        f_aero = 0.5 * self.rho_air * self.S * self.Cx * (self.v**2) * sign_v
        f_roll = self.m * self.g * self.Crr * np.cos(slope_rad) * sign_v
        f_slope = self.m * self.g * np.sin(slope_rad)
        f_fric = (self.friction_torque / self.r) * sign_v

        # 4. PFD
        f_net = f_tract - (f_aero + f_roll + f_slope + f_fric)
        
        # Gestion frottement statique (seuil de démarrage)
        if abs(self.v) < 1e-3 and abs(f_tract) < (abs(f_roll) + abs(f_slope) + abs(f_fric)):
            self.a = 0.0
            self.v = 0.0
        else:
            self.a = f_net / self.m
            self.v += self.a * dt
            
        return max(0, self.v) # On suppose marche avant pour la simu

# ============================================================================
# 3. MODÈLE MOTEUR STREJC (MODELISATION_MOTEUR.py)
# ============================================================================
class MotorStrejc:
    """Modèle : G(s) = K * exp(-Tu*s) / (1 + Ta*s)^n"""
    def __init__(self, K=1.0, n=2, Ta=0.05, Tu=0.02, dt=0.01):
        self.K = K
        self.n = int(n)
        self.Ta = Ta
        self.dt = dt
        
        # Buffer pour le retard pur (Tu)
        steps_delay = int(Tu / dt)
        self.buffer = [0.0] * max(1, steps_delay)
        
        # États pour la cascade de filtres (Ordre n)
        self.states = np.zeros(self.n)
        
    def step(self, u):
        # 1. Gain + Retard
        self.buffer.append(u * self.K)
        u_delayed = self.buffer.pop(0)
        
        # 2. Cascade de n filtres du 1er ordre
        input_signal = u_delayed
        for i in range(self.n):
            y_prev = self.states[i]
            # dy/dt = (u - y) / Ta
            dy = (input_signal - y_prev) / self.Ta
            self.states[i] += dy * self.dt
            input_signal = self.states[i] # La sortie i devient l'entrée i+1
            
        return self.states[-1]

# ============================================================================
# 4. STRATÉGIES D'ALLOCATION
# ============================================================================
class AllocatorBase:
    def fit_poly2(self, x, y, z):
        A = np.c_[np.ones_like(x), x, y, x**2, x*y, y**2]
        C, _, _, _ = np.linalg.lstsq(A, z, rcond=None)
        return C
    def eval_poly2(self, C, t, s):
        return C[0] + C[1]*t + C[2]*s + C[3]*t**2 + C[4]*t*s + C[5]*s**2

# --- Stratégie 1 : Baseline (allocation_couple.py) ---
class Strat1_Poly(AllocatorBase):
    def __init__(self):
        self.C = self.fit_poly2(T_DATA, S_DATA, Z_DATA)
    def compute(self, T_total, speed, dt=None):
        def cost(tf):
            return -(self.eval_poly2(self.C, tf, speed) + self.eval_poly2(self.C, (T_total/2.0)-tf, speed))
        res = minimize_scalar(cost, bounds=(0, T_total/2.0), method='bounded')
        tf = res.x
        return [tf, tf, (T_total/2.0)-tf, (T_total/2.0)-tf]

# --- Stratégie 2 : Piecewise (allocation_2.py) ---
class Strat2_Piecewise(AllocatorBase):
    def __init__(self):
        mask = T_DATA >= 0
        self.C_pos = self.fit_poly2(T_DATA[mask], S_DATA[mask], Z_DATA[mask])
        self.C_neg = self.C_pos # Simplification ici
    def get_cp(self, t, s):
        C = self.C_pos if t >= 0 else self.C_neg
        return self.eval_poly2(C, abs(t), s)
    def compute(self, T_total, speed, dt=None):
        def cost(tf):
            return -(self.get_cp(tf, speed) + self.get_cp((T_total/2.0)-tf, speed))
        res = minimize_scalar(cost, bounds=(0, T_total/2.0), method='bounded')
        tf = res.x
        return [tf, tf, (T_total/2.0)-tf, (T_total/2.0)-tf]

# --- Stratégie 3 : Smooth + Rate Limiter (allocation_3.py) ---
class Strat3_SmoothRL(AllocatorBase):
    def __init__(self, alpha=0.005, max_rate=300.0):
        self.base = Strat2_Piecewise()
        self.last_tf = 0.0
        self.alpha = alpha
        self.max_rate = max_rate # Nm/s
    def compute(self, T_total, speed, dt):
        # Rate Limiter : Bornes dynamiques
        d_max = self.max_rate * dt
        low = max(0, self.last_tf - d_max)
        high = min(T_total/2.0, self.last_tf + d_max)
        if low > high: low, high = 0, T_total/2.0 # Sécurité
        
        def cost(tf):
            tr = (T_total/2.0) - tf
            gain = -(self.base.get_cp(tf, speed) + self.base.get_cp(tr, speed))
            smooth = self.alpha * (tf - self.last_tf)**2
            return gain + smooth
            
        res = minimize_scalar(cost, bounds=(low, high), method='bounded')
        self.last_tf = res.x
        return [res.x, res.x, (T_total/2.0)-res.x, (T_total/2.0)-res.x]

# ============================================================================
# 5. SIMULATION
# ============================================================================
def run_simulation_finale():
    # Temps
    dt = 0.01
    temps = np.arange(0, 15.0, dt)
    ratio = 26.0
    
    # Scénario Couple Global Moteur (Profil test)
    # 0-5s: Montée progressive (Rampe)
    # 5-8s: Palier
    # 8-12s: Échelon brutal (Test Rate Limiter)
    # 12-15s: Relâchement (Roue libre, test véhicule)
    T_profil = np.zeros_like(temps)
    for i, t in enumerate(temps):
        if t < 5: T_profil[i] = 200 * (t/5)
        elif t < 8: T_profil[i] = 200
        elif t < 12: T_profil[i] = 500
        else: T_profil[i] = 0
        
    sims = {
        "1. Poly":      {"algo": Strat1_Poly(),      "veh": VehiculeAvance(), "mot": [MotorStrejc(dt=dt) for _ in range(4)], "h_tf": [], "h_tr": [], "h_v": []},
        "2. Piecewise": {"algo": Strat2_Piecewise(), "veh": VehiculeAvance(), "mot": [MotorStrejc(dt=dt) for _ in range(4)], "h_tf": [], "h_tr": [], "h_v": []},
        "3. Smooth+RL": {"algo": Strat3_SmoothRL(),  "veh": VehiculeAvance(), "mot": [MotorStrejc(dt=dt) for _ in range(4)], "h_tf": [], "h_tr": [], "h_v": []}
    }
    
    print("Simulation en cours...")
    
    for i, t in enumerate(temps):
        T_global_req = T_profil[i]
        
        for name, sim in sims.items():
            # A. Vitesse actuelle (RPM Moteur)
            v_rpm = (sim["veh"].v / 0.24) * ratio * 9.549
            
            # B. Allocation
            # On passe T_req * 4 pour simuler le couple total demandé (si allocateur divise par 2 essieux)
            # Hypothèse: l'allocateur gère la répartition sur T_total
            refs = sim["algo"].compute(T_global_req, v_rpm, dt)
            
            # C. Moteurs (Strejc)
            t_reels = [m.step(r) for m, r in zip(sim["mot"], refs)]
            
            # D. Véhicule
            t_roues = [c * ratio for c in t_reels]
            sim["veh"].update(t_roues, dt)
            
            # Logs
            sim["h_tf"].append(t_reels[0]) # Couple Moteur AVG
            sim["h_tr"].append(t_reels[2]) # Couple Moteur ARG
            sim["h_v"].append(sim["veh"].v * 3.6) # km/h

    # --- AFFICHAGE ---
    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    
    # 1. Couple Avant
    for name, sim in sims.items():
        axes[0].plot(temps, sim["h_tf"], label=name, linewidth=2 if "Smooth" in name else 1)
    axes[0].set_ylabel("Couple Moteur AVANT (Nm)")
    axes[0].set_title("Allocation : Essieu Avant")
    axes[0].grid(True, alpha=0.3); axes[0].legend()
    
    # 2. Couple Arrière
    for name, sim in sims.items():
        axes[1].plot(temps, sim["h_tr"], label=name)
    axes[1].set_ylabel("Couple Moteur ARRIÈRE (Nm)")
    axes[1].set_title("Allocation : Essieu Arrière")
    axes[1].grid(True, alpha=0.3)
    
    # 3. Vitesse Véhicule
    for name, sim in sims.items():
        axes[2].plot(temps, sim["h_v"], label=name)
    axes[2].set_ylabel("Vitesse (km/h)")
    axes[2].set_xlabel("Temps (s)")
    axes[2].set_title("Réponse Véhicule (Dynamique Longitudinale)")
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("simulation_openloop_finale.png")
    plt.show()
    print("Terminé. Voir 'simulation_openloop_finale.png'")

if __name__ == "__main__":
    run_simulation_finale()