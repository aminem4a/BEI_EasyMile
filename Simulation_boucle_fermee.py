# -*- coding: utf-8 -*-
"""
BEI EZDolly - Simulation Dynamique en Boucle Fermée
Inclus : Chargement efficiency_map.csv, Régulateur PI et Modèle 2nd Ordre.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict

# ============================================================================
# 1. CHARGEMENT DES DONNÉES 
# ============================================================================

def charger_donnees_moteur(file_name: str) -> pd.DataFrame:
    """Charge le fichier CSV pour extraire les paramètres de fonctionnement."""
    try:
        df = pd.read_csv(file_name, engine='python', on_bad_lines='skip', 
                         skipinitialspace=True, encoding='latin1', sep=None)
        
        df.columns = df.columns.astype(str).str.replace(r'#|\(|\)', '', regex=True).str.strip()
        mapping = {df.columns[0]: 'SpeedRPM', df.columns[1]: 'TorqueNm', df.columns[2]: 'CosPhi'}
        df = df.rename(columns=mapping)
        
        for col in ['SpeedRPM', 'TorqueNm', 'CosPhi']:
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(r'\[|\]|;', '', regex=True), errors='coerce')
        
        df = df.dropna(subset=['SpeedRPM', 'TorqueNm', 'CosPhi'])
        print(f"✓ Données de '{file_name}' chargées avec succès.")
        return df
    except Exception as e:
        print(f"❌ Erreur lors du chargement : {e}")
        return None

# ============================================================================
# 2. MODÈLES DE CONTRÔLE ET DYNAMIQUE
# ============================================================================

class PIController:
    """Régulateur Proportionnel Intégral avec Anti-Windup."""
    def __init__(self, Kp, Ki, dt, limits=(0, 4500)):
        self.Kp, self.Ki, self.dt = Kp, Ki, dt
        self.limits = limits
        self.integral = 0.0
        
    def step(self, target, measured):
        error = target - measured
        self.integral += self.Ki * error * self.dt
        u = self.Kp * error + self.integral
        
        # Anti-windup : on sature la commande et on fige l'intégrale
        if u > self.limits[1]: 
            u = self.limits[1]
            self.integral -= self.Ki * error * self.dt
        elif u < self.limits[0]:
            u = self.limits[0]
            self.integral -= self.Ki * error * self.dt
        return u

class Motor2ndOrder:
    """Modèle de réponse temporelle du moteur (2nd ordre)."""
    def __init__(self, omega_n=45.0, zeta=0.8):
        self.omega_n, self.zeta = omega_n, zeta
        self.state = np.array([0.0, 0.0]) # [Couple_actuel, dCouple/dt]
        
    def step(self, T_ref, dt):
        x1, x2 = self.state
        dx1 = x2
        dx2 = (self.omega_n**2) * (T_ref - x1) - 2 * self.zeta * self.omega_n * x2
        self.state += np.array([dx1, dx2]) * dt
        return self.state[0]

class VehicleDynamics:
    """Modèle physique du véhicule (Masse ponctuelle)."""
    def __init__(self, mass=5000, wheel_radius=0.24):
        self.m, self.r = mass, wheel_radius
        self.v = 0.0 # m/s
        
    def step(self, total_torque_wheel, dt):
        acceleration = (total_torque_wheel / self.r) / self.m
        self.v += acceleration * dt
        return self.v

# ============================================================================
# 3. BOUCLE DE SIMULATION
# ============================================================================

def executer_simulation():
    # Configuration
    dt = 0.005 # 5ms
    temps = np.arange(0, 8.0, dt)
    ratio_reduction = 26.0
    
    # Initialisation des blocs
    # Kp et Ki réglés pour une réponse stable à 5 tonnes
    regu = PIController(Kp=13000, Ki=4500, dt=dt)
    vehicule = VehicleDynamics(mass=5000)
    moteurs = [Motor2ndOrder() for _ in range(4)]
    
    # Consigne de vitesse : 15 km/h (4.17 m/s) à t = 0.5s
    v_cible_ms = 15 / 3.6
    
    history = {'t': [], 'v_ref': [], 'v_reel': [], 'torque': []}

    for t in temps:
        v_ref = v_cible_ms if t >= 0.5 else 0.0
        v_mesuree = vehicule.v
        
        # 1. Régulateur PI (Boucle de vitesse)
        T_global_roues = regu.step(v_ref, v_mesuree)
        
        # 2. Allocation (Égale)
        T_motor_ref = (T_global_roues / 4) / ratio_reduction
        
        # 3. Dynamique Moteurs
        T_reels = [m.step(T_motor_ref, dt) for m in moteurs]
        
        # 4. Dynamique Véhicule
        v_actuelle = vehicule.step(sum(T_reels) * ratio_reduction, dt)
        
        history['t'].append(t)
        history['v_ref'].append(v_ref * 3.6) # Passage en km/h pour le graphe
        history['v_reel'].append(v_actuelle * 3.6)
        history['torque'].append(sum(T_reels) * ratio_reduction)

    # --- AFFICHAGE DES RÉSULTATS ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    ax1.plot(history['t'], history['v_ref'], 'r--', label="Consigne (km/h)")
    ax1.plot(history['t'], history['v_reel'], 'b-', linewidth=2, label="Vitesse réelle (km/h)")
    ax1.set_title("Contrôle de Vitesse EZDolly en Boucle Fermée (PI)")
    ax1.set_ylabel("Vitesse (km/h)")
    ax1.legend(); ax1.grid(True, alpha=0.3)
    
    ax2.plot(history['t'], history['torque'], 'g-', label="Couple total aux roues (Nm)")
    ax2.set_ylabel("Couple (Nm)")
    ax2.set_xlabel("Temps (s)")
    ax2.legend(); ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# ============================================================================
# LANCEMENT
# ============================================================================

if __name__ == "__main__":
    # Charge le fichier sans faire de tracé 3D
    df_params = charger_donnees_moteur("efficiency_map.csv")
    
    # Exécute la simulation temporelle
    executer_simulation()