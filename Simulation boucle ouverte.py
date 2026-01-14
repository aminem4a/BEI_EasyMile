# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from typing import List, Dict

# ============================================================================
# 1. CHARGEMENT ET PRÉPARATION DES DONNÉES
# ============================================================================

def charger_donnees_moteur(file_name: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(file_name, engine='python', on_bad_lines='skip', 
                         skipinitialspace=True, encoding='latin1', sep=None)
        df.columns = df.columns.astype(str).str.replace(r'#|\(|\)', '', regex=True).str.strip()
        mapping = {df.columns[0]: 'SpeedRPM', df.columns[1]: 'TorqueNm', df.columns[2]: 'CosPhi'}
        df = df.rename(columns=mapping)
        for col in ['SpeedRPM', 'TorqueNm', 'CosPhi']:
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(r'\[|\]|;', '', regex=True), errors='coerce')
        df = df.dropna(subset=['SpeedRPM', 'TorqueNm', 'CosPhi'])
        print(f"✓ Données chargées : {len(df)} points valides.")
        return df
    except Exception as e:
        print(f"❌ Erreur lors du chargement du fichier : {e}")
        return None

# ============================================================================
# 2. VISUALISATION 3D
# ============================================================================

def tracer_surface_3d(df: pd.DataFrame):
    x = df['SpeedRPM'].values
    y = df['TorqueNm'].values
    z = df['CosPhi'].values
    
    xi = np.linspace(x.min(), x.max(), 100)
    yi = np.linspace(y.min(), y.max(), 100)
    XI, YI = np.meshgrid(xi, yi)
    ZI = griddata((x, y), z, (XI, YI), method='linear')
    
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(XI, YI, ZI, cmap='viridis', edgecolor='none', alpha=0.8)
    ax.scatter(x, y, z, color='red', marker='o', s=15, label='Points réels')
    
    ax.set_title('Cartographie 3D du Facteur de Puissance (Cos phi)')
    ax.set_xlabel('Vitesse (RPM)')
    ax.set_ylabel('Couple (Nm)')
    ax.set_zlabel('Cos phi')
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)
    plt.legend()
    
    # SAUVEGARDE ET AFFICHAGE
    plt.savefig('cosphi_3d_surface.png')
    print("✓ Graphique 3D 'cosphi_3d_surface.png' généré.")
    plt.show() # <--- Ouvre la fenêtre

# ============================================================================
# 3. MODÈLE DYNAMIQUE ET SIMULATION
# ============================================================================

class MotorDynamics:
    def __init__(self, omega_n=45.0, zeta=0.8):
        self.omega_n = omega_n
        self.zeta = zeta
        self.state = np.array([0.0, 0.0])
        
    def step(self, T_ref, dt):
        x1, x2 = self.state
        dx1 = x2
        dx2 = (self.omega_n**2) * (T_ref - x1) - 2 * self.zeta * self.omega_n * x2
        self.state[0] += dx1 * dt
        self.state[1] += dx2 * dt
        return self.state[0]

def simuler_reponse_temporelle():
    dt = 0.001
    temps = np.arange(0, 1.2, dt)
    ratio_reduction = 26.0
    moteur = MotorDynamics()
    
    T_roue_ref = 2500.0
    T_moteur_ref = (T_roue_ref / 4) / ratio_reduction
    
    res_consigne = []
    res_reel = []

    for t in temps:
        target = T_moteur_ref if t >= 0.2 else 0.0
        res_consigne.append(target)
        res_reel.append(moteur.step(target, dt))
        
    plt.figure(figsize=(10, 6))
    plt.plot(temps, res_consigne, 'r--', label="Consigne (Allocateur)")
    plt.plot(temps, res_reel, 'b-', linewidth=2, label="Réponse moteur (2nd ordre)")
    plt.title("Simulation : Réponse Temporelle du Couple")
    plt.xlabel("Temps (s)")
    plt.ylabel("Couple Moteur (Nm)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # SAUVEGARDE ET AFFICHAGE
    plt.savefig('reponse_dynamique.png')
    print("✓ Graphique temporel 'reponse_dynamique.png' généré.")
    plt.show() # <--- Ouvre la fenêtre

# ============================================================================
# POINT D'ENTRÉE
# ============================================================================

if __name__ == "__main__":
    # Assurez-vous que le nom du fichier correspond exactement à celui dans votre dossier
    nom_fichier = "efficiency_map.csv"
    
    df_moteur = charger_donnees_moteur(nom_fichier)
    
    if df_moteur is not None:
        simuler_reponse_temporelle()
        print("\nSimulation terminée. Les fenêtres de graphiques devraient être ouvertes.")