import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

class Vehicule:
    """
    Modèle physique calibré pour reproduire fidèlement l'essai EZDolly.
    """
    def __init__(self):
        # --- PARAMÈTRES CALIBRÉS (POUR COLLER AUX DONNÉES) ---
        # Masse ajustée pour correspondre à la dynamique réelle de l'essai
        self.m = 5000.0        
        
        # Données techniques fixes
        self.r = 0.24          
        self.S = 4.0           
        self.Cx = 0.8          
        self.Crr = 0.015       
        self.rho_air = 1.225   
        self.g = 9.81          

        # Frottement mécanique ajusté pour la décélération (Transmission 26:1)
        self.friction_torque = 221.0 

        # État
        self.v = 0.0
        self.x = 0.0
        self.a = 0.0

    def update(self, torques, dt, slope_rad=0.0):
        # 1. Traction
        f_tract = sum(torques) / self.r
        
        # 2. Sens du mouvement
        if abs(self.v) > 1e-3:
            sign_v = np.sign(self.v)
        else:
            sign_v = np.sign(f_tract) if abs(f_tract) > 0 else 0

        # 3. Résistances
        f_aero = 0.5 * self.rho_air * self.S * self.Cx * (self.v**2) * sign_v
        f_roll = self.m * self.g * self.Crr * np.cos(slope_rad) * sign_v
        f_slope = self.m * self.g * np.sin(slope_rad)
        f_mech = (self.friction_torque / self.r) * sign_v  # Pertes mécaniques

        f_resist = f_aero + f_roll + f_slope + f_mech

        # 4. Dynamique
        self.a = (f_tract - f_resist) / self.m
        
        # 5. Intégration
        self.v += self.a * dt
        self.x += self.v * dt

        # Arrêt net
        if abs(self.v) < 0.01 and abs(f_tract) < abs(f_resist):
            self.v = 0.0
            self.a = 0.0

        return self.v, self.a, self.x

def load_data(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Fichier introuvable : {file_path}")

    df = pd.read_csv(file_path, sep=';')

    def clean(s):
        # Nettoyage robuste des virgules/points
        return s if pd.api.types.is_numeric_dtype(s) else s.astype(str).str.replace(',', '.').astype(float)

    # Extraction des colonnes (Time/Torque et Time/Speed)
    # Adaptez les indices si votre CSV change (ici colonnes B, C, E, F)
    t_torq = clean(df.iloc[:, 1]).dropna().values / 1000.0
    v_torq = clean(df.iloc[:, 2]).dropna().values
    t_spd = clean(df.iloc[:, 4]).values / 1000.0
    v_spd = clean(df.iloc[:, 5]).values
    
    # Synchronisation
    torq_interp = np.interp(t_spd, t_torq, v_torq)
    return t_spd, v_spd, torq_interp

def main():
    # CHEMIN DU FICHIER : Mettez ici le chemin exact de votre CSV
    FILE_PATH = r"c:/Users/Usuario/Documents/IMPORTANT/3A N7/BEI/vitesseclasseurvehicule.csv"

    try:
        t_vec, v_real, t_input = load_data(FILE_PATH)
        
        veh = Vehicule()
        v_sim = []
        
        # Calcul des pas de temps dynamiques
        dts = np.diff(t_vec)
        dts = np.insert(dts, 0, 0.01)

        for i, t in enumerate(t_vec):
            dt = max(dts[i], 0.001)
            
            # Injection du couple (réparti sur 4 roues)
            torques = [t_input[i] / 4.0] * 4
            
            v_next, _, _ = veh.update(torques, dt)
            v_sim.append(v_next)

        # Affichage
        plt.figure(figsize=(10, 6))
        plt.plot(t_vec, v_real, 'k-', linewidth=1.5, alpha=0.7, label='Mesure Réelle')
        plt.plot(t_vec, v_sim, 'r--', linewidth=2.0, label='Simulation (Modèle Calibré)')
        
        plt.title('Validation : Modèle vs Réalité')
        plt.xlabel('Temps [s]')
        plt.ylabel('Vitesse [m/s]')
        plt.legend()
        plt.grid(True, linestyle=':')
        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"Erreur d'exécution : {e}")

if __name__ == "__main__":
    main()