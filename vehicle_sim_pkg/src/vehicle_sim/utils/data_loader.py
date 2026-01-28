import pandas as pd
import numpy as np

class DataLoader:
    def __init__(self, map_path=None):
        # Le map_path est gardé pour compatibilité mais on ne charge plus le CSV d'efficacité.
        self.map_path = map_path
        
        # --- COEFFICIENTS POLYNOMIAUX (Moindres Carrés) ---
        # Formule : Loss = A*T^2 + B*w^2 + G*T*w + D*T + E*w + F
        
        # Cas Couple POSITIF (Traction)
        self.COEFFS_POS = {
            'A': -0.000125338082179,
            'B': -7.33783997958e-08,
            'D': 0.0272621224832,
            'E': 0.000657907533808,
            'F': -0.634473528053,
            'G': -5.31545729369e-06
        }
        
        # Cas Couple NÉGATIF (Freinage)
        self.COEFFS_NEG = {
            'A': -0.000125338082179,
            'B': -7.33783997958e-08,
            'D': -0.0272621224832,  # Signe inversé
            'E': 0.000657907533808,
            'F': -0.634473528053,
            'G': 5.31545729369e-06   # Signe inversé
        }
        
        print("✅ DataLoader : Mode Moindres Carrés (Pas d'interpolation).")

    def get_loss(self, torque, rpm):
        """
        Calcule les pertes via le modèle polynomial (Moindres Carrés).
        Remplace l'ancienne méthode d'interpolation.
        """
        # Sélection des coefficients selon le signe du couple
        if torque >= 0:
            c = self.COEFFS_POS
        else:
            c = self.COEFFS_NEG
            
        w = abs(rpm) # Vitesse toujours positive dans le modèle
        T = torque   # Le signe est géré par le choix des coeffs D et G
        
        # Application de la formule : L = A*T^2 + B*w^2 + G*T*w + D*T + E*w + F
        loss = (c['A'] * T**2 + 
                c['B'] * w**2 + 
                c['G'] * T * w + 
                c['D'] * T + 
                c['E'] * w + 
                c['F'])
        
        # Sécurité : les pertes ne peuvent pas être négatives (artefact du modèle à vide)
        if loss < 0:
            return 0.0
            
        return loss

    def load_scenario(self, filepath):
        """
        Charge un fichier de scénario temporel (CSV ou Excel).
        Cette méthode reste indispensable pour lire les inputs (t, v, couple).
        """
        try:
            # Détection extension
            if filepath.endswith('.xlsx'):
                df = pd.read_excel(filepath)
            else:
                df = pd.read_csv(filepath, sep=None, engine='python')
                
            # Nettoyage des colonnes
            df.columns = [str(c).strip().lower() for c in df.columns]
            
            # Détection intelligente des colonnes
            col_t = next((c for c in df.columns if any(x in c for x in ['time', 'temps', 'sec'])), None)
            col_v = next((c for c in df.columns if any(x in c for x in ['speed', 'vit', 'velocity'])), None)
            col_trq = next((c for c in df.columns if any(x in c for x in ['torq', 'cpl', 'nm', 'setpoint'])), None)
            
            # Extraction
            t = df[col_t].values if col_t else np.array([0])
            v = df[col_v].values if col_v else np.zeros_like(t)
            trq = df[col_trq].values if col_trq else np.zeros_like(t)
            
            # Conversion ms -> s si nécessaire
            if np.max(t) > 10000:
                t = t / 1000.0
                
            # Recalage à t=0
            t = t - t[0]
            
            return t, v, trq
            
        except Exception as e:
            print(f"❌ Erreur lecture scénario {filepath} : {e}")
            return np.array([0]), np.array([0]), np.array([0])