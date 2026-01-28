import pandas as pd
import numpy as np
from scipy.interpolate import LinearNDInterpolator, interp1d

class DataLoader:
    def __init__(self, map_path):
        """
        Charge la map d'efficacité/pertes et prépare les interpolateurs.
        """
        self.map_path = map_path
        self.interpolator = None      # Pour l'efficacité (2D)
        self.torque_max_interp = None # Pour le couple max (1D) -> C'est ce qui manquait !
        self._load_map()

    def _load_map(self):
        try:
            df = pd.read_csv(self.map_path)
            
            # Nettoyage des noms de colonnes
            df.columns = [c.strip().lower() for c in df.columns]
            
            # Identification des colonnes
            col_speed = next((c for c in df.columns if 'speed' in c or 'rpm' in c or 'vitesse' in c), None)
            col_torque = next((c for c in df.columns if 'torque' in c or 'nm' in c or 'couple' in c), None)
            col_eff = next((c for c in df.columns if 'eff' in c or 'rendement' in c), None)

            if not (col_speed and col_torque and col_eff):
                print(f"⚠️ Colonnes non reconnues. Tentative avec indices 0, 1, 2.")
                # Fallback indices
                cols = df.columns
                col_speed, col_torque, col_eff = cols[0], cols[1], cols[2]

            # 1. Création de l'interpolateur d'efficacité (2D)
            # On prend la valeur absolue pour la vitesse et le couple (symétrie supposée)
            points = df[[col_speed, col_torque]].copy()
            points[col_speed] = points[col_speed].abs()
            points[col_torque] = points[col_torque].abs()
            values = df[col_eff].values

            # On utilise LinearNDInterpolator pour gérer les nuages de points non réguliers
            self.interpolator = LinearNDInterpolator(points.values, values, fill_value=0.5)

            # 2. Création de l'interpolateur de Couple Max (1D)
            # C'est ce qui manquait à simulation.py !
            # On cherche pour chaque vitesse, quel est le couple maximal renseigné dans la map
            
            # On arrondit la vitesse pour grouper les points similaires (évite les doublons flottants)
            df['speed_round'] = df[col_speed].abs().round(1)
            
            # On groupe par vitesse et on prend le couple max absolu
            envelope = df.groupby('speed_round')[col_torque].apply(lambda x: x.abs().max()).reset_index()
            envelope = envelope.sort_values('speed_round')

            # On crée une fonction d'interpolation 1D (Vitesse -> Couple Max)
            # bounds_error=False et fill_value permettent d'extrapoler ou de garder la dernière valeur
            self.torque_max_interp = interp1d(
                envelope['speed_round'], 
                envelope[col_torque], 
                kind='linear', 
                bounds_error=False, 
                fill_value=(envelope[col_torque].iloc[0], envelope[col_torque].iloc[-1])
            )
            
            print(f"✅ Map chargée : {len(df)} points. Enveloppe de couple max générée.")

        except Exception as e:
            print(f"❌ Erreur critique chargement map : {e}")
            # Fallbacks pour éviter le crash
            self.interpolator = lambda x: np.array([0.8])
            self.torque_max_interp = lambda x: 200.0 # Couple max par défaut

    def get_loss(self, torque, rpm):
        """
        Calcule les pertes (en Watts).
        """
        speed_abs = abs(rpm)
        torque_abs = abs(torque)
        
        # Récupérer l'efficacité
        try:
            eff = float(self.interpolator([speed_abs, torque_abs])[0])
        except:
            eff = 0.8 # Valeur par défaut si hors map

        # Sécurités
        if eff < 0.05: eff = 0.05
        if eff > 1.0: eff = 1.0 

        # Calcul Puissance Méca
        w_rad = speed_abs * 2 * np.pi / 60
        p_meca = torque_abs * w_rad

        if p_meca < 1e-3:
            return 10.0 + 0.01 * speed_abs 

        # Pertes = P_elec - P_meca = (P_meca/eff) - P_meca
        losses = p_meca * (1.0/eff - 1.0)
        return losses
    
    def get_max_torque(self, rpm):
        """Retourne le couple max disponible pour un RPM donné."""
        if self.torque_max_interp:
            return float(self.torque_max_interp(abs(rpm)))
        return 200.0

    def load_scenario(self, filepath):
        """
        Charge un scénario CSV et renvoie (t, v, trq).
        """
        try:
            # On utilise le moteur python pour éviter les erreurs de parsing C parfois
            df = pd.read_csv(filepath, sep=None, engine='python')
            
            df.columns = [c.strip().lower() for c in df.columns]

            # Recherche des colonnes
            col_t = next((c for c in df.columns if 'time' in c or 'temps' in c or 'sec' in c), df.columns[0])
            col_v = next((c for c in df.columns if 'speed' in c or 'vit' in c or 'vel' in c), None)
            col_trq = next((c for c in df.columns if 'torq' in c or 'cpl' in c or 'couple' in c), None)

            t = df[col_t].values
            
            if col_v:
                v = df[col_v].values
            else:
                # Si pas de vitesse, on suppose que c'est la dernière colonne ou 0
                v = np.zeros_like(t)

            if col_trq:
                trq = df[col_trq].values
            else:
                trq = np.zeros_like(t)

            return t, v, trq

        except Exception as e:
            print(f"Erreur load_scenario {filepath}: {e}")
            return np.array([0]), np.array([0]), np.array([0])