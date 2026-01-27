# -*- coding: utf-8 -*-
import os
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d

class DataLoader:
    """
    Chargeur avec Coefficients Polynomiaux FORCÉS (Hard-coded).
    Modèle : Z = A*T^2 + B*S^2 + G*T*S + D*T + E*S + F
    """
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        
        # --- 1. COEFFICIENTS FORCÉS (Donnés par l'utilisateur) ---
        # Ordre : [T^2 (A), S^2 (B), T*S (G), T (D), S (E), 1 (F)]
        
        # Cas C >= 0
        self.coeffs_pos = np.array([
            -0.000125338082179,   # A (T^2)
            -7.33783997958e-08,   # B (S^2)
            -5.31545729369e-06,   # G (T*S) --> Interaction
             0.0272621224832,     # D (T)
             0.000657907533808,   # E (S)
            -0.634473528053       # F (Constante)
        ])
        
        # Cas C < 0
        self.coeffs_neg = np.array([
            -0.000125338082179,   # A' (T^2)
            -7.33783997958e-08,   # B' (S^2)
             5.31545729369e-06,   # G' (T*S) --> Signe opposé
            -0.0272621224832,     # D' (T)   --> Signe opposé
             0.000657907533808,   # E' (S)
            -0.634473528053       # F' (Constante)
        ])

        # Pas de normalisation (Vos coeffs sont pour des valeurs brutes)
        self.scale_rpm = 1.0
        self.scale_torque = 1.0
        self.torque_max_interp = None
        
        # Chargement (uniquement pour les limites physiques, pas pour le modèle)
        self.load_torque_characteristics(os.path.join(data_dir, "engine_carac.csv"))
        
        # On charge quand même le CSV juste pour info ou limites, mais on ignore ses données pour le modèle
        path = os.path.join(data_dir, "efficiency_map_clean.csv")
        if not os.path.exists(path): path = os.path.join(data_dir, "efficiency_map.csv")
        self.load_efficiency_map(path)

    def _clean_column(self, series):
        s = series.astype(str).str.replace(';', '').str.replace(',', '.').str.strip()
        return pd.to_numeric(s, errors='coerce')

    def load_efficiency_map(self, filepath: str):
        # Cette fonction ne sert plus qu'à afficher que tout va bien
        # Les coefficients sont déjà chargés dans le __init__
        print(f"🔍 [Loader] Utilisation des coefficients MANUELS (A, B, D, E, F, G).")
        print(f"   -> Modèle Positif et Négatif chargés.")

    def get_cosphi(self, torque, rpm):
        """
        Calcul du CosPhi avec le modèle polynomial manuel.
        """
        t = float(torque)
        s = abs(float(rpm))
        
        # Sélection des coefficients
        if t >= 0:
            coeffs = self.coeffs_pos
            # Pour le cas positif, on utilise t directement
            # Vecteur : [t^2, s^2, t*s, t, s, 1]
            x = np.array([t**2, s**2, t*s, t, s, 1.0])
        else:
            coeffs = self.coeffs_neg
            # Pour le cas négatif, t est négatif (ex: -10)
            # Votre modèle pour C < 0 semble attendre le C brut (puisque D' est négatif)
            # Vérifions : D'*C = (-0.027) * (-10) = +0.27 (Positif, correct)
            # Vérifions : G'*C*S = (+5.3e-6) * (-10) * S = -... (Correct par rapport à G)
            x = np.array([t**2, s**2, t*s, t, s, 1.0])

        # Produit scalaire
        val = np.dot(x, coeffs)
        
        # Bornes physiques (CosPhi entre 0.1 et 1.0)
        # On garde 0.1 en plancher, mais avec vos coeffs, ça devrait être bien mieux centré
        return float(max(0.1, min(0.99, val)))

    def load_torque_characteristics(self, filepath: str):
        if not os.path.exists(filepath): return
        try:
            df = pd.read_csv(filepath, sep=None, engine='python')
            df.columns = [c.lower().strip() for c in df.columns]
            c_spd = next((c for c in df.columns if 'speed' in c), None)
            c_trq = next((c for c in df.columns if 'torque' in c), None)
            if c_spd and c_trq:
                df[c_spd] = self._clean_column(df[c_spd])
                df[c_trq] = self._clean_column(df[c_trq])
                df.dropna(inplace=True)
                df.sort_values(by=c_spd, inplace=True)
                self.torque_max_interp = interp1d(df[c_spd], df[c_trq], kind='linear', fill_value="extrapolate")
        except: pass

    def get_max_torque(self, rpm):
        if self.torque_max_interp is None: return 180.0
        return float(self.torque_max_interp(abs(rpm)))
    
    def load_scenario(self, filepath: str):
        if not os.path.exists(filepath): return [], [], []
        try:
            df = pd.read_csv(filepath, sep=None, engine='python')
            df.columns = [c.lower().strip() for c in df.columns]
            c_time = next((c for c in df.columns if 'time' in c), None)
            c_speed = next((c for c in df.columns if 'speed' in c), None)
            c_trq = next((c for c in df.columns if 'torque' in c), None)
            if not c_trq: c_trq = next((c for c in df.columns if 'feedback' in c), None)

            if c_time and c_speed and c_trq:
                t = self._clean_column(df[c_time])
                v = self._clean_column(df[c_speed])
                trq = self._clean_column(df[c_trq])
                mask = ~np.isnan(t) & ~np.isnan(v) & ~np.isnan(trq)
                t, v, trq = t[mask].values, v[mask].values, trq[mask].values
                if len(t) > 0 and t[-1] > 1000: t /= 1000.0
                return t, v, trq
        except: pass
        return [], [], []