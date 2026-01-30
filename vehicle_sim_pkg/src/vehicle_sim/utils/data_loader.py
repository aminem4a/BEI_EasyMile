import os
import numpy as np
import pandas as pd

class DataLoader:
    @staticmethod
    def get_project_root():
        current_dir = os.path.dirname(os.path.abspath(__file__))
        return os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))

    @staticmethod
    def load_efficiency_data():
        """
        Charge le CSV de map moteur.
        Gère les séparateurs ';' ou ',' et les décimales à virgule.
        """
        root = DataLoader.get_project_root()
        
        # Liste des noms possibles selon votre structure
        candidates = [
            "efficiency_map_cleaned.csv",  # Priorité 1 (vu dans votre image)
            "efficiency_map_clean.csv",    # Priorité 2 (nom que vous mentionnez)
            "efficiency_map.csv"           # Fallback
        ]
        
        path = None
        for name in candidates:
            p = os.path.join(root, "data", name)
            if os.path.exists(p):
                path = p
                print(f"✅ Map trouvée : {name}")
                break
        
        if path is None:
            raise FileNotFoundError(f"❌ Impossible de trouver la map dans data/. Cherché : {candidates}")
        
        # Lecture flexible
        try:
            df = pd.read_csv(path, sep=None, engine='python')
        except:
            df = pd.read_csv(path)

        # Nettoyage noms colonnes
        cols = [str(c).strip().lower() for c in df.columns]
        df.columns = cols
        
        # Identification intelligente des colonnes
        col_t = next((c for c in cols if any(x in c for x in ['torq','couple','t_'])), None)
        col_s = next((c for c in cols if any(x in c for x in ['speed','vit','rpm','s_'])), None)
        col_z = next((c for c in cols if any(x in c for x in ['eff','cos','z_'])), None)
        
        if not all([col_t, col_s, col_z]):
            raise ValueError(f"❌ Colonnes non reconnues dans le CSV : {cols}. Il faut Couple, Vitesse, Efficacité.")

        # --- CONVERSION STRICTE (Virgule -> Point) ---
        for c in [col_t, col_s, col_z]:
            if df[c].dtype == object: 
                # Remplace ',' par '.' et supprime les espaces
                df[c] = df[c].astype(str).str.replace(',', '.').str.strip()
            # Force conversion numérique (les erreurs deviennent NaN)
            df[c] = pd.to_numeric(df[c], errors='coerce')
        
        # Supprime les lignes invalides (NaN)
        len_before = len(df)
        df = df.dropna()
        len_after = len(df)
        
        if len_before != len_after:
            print(f"⚠️ {len_before - len_after} lignes invalides supprimées du CSV.")
        
        if len(df) == 0:
            raise ValueError("❌ Le fichier CSV est vide ou illisible (problème de format nombre).")

        return df[col_t].values, df[col_s].values, df[col_z].values

    @staticmethod
    def build_efficiency_map(clip_01=True):
        T_data, S_data, Z_data = DataLoader.load_efficiency_data()
        
        # Vérification si Z est en % (0-100) -> conversion (0-1)
        if np.mean(Z_data) > 1.0:
            print("ℹ️ Données d'efficacité détectées en %. Conversion /100 appliquée.")
            Z_data = Z_data / 100.0

        def build_A(t, s):
            t, s = np.asarray(t).ravel(), np.asarray(s).ravel()
            return np.column_stack([t**2, s**2, t*s, t, s, np.ones_like(t)])
            
        def fit(t, s, z):
            A = build_A(t, s)
            c, _, _, _ = np.linalg.lstsq(A, z, rcond=None)
            return c
            
        def predict(coeffs, t, s):
            A = build_A(np.asarray(t), np.asarray(s))
            return (A @ coeffs).reshape(np.asarray(t).shape)

        # Fitting bilatéral (Positif / Négatif)
        coeff_pos = fit(T_data, S_data, Z_data)
        coeff_neg = fit(-T_data, S_data, Z_data)

        def cosphi_map(T, S):
            T = np.asarray(T)
            S = np.asarray(S)
            # Prédiction
            val = np.where(T >= 0, predict(coeff_pos, T, S), predict(coeff_neg, T, S))
            
            if clip_01:
                return np.clip(val, 0.05, 1.0) # Clip min à 0.05 pour éviter les 0 graphiques
            return val

        cosphi_map.Cmax_data = float(np.max(np.abs(T_data)))
        return cosphi_map

    @staticmethod
    def load_scenario_csv(filename):
        # ... (Identique au précédent : gestion des virgules aussi ici)
        root = DataLoader.get_project_root()
        path = os.path.join(root, "data", "scenarios", filename)
        if not os.path.exists(path): path = os.path.join(root, "data", filename)
        
        if not os.path.exists(path):
            print(f"❌ Scénario introuvable : {filename}")
            return None, None, None

        try:
            df = pd.read_csv(path, sep=None, engine='python')
            df.columns = [c.strip().lower() for c in df.columns]
            
            col_t = next(c for c in df.columns if 'time' in c)
            col_c = next(c for c in df.columns if 'torq' in c)
            col_v = next(c for c in df.columns if 'speed' in c)
            
            for c in [col_t, col_c, col_v]:
                if df[c].dtype == object: df[c] = df[c].astype(str).str.replace(',', '.')
                df[c] = pd.to_numeric(df[c], errors='coerce')
            
            df = df.dropna()
            t = df[col_t].values
            # Conversion ms -> s
            if len(t) > 0 and np.max(t) > 10000: t = t / 1000.0
            
            idx = np.argsort(t)
            return t[idx]-t[idx][0], df[col_c].values[idx], df[col_v].values[idx]
        except Exception as e:
            print(f"❌ Erreur Scénario : {e}")
            return None, None, None