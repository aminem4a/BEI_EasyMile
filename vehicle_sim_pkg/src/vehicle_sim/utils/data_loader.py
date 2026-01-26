# -*- coding: utf-8 -*-
import os
import pandas as pd
import numpy as np
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator, interp1d

class DataLoader:
    """
    Classe utilitaire responsable du chargement des données.
    Version Corrective : Gère les points-virgules à la fin des lignes.
    """

    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.eff_map_interp = None
        self.eff_map_nearest = None 
        self.torque_max_interp = None
        
        # On essaie de charger le fichier clean, sinon l'original
        path_clean = os.path.join(data_dir, "efficiency_map_clean.csv")
        path_orig = os.path.join(data_dir, "efficiency_map.csv")
        
        if os.path.exists(path_clean):
            self.load_efficiency_map(path_clean)
        else:
            self.load_efficiency_map(path_orig)
            
        self.load_torque_characteristics(os.path.join(data_dir, "engine_carac.csv"))

    def _clean_column(self, series):
        """Nettoie une colonne : supprime ';', remplace ',' par '.' et convertit."""
        s = series.astype(str)
        # 1. Supprime le point-virgule parasite (le problème actuel)
        s = s.str.replace(';', '', regex=False)
        # 2. Remplace virgule décimale par point (au cas où)
        s = s.str.replace(',', '.', regex=False)
        # 3. Supprime les espaces
        s = s.str.strip()
        return pd.to_numeric(s, errors='coerce')

    def load_efficiency_map(self, filepath: str):
        print(f"🔍 [Loader] Chargement Map : {os.path.basename(filepath)}")
        if not os.path.exists(filepath): 
            print("❌ [Loader] Fichier introuvable.")
            return

        try:
            # Lecture CSV avec séparateur virgule
            df = pd.read_csv(filepath, sep=',', engine='python')
            
            # Nettoyage des noms de colonnes (enlève #, espaces, parenthèses)
            df.columns = [c.lower().split('(')[0].replace('#','').strip() for c in df.columns]
            df.columns = [c.replace('phi', '').strip() if 'cos' in c else c for c in df.columns]
            
            print(f"   -> Colonnes trouvées : {list(df.columns)}")

            # Identification
            c_spd = next((c for c in df.columns if 'speed' in c or 'rpm' in c), None)
            c_trq = next((c for c in df.columns if 'torque' in c or 'couple' in c), None)
            c_phi = next((c for c in df.columns if 'cos' in c or 'eff' in c), None)

            if c_spd and c_trq and c_phi:
                # Nettoyage des données (C'est ici que le ; est supprimé)
                df[c_spd] = self._clean_column(df[c_spd])
                df[c_trq] = self._clean_column(df[c_trq])
                df[c_phi] = self._clean_column(df[c_phi])
                
                # Suppression des lignes invalides
                df.dropna(subset=[c_spd, c_trq, c_phi], inplace=True)
                
                if len(df) > 3:
                    points = np.column_stack([np.abs(df[c_trq]), np.abs(df[c_spd])])
                    values = df[c_phi].values
                    
                    self.eff_map_nearest = NearestNDInterpolator(points, values)
                    try:
                        self.eff_map_interp = LinearNDInterpolator(points, values)
                        print(f"✅ [Loader] Map chargée avec succès : {len(df)} points.")
                    except:
                        self.eff_map_interp = None
                        print(f"⚠️ [Loader] Map chargée (Mode Nearest uniquement) : {len(df)} points.")
                else:
                    print(f"❌ [Loader] Trop peu de points valides ({len(df)}) après nettoyage.")
                    # Affiche un aperçu pour comprendre pourquoi
                    print("Aperçu des données rejetées (si NaN):")
                    print(df.head())
            else:
                print(f"❌ [Loader] Colonnes manquantes : {list(df.columns)}")

        except Exception as e:
            print(f"❌ [Loader] Erreur : {e}")

    def load_torque_characteristics(self, filepath: str):
        if not os.path.exists(filepath): return
        try:
            df = pd.read_csv(filepath, sep=None, engine='python')
            df.columns = [c.lower().strip() for c in df.columns]
            c_spd = next((c for c in df.columns if 'speed' in c), None)
            c_trq = next((c for c in df.columns if 'torque_5mn' in c or 'torque' in c), None)
            if c_spd and c_trq:
                df[c_spd] = self._clean_column(df[c_spd])
                df[c_trq] = self._clean_column(df[c_trq])
                df.dropna(inplace=True)
                df.sort_values(by=c_spd, inplace=True)
                self.torque_max_interp = interp1d(df[c_spd], df[c_trq], kind='linear', fill_value="extrapolate")
                print(f"✅ [Loader] Limites Couple chargées.")
        except:
            pass

    def get_cosphi(self, torque, rpm):
        t, r = abs(torque), abs(rpm)
        if self.eff_map_interp:
            val = self.eff_map_interp(t, r)
            if not np.isnan(val): return float(max(0.1, min(1.0, val)))
        if self.eff_map_nearest:
            val = self.eff_map_nearest(t, r)
            return float(max(0.1, min(1.0, val)))
        return 0.5

    def get_max_torque(self, rpm):
        if self.torque_max_interp is None: return 180.0
        return float(self.torque_max_interp(abs(rpm)))

    def load_scenario(self, filepath: str):
        print(f"\n🔍 [Loader] Scénario : {os.path.basename(filepath)}")
        if not os.path.exists(filepath): raise FileNotFoundError(filepath)
        try:
            df = pd.read_csv(filepath, sep=None, engine='python')
            df.columns = [c.lower().strip() for c in df.columns]
            c_time = next((c for c in df.columns if 'time' in c), None)
            c_speed = next((c for c in df.columns if 'speed' in c), None)
            if c_time and c_speed:
                t = self._clean_column(df[c_time])
                v = self._clean_column(df[c_speed])
                mask = ~np.isnan(t) & ~np.isnan(v)
                t, v = t[mask].values, v[mask].values
                if len(t) > 0 and t[-1] > 1000: t /= 1000.0
                print(f"✅ [Loader] Chargé : {len(t)} pts.")
                return t, v
        except: pass
        return np.array([]), np.array([])