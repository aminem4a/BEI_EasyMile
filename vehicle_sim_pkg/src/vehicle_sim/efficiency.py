import os
import numpy as np
import pandas as pd

def get_project_root():
    """Remonte arborescence depuis src/vehicle_sim/efficiency.py vers la racine"""
    # efficiency.py est dans src/vehicle_sim/
    # donc os.path.dirname = src/vehicle_sim
    # encore dirname = src
    # encore dirname = racine projet
    current = os.path.dirname(os.path.abspath(__file__))
    return os.path.dirname(os.path.dirname(current))

def load_efficiency_data():
    """Charge T, S, Z depuis data/efficiency_map_clean.csv"""
    root = get_project_root()
    path = os.path.join(root, "data", "efficiency_map_clean.csv")
    
    if not os.path.exists(path):
        raise FileNotFoundError(f"Impossible de trouver le fichier map : {path}")
    
    df = pd.read_csv(path)
    
    # Adaptation selon les noms de colonnes de votre CSV
    # On cherche (Couple, Vitesse, Efficacité/CosPhi)
    # On suppose l'ordre ou des noms probables
    cols = [c.lower() for c in df.columns]
    
    # Détection basique
    col_t = next((c for c in df.columns if 'torq' in c.lower() or 'couple' in c.lower()), df.columns[0])
    col_s = next((c for c in df.columns if 'speed' in c.lower() or 'vit' in c.lower() or 'rpm' in c.lower()), df.columns[1])
    col_z = next((c for c in df.columns if 'eff' in c.lower() or 'cos' in c.lower()), df.columns[2])
    
    T = df[col_t].values
    S = df[col_s].values
    Z = df[col_z].values
    
    return T, S, Z

# --- Logique Moindres Carrés ---

def build_A_full(T, S):
    T = np.asarray(T).ravel()
    S = np.asarray(S).ravel()
    # Modèle quadratique complet : T^2, S^2, TS, T, S, 1
    return np.column_stack([T**2, S**2, T*S, T, S, np.ones_like(T)])

def fit_full(T, S, Z):
    A = build_A_full(T, S)
    coeffs, _, _, _ = np.linalg.lstsq(A, Z, rcond=None)
    return coeffs

def predict_full(coeffs, T, S):
    T = np.asarray(T)
    S = np.asarray(S)
    A = build_A_full(T, S)
    return (A @ coeffs).reshape(T.shape)

def build_efficiency_map(clip_01=True):
    """
    Construit et retourne la fonction map(T, S) -> CosPhi
    """
    T_data, S_data, Z_data = load_efficiency_data()
    
    # Fit positif et négatif (miroir)
    coeff_pos = fit_full(T_data, S_data, Z_data)
    coeff_neg = fit_full(-T_data, S_data, Z_data)

    def cosphi_map(T, S):
        T = np.asarray(T)
        S = np.asarray(S)
        Zp = predict_full(coeff_pos, T, S)
        Zn = predict_full(coeff_neg, T, S)
        
        # Collage
        val = np.where(T >= 0, Zp, Zn)
        
        if clip_01:
            return np.clip(val, 0.0, 1.0)
        return val

    # On attache le Cmax détecté pour l'allocateur
    cosphi_map.Cmax_data = float(np.max(np.abs(T_data)))
    
    # Pour Quadratic, on a besoin des coeffs
    cosphi_map.coeff_pos = coeff_pos
    cosphi_map.coeff_neg = coeff_neg
    
    return cosphi_map