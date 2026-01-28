import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import interpolate, signal

# --- PATH HACK ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def load_excel_data():
    """Lecture des données brute comme dans MODELISATION_MOTEUR.py"""
    base_dir = os.path.dirname(os.path.dirname(__file__))
    file_path = os.path.join(base_dir, "data", "modeli_simplifie.xlsx")

    if not os.path.exists(file_path):
        # Fallback pour exécution locale
        file_path = "modeli_simplifie.xlsx"
        if not os.path.exists(file_path):
            print(f"❌ Fichier introuvable : {file_path}")
            return None, None

    print(f"📂 Chargement de {file_path}...")
    
    try:
        # Lecture sans header
        df_ref = pd.read_excel(file_path, sheet_name='reff', header=None)
        df_feed = pd.read_excel(file_path, sheet_name='feedB', header=None)
        
        # Extraction brute (numpy values)
        data_ref = df_ref.values
        data_feed = df_feed.values
        
        # --- LOGIQUE EXACTE DU CAMARADE ---
        # Ref: Col 6 (Temps), Col 7 (Couple) - Indices 0-based
        # time_ref = data_ref[:, 6] - data_ref[0, 6]
        time_ref = data_ref[:, 6] - data_ref[0, 6]
        C_ref = data_ref[:, 7]
        
        # Feed: Col 1 (Temps), Col 4 (Couple)
        # time_feed = data_feed[:, 1] - data_feed[0, 1]
        time_feed = data_feed[:, 1] - data_feed[0, 1]
        C_feed = data_feed[:, 4]
        
        return (time_ref, C_ref), (time_feed, C_feed)

    except Exception as e:
        print(f"❌ Erreur lecture Excel : {e}")
        return None, None

def main():
    print("🧪 VALIDATION MOTEUR : Reproduction Stricte MODELISATION_MOTEUR.py")
    
    # 1. Chargement
    data_ref, data_feed = load_excel_data()
    if data_ref is None: return
    
    time_ref, C_ref = data_ref
    time_feed, C_feed = data_feed

    # 2. Détection du Front de Step (Logique EXACTE)
    dy = np.diff(C_ref)
    idx_step = np.argmax(np.abs(dy))

    if idx_step + 1 < len(time_ref):
        t_step = time_ref[idx_step + 1]
    else:
        t_step = time_ref[-1]

    print(f"Step détecté à t = {t_step:.4f} s (index {idx_step})")

    # 3. Fenêtrage (Logique EXACTE)
    t_before = 0.2
    t_after = 2.0

    # Interpolation de la référence sur le temps du feedback
    f_interp = interpolate.interp1d(
        time_ref, C_ref,
        kind='linear',
        fill_value='extrapolate',
        bounds_error=False
    )
    C_ref_interp = f_interp(time_feed)

    # Masque
    mask = (time_feed >= t_step - t_before) & (time_feed <= t_step + t_after)

    if np.sum(mask) < 2:
        mask = (time_feed >= t_step)

    # Extraction des fenêtres
    start_idx = np.where(mask)[0][0]
    
    t = time_feed[mask] - time_feed[start_idx] # Temps relatif local
    yf = C_feed[mask]     # Feedback fenêtré
    yr = C_ref_interp[mask] # Reference fenêtrée

    # 4. Méthode de STREJC (Logique EXACTE)
    # Gain statique
    k = yf[-1] / yr[-1]
    dt = t[1] - t[0]

    # Dérivée numérique
    dyf = np.diff(yf) / dt
    dyf = np.append(dyf, 0) # Padding pour garder la taille

    # Point d'inflexion
    dyfmax_idx = np.argmax(dyf)
    dyfmax = dyf[dyfmax_idx]
    tmax = t[dyfmax_idx]
    ypi = yf[dyfmax_idx]

    # Tangente : y = ax + b
    a = dyfmax
    b = ypi - a * tmax

    # Intersections
    t1 = -b / a if a != 0 else 0
    t2 = (k - b) / a if a != 0 else 0

    Td = t1
    Ta = t2 - t1

    # Temps de retard Tr (premier point non nul)
    non_zero_idx = np.where(yf != 0)[0]
    Tr = t[non_zero_idx[0]] if len(non_zero_idx) > 0 else 0

    Tu = Td - Tr
    rapport = Tu / Ta if Ta != 0 else 0
    n = 2

    # Formules Strejc spécifiques du code original
    tau = Ta / np.exp(1)
    Tutab = Ta * 0.104
    TR_val = Tu - Tutab
    T_delay = max(Tr + TR_val, 0) # Le retard global 'T' du script

    print("\n" + "="*40)
    print("RÉSULTATS IDENTIFICATION (STREJC)")
    print("="*40)
    print(f"Gain K   = {k:.4f}")
    print(f"Retard T = {T_delay:.4f} s")
    print(f"Tau      = {tau:.4f} s")
    print(f"Ordre n  = {n}")

    # 5. Simulation du Modèle Identifié (Logique EXACTE)
    # Fonction de transfert : F(s) = K * exp(-T*s) / (tau*s + 1)^n
    # On simule la partie sans retard d'abord
    num = [k]
    # Dénominateur : (tau*s + 1)^2 = tau^2*s^2 + 2*tau*s + 1
    den = [tau**2, 2*tau, 1]
    
    sys_tf = signal.TransferFunction(num, den)

    # Vecteur temps pour la simulation (long pour bien voir)
    # Comme dans le script matlab : 0:dt:5*(tau + T)
    t_sim_end = 5 * (tau + T_delay)
    if t_sim_end < t[-1]: t_sim_end = t[-1] # On couvre au moins la fenêtre
    
    t_sim = np.arange(0, t_sim_end, dt)
    
    # Entrée échelon unitaire (car k prend déjà en compte l'amplitude ?)
    # ATTENTION : dans le script matlab, step(F) fait un échelon unitaire.
    # Mais ici 'k' est le gain statique y_final/y_ref.
    # Si on veut superposer à yf, il faut multiplier par l'amplitude de l'échelon (yr[-1]).
    # Ou alors F contient déjà K*yr[-1] ? Non, K est adimensionnel dans le code python, 
    # mais K = yf_end / yr_end. Donc step(sys) tend vers K.
    # Pour comparer à yf, il faut multiplier par yr[-1] (la consigne).
    
    step_amplitude = yr[-1]
    
    # Step response de scipy
    _, y_sim_raw = signal.step(sys_tf, T=t_sim)
    
    # Mise à l'échelle (step unitaire -> amplitude réelle)
    y_sim_scaled = y_sim_raw * step_amplitude

    # Application du Retard (T_delay)
    # On décale le signal
    idx_delay = int(T_delay / dt)
    if idx_delay > 0:
        y_sim_delayed = np.concatenate([np.zeros(idx_delay), y_sim_scaled[:-idx_delay]])
    else:
        y_sim_delayed = y_sim_scaled
        
    # On coupe à la même longueur que t_sim
    y_sim_delayed = y_sim_delayed[:len(t_sim)]

    # 6. Affichage Comparatif
    plt.figure(figsize=(10, 6))
    
    # Données réelles fenêtrées (offsettées à 0)
    plt.plot(t, yr, 'b--', linewidth=1.5, label='Consigne Fenêtrée')
    plt.plot(t, yf, 'r-', linewidth=2.5, alpha=0.6, label='Mesure Réelle')
    
    # Simulation Modèle
    plt.plot(t_sim, y_sim_delayed, 'g-', linewidth=2.0, label=f'Modèle Strejc (Tau={tau:.3f}s)')

    plt.title('Validation : Modèle Identifié vs Réalité (Fenêtre Step)')
    plt.xlabel('Temps relatif (s)')
    plt.ylabel('Couple')
    plt.legend()
    plt.grid(True)
    plt.xlim(0, max(t[-1], t_sim[-1]))
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()