import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- PATH HACK ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.vehicle_sim.simulation import Simulation

def load_real_scenario(filename):
    """
    Charge un scénario réel (Excel ou CSV) et extrait t, v, couple.
    """
    base_dir = os.path.dirname(os.path.dirname(__file__))
    file_path = os.path.join(base_dir, "data", filename)
    
    if not os.path.exists(file_path):
        file_path = os.path.join(base_dir, "data", "scenarios", filename)
        if not os.path.exists(file_path):
            file_path = filename 

    if not os.path.exists(file_path):
        print(f"❌ Fichier introuvable : {filename}")
        return None, None, None

    print(f"📂 Chargement de {os.path.basename(file_path)}...")

    try:
        # 1. Lecture
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path, sep=None, engine='python')
        elif file_path.endswith('.xlsx'):
            xl = pd.ExcelFile(file_path)
            if 'reff' in xl.sheet_names:
                df = pd.read_excel(file_path, sheet_name='reff')
            elif 'Axle_torque_setpoint' in xl.sheet_names:
                 df = pd.read_excel(file_path, sheet_name='Axle_torque_setpoint')
            else:
                df = pd.read_excel(file_path)
        else:
            print("❌ Format non supporté (utilisez .csv ou .xlsx)")
            return None, None, None

        # 2. Nettoyage colonnes
        df.columns = [str(c).strip().lower() for c in df.columns]
        
        # 3. Détection
        col_t = next((c for c in df.columns if any(x in c for x in ['time', 'temps', 'sec'])), None)
        col_v = next((c for c in df.columns if any(x in c for x in ['speed', 'vit', 'velocity'])), None)
        col_trq = next((c for c in df.columns if any(x in c for x in ['torq', 'cpl', 'nm', 'setpoint'])), None)

        if not col_t or not col_trq:
            print(f"⚠️ Colonnes non trouvées. Cols dispo: {df.columns}")
            return None, None, None

        # 4. Extraction
        t = df[col_t].values
        trq = df[col_trq].values
        
        if col_v:
            v = df[col_v].values
        else:
            print("⚠️ Pas de vitesse. Défaut 50 km/h.")
            v = np.full_like(t, 50.0/3.6)

        # Unités (ms -> s)
        if np.max(t) > 10000: 
            t = t / 1000.0

        # Tri
        sort_idx = np.argsort(t)
        t = t[sort_idx]
        v = v[sort_idx]
        trq = trq[sort_idx]
        t = t - t[0]

        return t, trq, v

    except Exception as e:
        print(f"❌ Erreur lecture : {e}")
        return None, None, None

def main():
    # --- 1. CONFIGURATION ---
    # Fichier à tester
    filename = "nominal_driving_5kmh_unloaded.csv" 
    
    veh_cfg = {'wheel_radius': 0.3}
    sim_cfg = {}

    # --- 2. CHARGEMENT ---
    t, trq_profile, v_profile = load_real_scenario(filename)
    if t is None: return

    print(f"✅ Scénario chargé : {len(t)} points, Durée {t[-1]:.1f}s")

    # --- 3. SIMULATION ---
    sim = Simulation(sim_cfg, veh_cfg)
    results = sim.run_open_loop(t, trq_profile, v_profile)

    # --- 4. AFFICHAGE (Double Plot par Fenêtre) ---
    print("Génération des graphiques...")
    
    styles = {
        'Inverse':   ('-',  3.0, 0.5), # Large, transparent
        'Piecewise': ('--', 2.0, 0.8),
        'Smooth':    ('-.', 2.0, 1.0),
        'Quadratic': (':',  2.5, 1.0)  # Pointillé bien visible
    }
    
    # Fonction utilitaire pour tracer la consigne en haut
    def plot_reference_top(ax, t, trq):
        ax.plot(t, trq, 'k', linewidth=1.5, label="Consigne (Couple)")
        ax.set_title("ENTRÉE : Consigne de Couple")
        ax.set_ylabel("Couple (Nm)")
        ax.grid(True, linestyle=':')
        ax.legend(loc='upper right')
        # On supprime les labels X pour ne pas surcharger, car partagé avec le bas
        ax.set_xticklabels([]) 

    # --- FENÊTRE 1 : Suivi de Consigne ---
    fig1 = plt.figure("Suivi de Consigne", figsize=(10, 8))
    
    # Haut : Consigne Seule
    ax1_top = fig1.add_subplot(2, 1, 1)
    plot_reference_top(ax1_top, t, trq_profile)
    
    # Bas : Réponse des Stratégies
    ax1_bot = fig1.add_subplot(2, 1, 2)
    for name, data in results.items():
        ls, lw, alpha = styles.get(name, ('-', 1, 1))
        ax1_bot.plot(data['time'], data['trq_achieved'], label=name, ls=ls, lw=lw, alpha=alpha)
    
    ax1_bot.set_title("SORTIE : Couple Réellement Délivré par les Stratégies")
    ax1_bot.set_xlabel("Temps (s)")
    ax1_bot.set_ylabel("Couple (Nm)")
    ax1_bot.grid(True)
    ax1_bot.legend()
    plt.tight_layout()

    # --- FENÊTRE 2 : Ratio ---
    fig2 = plt.figure("Répartition Couple", figsize=(10, 8))
    
    # Haut : Consigne (Contexte)
    ax2_top = fig2.add_subplot(2, 1, 1)
    plot_reference_top(ax2_top, t, trq_profile)
    
    # Bas : Ratio
    ax2_bot = fig2.add_subplot(2, 1, 2)
    for name, data in results.items():
        ls, lw, alpha = styles.get(name, ('-', 1, 1))
        ax2_bot.plot(data['time'], data['front_ratio'], label=name, ls=ls, lw=lw, alpha=alpha)
        
    ax2_bot.set_title("SORTIE : Répartition (Ratio Avant)")
    ax2_bot.set_ylabel("0 = 100% Arrière | 1 = 100% Avant")
    ax2_bot.set_xlabel("Temps (s)")
    ax2_bot.grid(True)
    ax2_bot.legend()
    plt.tight_layout()

    # --- FENÊTRE 3 : Cos Phi AVANT ---
    fig3 = plt.figure("Rendement Moteur AVANT", figsize=(10, 8))
    
    # Haut : Consigne
    ax3_top = fig3.add_subplot(2, 1, 1)
    plot_reference_top(ax3_top, t, trq_profile)
    
    # Bas : Cos Phi
    ax3_bot = fig3.add_subplot(2, 1, 2)
    for name, data in results.items():
        ls, lw, alpha = styles.get(name, ('-', 1, 1))
        ax3_bot.plot(data['time'], data['cos_phi_f'], label=name, ls=ls, lw=lw, alpha=alpha)
        
    ax3_bot.set_title("SORTIE : Efficacité Moteur AVANT")
    ax3_bot.set_ylabel("Cos Phi")
    ax3_bot.set_xlabel("Temps (s)")
    ax3_bot.set_ylim(-0.1, 1.1)
    ax3_bot.grid(True)
    ax3_bot.legend()
    plt.tight_layout()

    # --- FENÊTRE 4 : Cos Phi ARRIERE ---
    fig4 = plt.figure("Rendement Moteur ARRIERE", figsize=(10, 8))
    
    # Haut : Consigne
    ax4_top = fig4.add_subplot(2, 1, 1)
    plot_reference_top(ax4_top, t, trq_profile)
    
    # Bas : Cos Phi
    ax4_bot = fig4.add_subplot(2, 1, 2)
    for name, data in results.items():
        ls, lw, alpha = styles.get(name, ('-', 1, 1))
        ax4_bot.plot(data['time'], data['cos_phi_r'], label=name, ls=ls, lw=lw, alpha=alpha)
        
    ax4_bot.set_title("SORTIE : Efficacité Moteur ARRIERE")
    ax4_bot.set_ylabel("Cos Phi")
    ax4_bot.set_xlabel("Temps (s)")
    ax4_bot.set_ylim(-0.1, 1.1)
    ax4_bot.grid(True)
    ax4_bot.legend()
    plt.tight_layout()

    # --- FENÊTRE 5 : Puissance Elec ---
    fig5 = plt.figure("Puissance Consommée", figsize=(10, 8))
    
    # Haut : Consigne
    ax5_top = fig5.add_subplot(2, 1, 1)
    plot_reference_top(ax5_top, t, trq_profile)
    
    # Bas : Puissance
    ax5_bot = fig5.add_subplot(2, 1, 2)
    for name, data in results.items():
        ls, lw, alpha = styles.get(name, ('-', 1, 1))
        ax5_bot.plot(data['time'], data['power'], label=name, ls=ls, lw=lw, alpha=alpha)
        
    ax5_bot.set_title("SORTIE : Puissance Électrique Consommée")
    ax5_bot.set_ylabel("Puissance (Watts)")
    ax5_bot.set_xlabel("Temps (s)")
    ax5_bot.grid(True)
    ax5_bot.legend()
    plt.tight_layout()
    
    plt.show()

if __name__ == "__main__":
    main()