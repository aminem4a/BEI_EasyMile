import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# --- PATH HACK ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.vehicle_sim.simulation import Simulation

# --- IMPORTATION DU GÉNÉRATEUR ---
try:
    # On importe la fonction qui génère vos profils dans allocation_3.py
    from allocation_3 import generate_test_reference
    print("✅ Fonction 'generate_test_reference' trouvée dans allocation_3.")
except ImportError:
    print("❌ ERREUR : Le fichier 'allocation_3.py' est introuvable.")
    print("   Veuillez le placer dans le dossier 'examples/' (à côté de ce script).")
    sys.exit(1)

def main():
    # --- 1. CONFIGURATION ---
    veh_cfg = {'wheel_radius': 0.24} # Rayon 0.24m (comme dans votre allocation_3 ?)
    sim_cfg = {}

    # --- 2. GÉNÉRATION DES CONSIGNES ---
    # On appelle votre fonction pour avoir les mêmes données que votre graphe
    print("Génération du scénario test...")
    t, v_rpm, trq_profile = generate_test_reference(duration_s=60.0, dt=0.1, torque_peak_total=360.0)

    # Conversion Vitesse : RPM -> m/s pour le simulateur physique
    # v (m/s) = rpm * 2pi/60 * R
    v_ms = v_rpm * (2 * np.pi / 60.0) * veh_cfg['wheel_radius']
    
    print(f"✅ Scénario généré : {len(t)} points, Durée {t[-1]:.1f}s")
    print(f"   Couple Max : {np.max(trq_profile):.1f} Nm")
    print(f"   Vitesse Max : {np.max(v_rpm):.0f} RPM ({np.max(v_ms)*3.6:.1f} km/h)")

    # --- 3. SIMULATION ---
    # Le simulateur va utiliser le TorqueAllocator (qui utilise vos Moindres Carrés)
    sim = Simulation(sim_cfg, veh_cfg)
    
    # Exécution de la boucle ouverte
    results = sim.run_open_loop(t, trq_profile, v_ms)

    # --- 4. AFFICHAGE (5 Fenêtres Doubles) ---
    styles = {
        'Inverse':   ('-',  3.0, 0.5),
        'Piecewise': ('--', 2.0, 0.8),
        'Smooth':    ('-.', 2.0, 1.0),
        'Quadratic': (':',  2.5, 1.0)
    }
    
    def plot_ref_top(ax, t, trq):
        """Affiche la consigne en haut de chaque fenêtre"""
        ax.plot(t, trq, 'k', linewidth=1.5, label="Consigne (allocation_3)")
        ax.set_title("ENTRÉE : Consigne Générée")
        ax.set_ylabel("Couple (Nm)")
        ax.grid(True, linestyle=':')
        ax.legend(loc='upper right')
        ax.set_xticklabels([]) # Pas de texte X pour ne pas surcharger

    print("Affichage des résultats...")

    # --- FENÊTRE 1 : Suivi ---
    fig1 = plt.figure("Suivi de Consigne", figsize=(10, 8))
    ax1 = fig1.add_subplot(211)
    plot_ref_top(ax1, t, trq_profile)
    
    ax2 = fig1.add_subplot(212)
    for name, data in results.items():
        ls, lw, al = styles.get(name, ('-', 1, 1))
        ax2.plot(data['time'], data['trq_achieved'], label=name, ls=ls, lw=lw, alpha=al)
    ax2.set_title("SORTIE : Couple Réel")
    ax2.set_xlabel("Temps (s)")
    ax2.set_ylabel("Couple (Nm)")
    ax2.legend()
    ax2.grid(True)
    plt.tight_layout()

    # --- FENÊTRE 2 : Ratio ---
    fig2 = plt.figure("Répartition", figsize=(10, 8))
    ax1 = fig2.add_subplot(211)
    plot_ref_top(ax1, t, trq_profile)
    
    ax2 = fig2.add_subplot(212)
    for name, data in results.items():
        ls, lw, al = styles.get(name, ('-', 1, 1))
        ax2.plot(data['time'], data['front_ratio'], label=name, ls=ls, lw=lw, alpha=al)
    ax2.set_title("SORTIE : Ratio Avant (0=Arr, 1=Av)")
    ax2.set_xlabel("Temps (s)")
    ax2.set_ylabel("Ratio")
    ax2.legend()
    ax2.grid(True)
    plt.tight_layout()

    # --- FENÊTRE 3 : Cos Phi AV ---
    fig3 = plt.figure("Rendement Avant", figsize=(10, 8))
    ax1 = fig3.add_subplot(211)
    plot_ref_top(ax1, t, trq_profile)
    
    ax2 = fig3.add_subplot(212)
    for name, data in results.items():
        ls, lw, al = styles.get(name, ('-', 1, 1))
        ax2.plot(data['time'], data['cos_phi_f'], label=name, ls=ls, lw=lw, alpha=al)
    ax2.set_title("SORTIE : Cos Phi Moteur Avant")
    ax2.set_xlabel("Temps (s)")
    ax2.set_ylabel("Cos Phi")
    ax2.set_ylim(0, 1.1)
    ax2.legend()
    ax2.grid(True)
    plt.tight_layout()

    # --- FENÊTRE 4 : Cos Phi AR ---
    fig4 = plt.figure("Rendement Arrière", figsize=(10, 8))
    ax1 = fig4.add_subplot(211)
    plot_ref_top(ax1, t, trq_profile)
    
    ax2 = fig4.add_subplot(212)
    for name, data in results.items():
        ls, lw, al = styles.get(name, ('-', 1, 1))
        ax2.plot(data['time'], data['cos_phi_r'], label=name, ls=ls, lw=lw, alpha=al)
    ax2.set_title("SORTIE : Cos Phi Moteur Arrière")
    ax2.set_xlabel("Temps (s)")
    ax2.set_ylabel("Cos Phi")
    ax2.set_ylim(0, 1.1)
    ax2.legend()
    ax2.grid(True)
    plt.tight_layout()

    # --- FENÊTRE 5 : Puissance ---
    fig5 = plt.figure("Puissance", figsize=(10, 8))
    ax1 = fig5.add_subplot(211)
    plot_ref_top(ax1, t, trq_profile)
    
    ax2 = fig5.add_subplot(212)
    for name, data in results.items():
        ls, lw, al = styles.get(name, ('-', 1, 1))
        ax2.plot(data['time'], data['power'], label=name, ls=ls, lw=lw, alpha=al)
    ax2.set_title("SORTIE : Puissance Électrique Totale")
    ax2.set_xlabel("Temps (s)")
    ax2.set_ylabel("Watts")
    ax2.legend()
    ax2.grid(True)
    plt.tight_layout()

    plt.show()

if __name__ == "__main__":
    main()