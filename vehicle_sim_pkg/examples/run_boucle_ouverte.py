import sys
import os
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.vehicle_sim.simulation import Simulation

def main():
    veh_cfg = {'wheel_radius': 0.3}
    sim_cfg = {'rpm': 500.0}

    # SCENARIO
    t = np.linspace(0, 10, 200)
    # Sinus + une petite composante constante pour ne pas rester autour de 0 tout le temps
    trq_profile = 100 + 100 * np.sin(2 * np.pi * 0.2 * t) 
    v_profile = np.full_like(t, 50.0 / 3.6) 

    sim = Simulation(sim_cfg, veh_cfg)
    results = sim.run_open_loop(t, trq_profile, v_profile)

    # AFFICHAGE COMPLET
    plt.figure(figsize=(15, 12)) # Plus grand pour tout afficher
    
    styles = {
        'Inverse': ('-', 3, 0.4),
        'Piecewise': ('--', 2, 0.8),
        'Smooth': ('-.', 2, 1.0),
        'Quadratic': (':', 2.5, 1.0) # On met en avant le quadratic
    }

    # 1. SUIVI DE CONSIGNE (Total Torque)
    plt.subplot(3, 2, 1)
    # Trace la consigne en noir
    plt.plot(t, trq_profile, 'k', linewidth=1, label="Consigne")
    for name, data in results.items():
        ls, lw, alpha = styles.get(name, ('-', 1, 1))
        # On trace le 'trq_achieved'
        plt.plot(data['time'], data['trq_achieved'], label=name, 
                 linestyle=ls, linewidth=lw, alpha=alpha)
    plt.title("Suivi de Consigne (Couple Total)")
    plt.ylabel("Couple (Nm)")
    plt.grid(True)
    plt.legend()

    # 2. RATIO
    plt.subplot(3, 2, 2)
    for name, data in results.items():
        ls, lw, alpha = styles.get(name, ('-', 1, 1))
        plt.plot(data['time'], data['front_ratio'], label=name, 
                 linestyle=ls, linewidth=lw, alpha=alpha)
    plt.title("Répartition (Ratio Avant)")
    plt.ylabel("0=Arr, 1=Av")
    plt.grid(True)

    # 3. COS PHI AVANT
    plt.subplot(3, 2, 3)
    for name, data in results.items():
        ls, lw, alpha = styles.get(name, ('-', 1, 1))
        plt.plot(data['time'], data['cos_phi_f'], label=name, 
                 linestyle=ls, linewidth=lw, alpha=alpha)
    plt.title("Rendement Moteur AVANT")
    plt.ylabel("Cos Phi")
    plt.ylim(0, 1.1)
    plt.grid(True)

    # 4. COS PHI ARRIERE
    plt.subplot(3, 2, 4)
    for name, data in results.items():
        ls, lw, alpha = styles.get(name, ('-', 1, 1))
        plt.plot(data['time'], data['cos_phi_r'], label=name, 
                 linestyle=ls, linewidth=lw, alpha=alpha)
    plt.title("Rendement Moteur ARRIERE")
    plt.ylabel("Cos Phi")
    plt.ylim(0, 1.1)
    plt.grid(True)

    # 5. PUISSANCE ELEC TOTALE (Pour voir qui gagne)
    plt.subplot(3, 2, (5, 6)) # Prend toute la largeur en bas
    for name, data in results.items():
        ls, lw, alpha = styles.get(name, ('-', 1, 1))
        plt.plot(data['time'], data['power'], label=name, 
                 linestyle=ls, linewidth=lw, alpha=alpha)
    plt.title("Puissance Électrique Consommée (Totale)")
    plt.xlabel("Temps (s)")
    plt.ylabel("Watts")
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()