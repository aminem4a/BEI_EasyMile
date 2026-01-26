# -*- coding: utf-8 -*-
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src'))
if root_path not in sys.path: sys.path.insert(0, root_path)

from vehicle_sim.simulation import Simulation
from vehicle_sim.config import SimulationConfig, VehicleConfig

def calculate_kpis(results, dt):
    # 1. Efficacité (Moyenne temporelle du CosPhi moyen des 2 moteurs)
    cp_global = (np.array(results["cosphi_av"]) + np.array(results["cosphi_ar"])) / 2.0
    kpi_eff = np.mean(cp_global)

    # 2. Rugosité (Moyenne de la dérivée absolue du couple AVANT)
    torque_av = np.array(results["torque_fl"])
    derivative = np.diff(torque_av) / dt
    kpi_roughness = np.mean(np.abs(derivative))

    # 3. Précision (RMSE sur tout le profil)
    # On compare la consigne (15 km/h) à la vitesse réelle
    v_real = np.array(results["velocity"])
    v_target = np.array(results["target_speed"])
    mse = ((v_real - v_target) ** 2).mean()
    kpi_rmse = np.sqrt(mse) * 3.6 # Conversion en km/h pour que ce soit parlant

    return kpi_eff, kpi_roughness, kpi_rmse

def main():
    sim_cfg = SimulationConfig(dt=0.01, duration=15.0)
    veh_cfg = VehicleConfig()
    
    modes = ["inverse", "piecewise", "smooth"]
    metrics = {"Mode": [], "Efficacité": [], "Stress (Nm/s)": [], "Erreur (km/h)": []}

    print("--- DÉBUT DU BENCHMARK ---")

    for mode in modes:
        print(f"Test du mode : {mode}...")
        sim = Simulation(sim_cfg, veh_cfg)
        sim.allocator.mode = mode
        # On passe la cible en m/s
        results = sim.run(target_speed=15.0/3.6)
        
        eff, stress, rmse = calculate_kpis(results, sim_cfg.dt)
        
        metrics["Mode"].append(mode)
        metrics["Efficacité"].append(eff)
        metrics["Stress (Nm/s)"].append(stress)
        metrics["Erreur (km/h)"].append(rmse)

    df = pd.DataFrame(metrics)
    print("\n--- RÉSULTATS DU COMPARATIF ---")
    print(df.to_string(index=False))
    
    # --- VISUALISATION (3 Graphiques) ---
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 5))
    
    colors = ['gray', 'orange', 'green']
    
    # Graphe 1 : Efficacité
    ax1.bar(df["Mode"], df["Efficacité"], color=colors, alpha=0.7)
    ax1.set_title("Efficacité Moyenne (CosPhi)\n(Plus haut = Mieux)")
    ax1.set_ylim(0, 1.0)
    for i, v in enumerate(df["Efficacité"]):
        ax1.text(i, v + 0.01, f"{v:.2f}", ha='center', fontweight='bold')

    # Graphe 2 : Stress
    ax2.bar(df["Mode"], df["Stress (Nm/s)"], color=colors, alpha=0.7)
    ax2.set_title("Stress Mécanique Actionneurs\n(Plus bas = Mieux)")
    ax2.set_ylabel("Nm/s")
    for i, v in enumerate(df["Stress (Nm/s)"]):
        ax2.text(i, v + 1, f"{v:.0f}", ha='center', fontweight='bold')

    # Graphe 3 : RMSE (Nouveau)
    ax3.bar(df["Mode"], df["Erreur (km/h)"], color=colors, alpha=0.7)
    ax3.set_title("Erreur de Suivi Vitesse (RMSE)\n(Plus bas = Mieux)")
    ax3.set_ylabel("Erreur Moyenne (km/h)")
    for i, v in enumerate(df["Erreur (km/h)"]):
        ax3.text(i, v + 0.01, f"{v:.3f}", ha='center', fontweight='bold')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()