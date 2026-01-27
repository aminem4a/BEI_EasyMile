# -*- coding: utf-8 -*-
import sys
import os
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import numpy as np

# Ajout du chemin src
root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src'))
if root not in sys.path: sys.path.insert(0, root)

from vehicle_sim import Simulation, SimulationConfig, VehicleConfig
from vehicle_sim.utils import DataLoader

def main():
    # 1. Configuration des chemins
    base_dir = os.path.dirname(os.path.dirname(__file__)) 
    data_dir = os.path.join(base_dir, "data")
    scenario_file = os.path.join(data_dir, "scenarios", "nominal_driving_13kmh_unloaded.csv")
    
    print(f"--- 📂 Lancement Scénario ---")
    
    # 2. Chargement du Scénario
    loader = DataLoader(data_dir)
    t_ref, v_ref = loader.load_scenario(scenario_file)

    if len(t_ref) == 0:
        print("❌ Arrêt : Scénario vide ou illisible.")
        return

    # --- LIMITATION DURÉE (Pour test rapide) ---
    # Remplacez 20.0 par t_ref[-1] pour simuler tout le fichier
    duration = min(t_ref[-1], 20.0) 
    print(f"⏱️ Durée simulation limitée à : {duration} secondes")

    # 3. Préparation Simulation
    target_profile = interp1d(t_ref, v_ref, bounds_error=False, fill_value=v_ref[-1])
    sim_cfg = SimulationConfig(dt=0.01, duration=duration)
    veh_cfg = VehicleConfig()

    modes = ["inverse", "piecewise", "smooth", "quadratic"]
    results = {}

    # 4. Exécution des Stratégies
    for m in modes:
        print(f"🚀 Simulation : {m.upper()}...")
        sim = Simulation(sim_cfg, veh_cfg, data_dir=data_dir)
        sim.allocator.mode = m
        results[m] = sim.run(target_speed_profile=target_profile)

    # ==========================================
    #               AFFICHAGE
    # ==========================================
    print("📊 Génération des graphiques...")

    # --- FIGURE 1 : DYNAMIQUE (Vitesse & Couple) ---
    plt.figure("Dynamique Véhicule", figsize=(10, 8))
    
    # Sous-graphique 1 : Vitesse
    plt.subplot(2, 1, 1)
    plt.plot(t_ref, v_ref * 3.6, 'k--', label="Référence (Mesure)", linewidth=2, alpha=0.6)
    for m in modes:
        v_kmh = [x * 3.6 for x in results[m]["velocity"]]
        plt.plot(results[m]["time"], v_kmh, label=f"Simu {m}")
    plt.title("Suivi de Vitesse")
    plt.ylabel("Vitesse (km/h)")
    plt.legend()
    plt.grid(True)
    plt.xlim(0, duration)

    # Sous-graphique 2 : Couple Total (Avant + Arrière) ou juste Avant
    plt.subplot(2, 1, 2)
    for m in modes:
        # On affiche le couple moteur Avant pour comparer l'activité
        plt.plot(results[m]["time"], results[m]["torque_fl"], label=f"Couple AV ({m})")
    plt.title("Sollicitation Moteur Avant")
    plt.xlabel("Temps (s)")
    plt.ylabel("Couple (Nm)")
    plt.grid(True)
    plt.xlim(0, duration)

    # --- FIGURE 2 : EFFICACITÉ (Cos Phi Avant vs Arrière) ---
    plt.figure("Analyse Efficacité (CosPhi)", figsize=(10, 8))

    # Sous-graphique 1 : Cos Phi AVANT
    plt.subplot(2, 1, 1)
    for m in modes:
        plt.plot(results[m]["time"], results[m]["cosphi_av"], label=m)
    plt.title("Efficacité Moteur AVANT (Cos φ)")
    plt.ylabel("Cos φ")
    plt.ylim(0, 1.05)
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.xlim(0, duration)

    # Sous-graphique 2 : Cos Phi ARRIÈRE
    plt.subplot(2, 1, 2)
    for m in modes:
        plt.plot(results[m]["time"], results[m]["cosphi_ar"], label=m, linestyle='--')
    plt.title("Efficacité Moteur ARRIÈRE (Cos φ)")
    plt.xlabel("Temps (s)")
    plt.ylabel("Cos φ")
    plt.ylim(0, 1.05)
    plt.grid(True)
    plt.xlim(0, duration)

    # Affichage final des deux fenêtres
    plt.show()

if __name__ == "__main__":
    main()