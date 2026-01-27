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
    base_dir = os.path.dirname(os.path.dirname(__file__)) 
    data_dir = os.path.join(base_dir, "data")
    scenario_file = os.path.join(data_dir, "scenarios", "nominal_driving_13kmh_unloaded.csv")
    
    print(f"--- 📂 Lancement Simulation BOUCLE OUVERTE ---")
    
    # 1. Chargement (Temps, Vitesse, Couple)
    loader = DataLoader(data_dir)
    t_ref, v_ref, trq_ref = loader.load_scenario(scenario_file)

    if len(t_ref) == 0:
        print("❌ Arrêt : Scénario vide.")
        return

    # Durée
    duration = min(t_ref[-1], 20.0) # On limite à 20s pour tester
    print(f"⏱️ Durée : {duration}s")

    # 2. Création de la fonction d'interpolation du COUPLE
    # C'est ce qu'on va "rejouer" dans la simulation
    torque_profile = interp1d(t_ref, trq_ref, bounds_error=False, fill_value=0.0)

    sim_cfg = SimulationConfig(dt=0.01, duration=duration)
    veh_cfg = VehicleConfig()

    modes = ["inverse", "piecewise", "smooth", "quadratic"]
    results = {}

    for m in modes:
        print(f"🚀 Simulation Open Loop : {m.upper()}...")
        sim = Simulation(sim_cfg, veh_cfg, data_dir=data_dir)
        sim.allocator.mode = m
        
        # APPEL DE LA NOUVELLE MÉTHODE OPEN LOOP
        results[m] = sim.run_open_loop(torque_profile)

    # 3. Affichage
    print("📊 Génération des graphiques...")
    
    # Figure 1 : Vérification Physique (Vitesse + Couple injecté)
    plt.figure("Open Loop - Dynamique", figsize=(10, 8))
    
    plt.subplot(2, 1, 1)
    plt.plot(t_ref, v_ref * 3.6, 'k--', label="Vitesse Réelle (Target)", linewidth=2, alpha=0.5)
    for m in modes:
        v_sim = [x * 3.6 for x in results[m]["velocity"]]
        plt.plot(results[m]["time"], v_sim, label=f"Simu {m}")
    plt.title("Vitesse : Réalité vs Simulation (Dérive attendue)")
    plt.ylabel("km/h")
    plt.legend()
    plt.grid(True)
    plt.xlim(0, duration)

    plt.subplot(2, 1, 2)
    plt.plot(t_ref, trq_ref, 'r-', label="Couple Total injecté (CSV)", alpha=0.3)
    # On superpose le couple total généré par l'allocateur pour vérifier
    t_sim = results["smooth"]["time"]
    trq_sim_tot = [(results["smooth"]["torque_fl"][i] + results["smooth"]["torque_rl"][i])*veh_cfg.ratio_reduction for i in range(len(t_sim))]
    plt.plot(t_sim, trq_sim_tot, 'b--', label="Couple Total Simu (Vérif)")
    plt.title("Entrée : Couple à la roue")
    plt.ylabel("Nm")
    plt.legend()
    plt.grid(True)
    plt.xlim(0, duration)

    # Figure 2 : Efficacité (CosPhi)
    plt.figure("Open Loop - Efficacité", figsize=(10, 8))
    plt.subplot(2, 1, 1)
    for m in modes:
        plt.plot(results[m]["time"], results[m]["cosphi_av"], label=m)
    plt.title("Cos Phi Avant")
    plt.ylim(0, 1.1)
    plt.legend()
    plt.grid(True)
    plt.xlim(0, duration)
    
    plt.subplot(2, 1, 2)
    for m in modes:
        plt.plot(results[m]["time"], results[m]["cosphi_ar"], label=m)
    plt.title("Cos Phi Arrière")
    plt.ylim(0, 1.1)
    plt.grid(True)
    plt.xlim(0, duration)

    plt.show()

if __name__ == "__main__":
    main()