import matplotlib.pyplot as plt
from vehicle_sim.simulation import Simulation
from vehicle_sim.config import SimulationConfig, VehicleConfig

import sys
import os

# Cette ligne remonte d'un niveau (hors de examples) et entre dans src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

def main():
    # Setup configs
    sim_cfg = SimulationConfig(dt=0.01, duration=15.0)
    veh_cfg = VehicleConfig() # Utilise les valeurs par défaut du EZDolly (5637kg, etc.)

    # Initialize Simulation
    sim = Simulation(sim_cfg, veh_cfg)

    # Run Simulation
    results = sim.run(target_speed=15.0/3.6) # 15 km/h

    # --- NOUVEL AFFICHAGE MULTI-FENÊTRES ---

    # Fenêtre 1 : Dynamique du véhicule
    plt.figure(figsize=(10, 6))
    plt.subplot(2, 1, 1)
    plt.plot(results["time"], [v*3.6 for v in results["velocity"]], label="Vitesse réelle")
    plt.axhline(15.0, color='r', linestyle='--', label="Cible")
    plt.title("Réponse en vitesse")
    plt.ylabel("km/h")
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 1, 2)
    plt.plot(results["time"], results["cmd_torque"], color='orange', label="Couple Global Demandé")
    plt.title("Commande de couple total")
    plt.ylabel("Nm")
    plt.xlabel("Temps (s)")
    plt.legend()
    plt.grid(True)

    # Fenêtre 2 : Analyse de l'Allocation et Efficacité
    plt.figure(figsize=(10, 6))
    plt.subplot(2, 1, 1)
    plt.plot(results["time"], results["torque_fl"], label="Moteur AV")
    plt.plot(results["time"], results["torque_rl"], label="Moteur AR")
    plt.title("Répartition du couple (Optimisation)")
    plt.ylabel("Couple Moteur (Nm)")
    plt.legend()
    plt.grid(True)

    # Note : Pour tracer le CosPhi, assure-toi de l'avoir ajouté 
    # dans le dictionnaire history de simulation.py
    if "cosphi_av" in results:
        plt.subplot(2, 1, 2)
        plt.plot(results["time"], results["cosphi_av"], color='green', label="CosPhi Avant")
        plt.title("Efficacité énergétique")
        plt.ylabel("Cos Phi")
        plt.xlabel("Temps (s)")
        plt.legend()
        plt.grid(True)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()