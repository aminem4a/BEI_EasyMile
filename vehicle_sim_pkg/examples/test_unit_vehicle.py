import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Hack Path pour importer src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.vehicle_sim.vehicle import Vehicle
from src.vehicle_sim.utils.data_loader import DataLoader

def main():
    print("🚀 VALIDATION FORWARD : Couple Réel -> Vitesse Simulée")

    # 1. Chargement des données réelles
    # Remplace par le nom exact de ton fichier
    filename = "nominal_driving_5kmh_unloaded.csv" # Ou vitesseclasseurvehicule.csv si tu l'as mis dans data/
    
    base_dir = os.path.dirname(os.path.dirname(__file__))
    file_path = os.path.join(base_dir, "data", filename)
    if not os.path.exists(file_path):
         # Cherche aussi dans scenarios
         file_path = os.path.join(base_dir, "data", "scenarios", filename)
         
    if not os.path.exists(file_path):
        print(f"❌ Fichier introuvable : {filename}")
        return

    loader = DataLoader(file_path)
    # On récupère Temps, Vitesse Réelle, Couple Réel Total
    t_real, v_real, trq_input = loader.load_scenario(file_path)

    # 2. Initialisation du Véhicule Calibré
    veh = Vehicle()
    
    v_simulated = []
    
    # 3. Boucle de Simulation
    print(f"Simulation sur {len(t_real)} points...")
    
    # On calcule le dt dynamiquement car les CSV n'ont pas toujours un pas de temps fixe
    for i in range(len(t_real)):
        if i == 0:
            dt = 0.01 # Pas par défaut au début
        else:
            dt = t_real[i] - t_real[i-1]
            if dt <= 0: dt = 0.01 # Sécurité
        
        # On récupère le couple total à cet instant (input du pilote)
        torque_now = trq_input[i]
        
        # On met à jour le modèle physique
        # Note : Vehicle.update prend le couple TOTAL
        v_next, _, _ = veh.update(torque_now, dt)
        
        v_simulated.append(v_next) # Converti en km/h pour l'affichage si besoin

    v_simulated = np.array(v_simulated)

    # 4. Affichage Comparatif
    plt.figure(figsize=(12, 8))
    
    # Graphique Vitesse
    plt.subplot(2, 1, 1)
    plt.plot(t_real, v_real * 3.6, 'k-', linewidth=1.5, alpha=0.6, label='Vitesse Mesurée (Réelle)')
    plt.plot(t_real, v_simulated * 3.6, 'r--', linewidth=2.0, label='Vitesse Simulée (Modèle)')
    
    plt.title(f"Validation du Modèle Véhicule (Masse={veh.m}kg, Frottement={veh.friction_torque}Nm)")
    plt.ylabel("Vitesse [km/h]")
    plt.legend()
    plt.grid(True)

    # Graphique Couple (Entrée)
    plt.subplot(2, 1, 2)
    plt.plot(t_real, trq_input, 'b-', label='Couple Total Injecté')
    plt.ylabel("Couple [Nm]")
    plt.xlabel("Temps [s]")
    plt.title("Entrée du système")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()