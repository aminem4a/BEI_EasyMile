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

# --- NOUVELLE FONCTION : Calcul des métriques ---
def calculate_metrics(history, veh_cfg, dt):
    """Calcule l'énergie totale consommée (Wh) et le CosPhi moyen."""
    # 1. Vitesse moteur (rad/s)
    v_ms = np.array(history["velocity"])
    w_motor = (v_ms / veh_cfg.wheel_radius) * veh_cfg.ratio_reduction
    
    # 2. Couples Moteurs (Nm)
    t_fl = np.array(history["torque_fl"])
    t_rl = np.array(history["torque_rl"])
    
    # 3. Puissance Instantanée (Watts) = |Couple * Vitesse|
    # On prend la valeur absolue (conso pure)
    power_fl = np.abs(t_fl * w_motor)
    power_rl = np.abs(t_rl * w_motor)
    total_power = power_fl + power_rl
    
    # 4. Énergie (Wh)
    total_energy_wh = np.sum(total_power * dt) / 3600.0
    
    # 5. CosPhi Moyen (Moyenne simple des valeurs non nulles)
    cp_av = np.array(history["cosphi_av"])
    cp_ar = np.array(history["cosphi_ar"])
    avg_cosphi = (np.mean(cp_av) + np.mean(cp_ar)) / 2.0
    
    return total_energy_wh, avg_cosphi

def main():
    base_dir = os.path.dirname(os.path.dirname(__file__)) 
    data_dir = os.path.join(base_dir, "data")
    scenario_file = os.path.join(data_dir, "scenarios", "nominal_driving_13kmh_unloaded.csv")
    
    print(f"--- 🏆 Lancement VALIDATION COMPLÈTE (Graphiques + Chiffres) ---")
    
    # 1. Chargement
    loader = DataLoader(data_dir)
    t_ref, v_ref, trq_ref = loader.load_scenario(scenario_file)

    if len(t_ref) == 0:
        print("❌ Scénario vide.")
        return

    # Durée
    duration = min(t_ref[-1], 20.0)
    print(f"⏱️ Durée analysée : {duration}s")

    # Préparation Boucle Ouverte
    torque_profile = interp1d(t_ref, trq_ref, bounds_error=False, fill_value=0.0)
    sim_cfg = SimulationConfig(dt=0.01, duration=duration)
    veh_cfg = VehicleConfig()

    modes = ["inverse", "piecewise", "smooth", "quadratic"]
    results = {}
    metrics = {}

    # 2. Exécution des Simulations
    for m in modes:
        print(f"🚀 Simulation : {m.upper()}...")
        sim = Simulation(sim_cfg, veh_cfg, data_dir=data_dir)
        sim.allocator.mode = m
        # On utilise le mode Open Loop qui fonctionne bien maintenant
        res = sim.run_open_loop(torque_profile)
        results[m] = res
        
        # Calcul immédiat des scores
        e_wh, cp_avg = calculate_metrics(res, veh_cfg, sim_cfg.dt)
        metrics[m] = {"E": e_wh, "CP": cp_avg}

    # ==========================================
    # PARTIE 1 : TABLEAU DES RÉSULTATS (NOUVEAU)
    # ==========================================
    print("\n" + "="*65)
    print(f"{'STRATÉGIE':<15} | {'ÉNERGIE (Wh)':<15} | {'COSPHI MOYEN':<15} | {'GAIN %':<10}")
    print("-" * 65)
    
    ref_energy = metrics["inverse"]["E"] # Référence = Inverse
    
    for m in modes:
        e = metrics[m]["E"]
        cp = metrics[m]["CP"]
        # Calcul du gain (Combien on a économisé par rapport à Inverse)
        if ref_energy > 0:
            gain = ((ref_energy - e) / ref_energy) * 100.0
        else:
            gain = 0.0
            
        print(f"{m.upper():<15} | {e:.4f} Wh        | {cp:.3f}           | {gain:+.2f}%")
    print("="*65 + "\n")

    # ==========================================
    # PARTIE 2 : GRAPHIQUES (ANCIEN CODE CONSERVÉ)
    # ==========================================
    print("📊 Génération des graphiques...")
    plt.figure("Validation Complète", figsize=(10, 10))
    
    # Graphe 1 : Vitesse (Vérif modèle)
    plt.subplot(3, 1, 1)
    plt.plot(t_ref, v_ref * 3.6, 'k--', label="Réalité", linewidth=2, alpha=0.5)
    for m in modes:
        v_kmh = [x * 3.6 for x in results[m]["velocity"]]
        plt.plot(results[m]["time"], v_kmh, label=m)
    plt.title("Suivi de Vitesse (Boucle Ouverte)")
    plt.ylabel("km/h")
    plt.legend()
    plt.grid(True)
    plt.xlim(0, duration)

    # Graphe 2 : Couple (Vérif Entrée)
    plt.subplot(3, 1, 2)
    plt.plot(t_ref, trq_ref, 'k--', label="Consigne Totale (CSV)", alpha=0.3)
    for m in modes:
        # On affiche le couple AVANT pour voir la répartition
        plt.plot(results[m]["time"], results[m]["torque_fl"], label=f"Couple AV ({m})")
    plt.title("Répartition du Couple (Zoom sur l'Avant)")
    plt.ylabel("Nm")
    plt.legend()
    plt.grid(True)
    plt.xlim(0, duration)

    # Graphe 3 : CosPhi (Vérif Efficacité)
    plt.subplot(3, 1, 3)
    for m in modes:
        # Moyenne AV+AR pour lisibilité
        cp_moy = [(a+b)/2 for a,b in zip(results[m]["cosphi_av"], results[m]["cosphi_ar"])]
        plt.plot(results[m]["time"], cp_moy, label=m)
    plt.title("CosPhi Moyen Instantané")
    plt.xlabel("Temps (s)")
    plt.ylabel("Cos Phi")
    plt.ylim(0, 1.1)
    plt.grid(True)
    plt.xlim(0, duration)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()