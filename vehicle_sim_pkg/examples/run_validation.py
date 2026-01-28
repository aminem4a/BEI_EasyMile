import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.vehicle_sim.control.allocation import TorqueAllocator
from src.vehicle_sim.utils.data_loader import DataLoader

def main():
    # 1. Setup
    base_dir = os.path.dirname(os.path.dirname(__file__))
    data_dir = os.path.join(base_dir, "data")
    
    # Choix du fichier
    filename = "nominal_driving_5kmh_unloaded.csv"
    scenario_path = os.path.join(data_dir, filename)
    if not os.path.exists(scenario_path):
        scenario_path = os.path.join(data_dir, "scenarios", filename)

    map_path = os.path.join(data_dir, "efficiency_map_clean.csv")
    
    loader = DataLoader(map_path)
    allocator = TorqueAllocator(loader)

    print(f"Validation sur : {filename}")

    # Chargement (avec le fix unpack au cas où)
    try:
        t, v, trq_exp = loader.load_scenario(scenario_path)
    except ValueError:
        t, v = loader.load_scenario(scenario_path)
        trq_exp = np.zeros_like(t) # Fallback

    strategies = ["Inverse", "Piecewise", "Smooth", "Quadratic"]
    summary = {s: {'energy': 0, 'avg_cosphi': 0} for s in strategies}
    logs = {s: {'power': []} for s in strategies}

    dt = t[1] - t[0] if len(t) > 1 else 0.1

    # 2. Calculs
    for i in range(len(t)):
        rpm = (v[i] * 60) / (2 * np.pi * 0.3)
        if rpm < 10: rpm = 10
        T_req = trq_exp[i] 
        
        # Si le fichier CSV contient des 0 partout en couple, on force une valeur pour tester
        if np.all(trq_exp == 0):
            T_req = 50.0 # Valeur par défaut pour voir qqchose

        for strat in strategies:
            # Gestion basique du 'previous' pour smooth
            prev = 0.5 
            
            res = allocator.optimize(strat, T_req, rpm, prev_front_ratio=prev)
            
            p_meca = abs(T_req * v[i] / 0.3)
            p_elec = p_meca + res['P_loss']
            
            # Intégration Energie (W * s = Joules)
            summary[strat]['energy'] += p_elec * dt
            logs[strat]['power'].append(p_elec)

    # 3. Tableau Résultats
    print("\n--- RÉSULTATS ---")
    print(f"{'Stratégie':<15} | {'Energie (Wh)':<15} | {'Gain vs Inv':<15}")
    print("-" * 50)
    
    base_energy = summary['Inverse']['energy'] / 3600 # Wh
    
    for s in strategies:
        e_wh = summary[s]['energy'] / 3600
        gain = ((base_energy - e_wh) / base_energy) * 100 if base_energy > 0 else 0
        print(f"{s:<15} | {e_wh:.4f} Wh       | {gain:+.2f} %")

    # 4. Plot (Optionnel)
    plt.figure(figsize=(10, 6))
    
    styles_map = {
        'Inverse':   ('-',  4.0, 0.4),
        'Piecewise': ('--', 2.5, 0.8),
        'Smooth':    ('-.', 2.0, 1.0),
        'Quadratic': (':',  2.0, 1.0)
    }

    for name in strategies:
        ls, lw, alpha = styles_map.get(name, ('-', 1.5, 1.0))
        plt.plot(t, logs[name]['power'], label=name,
                 linestyle=ls, linewidth=lw, alpha=alpha)

    plt.title("Consommation Instantanée")
    plt.xlabel("Temps (s)")
    plt.ylabel("Puissance (W)")
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()