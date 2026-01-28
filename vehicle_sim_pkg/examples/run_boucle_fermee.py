import sys
import os
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.vehicle_sim.control.allocation import TorqueAllocator
from src.vehicle_sim.utils.data_loader import DataLoader

def main():
    # 1. Setup des chemins
    base_dir = os.path.dirname(os.path.dirname(__file__))
    data_dir = os.path.join(base_dir, "data")
    
    # FICHIER SCÉNARIO À CHOISIR ICI
    # Assure-toi que ce fichier existe dans data/ ou data/scenarios/
    scenario_filename = "nominal_driving_5kmh_unloaded.csv"
    
    scenario_path = os.path.join(data_dir, scenario_filename)
    if not os.path.exists(scenario_path):
        # Essai dans le sous-dossier scenarios si pas trouvé
        scenario_path = os.path.join(data_dir, "scenarios", scenario_filename)

    map_path = os.path.join(data_dir, "efficiency_map_clean.csv")

    loader = DataLoader(map_path)
    allocator = TorqueAllocator(loader)

    print(f"Lecture du scénario : {scenario_path}")

    # --- CORRECTION DU BUG D'UNPACKING ICI ---
    # On récupère 3 valeurs (Temps, Vitesse, Couple)
    # Le _ sert à ignorer la 3ème valeur si on recalcule le couple nous-mêmes,
    # ou alors on l'utilise directement (trq_ref).
    try:
        t_ref, v_ref, trq_ref_csv = loader.load_scenario(scenario_path)
    except ValueError:
        # Si jamais le loader n'en renvoie que 2 (ancienne version)
        t_ref, v_ref = loader.load_scenario(scenario_path)
        trq_ref_csv = np.zeros_like(t_ref) # Fallback

    # Pour l'exemple, on peut ignorer le couple du CSV et simuler une demande constante
    # OU utiliser celui du fichier. Ici, je prends une demande constante pour tester.
    trq_req = 100.0 # Demande de 100 Nm constante (modifiable)

    strategies = ["Inverse", "Piecewise", "Smooth", "Quadratic"]
    results = {s: {'power': [], 'cos_phi': []} for s in strategies}

    print("Simulation en cours...")
    
    # 2. Boucle Temporelle
    for i, t in enumerate(t_ref):
        rpm = (v_ref[i] * 60) / (2 * np.pi * 0.3) # Rayon roue approx 0.3m
        if rpm < 10: rpm = 10 # Évite div par 0
        
        # Si tu veux utiliser le couple du CSV, décommente la ligne ci-dessous :
        # trq_req = trq_ref_csv[i]

        for strat in strategies:
            # Pour Smooth, on a besoin du ratio précédent
            prev_ratio = 0.5 
            if i > 0 and len(results[strat]['power']) > 0:
                # (Simplification: on ne stocke pas le ratio dans results ici, on suppose 0.5)
                pass 

            res = allocator.optimize(strat, trq_req, rpm, prev_front_ratio=prev_ratio)
            
            # Stockage simple
            # Puissance élec = Puissance Méca + Pertes
            p_meca = (trq_req * v_ref[i] / 0.3) 
            p_elec = p_meca + res['P_loss']
            
            # Calcul CosPhi fictif (si P_elec > 0)
            cos_phi = p_meca / p_elec if p_elec > 1 else 0

            results[strat]['power'].append(p_elec)
            results[strat]['cos_phi'].append(cos_phi)

    # 3. Affichage
    plt.figure(figsize=(12, 8))
    
    styles_map = {
        'Inverse':   ('-',  4.0, 0.4),
        'Piecewise': ('--', 2.5, 0.8),
        'Smooth':    ('-.', 2.0, 1.0),
        'Quadratic': (':',  2.0, 1.0)
    }

    plt.subplot(2, 1, 1)
    for name, data in results.items():
        ls, lw, alpha = styles_map.get(name, ('-', 1.5, 1.0))
        plt.plot(t_ref, data['power'], label=name, 
                 linestyle=ls, linewidth=lw, alpha=alpha)
    plt.ylabel("Puissance Elec (W)")
    plt.title(f"Simulation sur {scenario_filename}")
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 1, 2)
    for name, data in results.items():
        ls, lw, alpha = styles_map.get(name, ('-', 1.5, 1.0))
        plt.plot(t_ref, data['cos_phi'], label=name,
                 linestyle=ls, linewidth=lw, alpha=alpha)
    plt.ylabel("Efficacité (Est.)")
    plt.xlabel("Temps (s)")
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()