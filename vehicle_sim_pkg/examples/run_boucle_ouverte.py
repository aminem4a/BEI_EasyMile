import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.append(os.path.join(PROJECT_ROOT, "src"))

from vehicle_sim.control.allocation import TorqueAllocator
from vehicle_sim.utils.data_loader import DataLoader
from vehicle_sim.models.vehicle import Vehicle

@dataclass
class SimConfig:
    motor_max_torque: float = 200.0
    strejc_k: float = 1.0
    strejc_n: float = 1.0
    strejc_ta: float = 0.05
    strejc_tu: float = 0.05
    ratio_reduction: float = 26.0
    wheel_radius: float = 0.24
    surface_frontale: float = 3.0
    cx: float = 0.7
    mass: float = 4900.0
    crr: float = 0.015
    friction_torque: float = 80.0

def run_simulation():
    filename = "nominal_driving_5kmh_unloaded.csv"
    
    t, trq_csv, v_csv = DataLoader.load_scenario_csv(filename)
    if t is None: return

    dt_mean = np.mean(np.diff(t)) if len(t) > 1 else 0.01
    cfg = SimConfig()
    
    # Chargement de la map (va afficher "✅ Map trouvée : ..." ou une erreur explicite)
    allocator = TorqueAllocator() 

    strats = ["Inverse", "Smooth", "Quadratic", "Piecewise"]
    results = {s: {"time": t, "trq_alloc_total": [], "cos_f": [], "cos_r": [], "cav": []} for s in strats}
    m_s_to_rpm = (cfg.ratio_reduction / cfg.wheel_radius) * (60.0 / (2.0 * np.pi))

    print(f"Simulation sur {len(t)} points...")

    for strat in strats:
        print(f"   -> {strat}")
        veh = Vehicle(cfg, dt=dt_mean)
        veh.velocity = v_csv[0]
        cav_p, cav_p2 = None, None
        
        for k in range(len(t)):
            C_req = trq_csv[k]
            
            # On utilise le RPM théorique du CSV pour taper correctement dans la map
            current_rpm = v_csv[k] * m_s_to_rpm
            rpm_safe = max(abs(current_rpm), 1.0)
            
            if strat == "Smooth":
                cav, _, ef, er, _ = allocator.solve_allocation(strat, C_req, rpm_safe, prev_cav=cav_p, prev_cav2=cav_p2)
                cav_p2 = cav_p
                cav_p = cav
            else:
                cav, _, ef, er, _ = allocator.solve_allocation(strat, C_req, rpm_safe)
            
            car = 0.5 * C_req - cav
            C_total_alloc = 2*cav + 2*car
            
            results[strat]["trq_alloc_total"].append(C_total_alloc)
            results[strat]["cos_f"].append(float(ef))
            results[strat]["cos_r"].append(float(er))
            results[strat]["cav"].append(cav)

    plot_analysis(t, trq_csv, results)

def plot_analysis(t, trq_csv, results):
    strats = list(results.keys())

    # --- Fig 1: Couple ---
    fig1, axs1 = plt.subplots(2, 2, figsize=(14, 8), sharex=True)
    fig1.canvas.manager.set_window_title("1. Couple Total")
    for i, s in enumerate(strats):
        ax = axs1.ravel()[i]
        c_alloc = np.array(results[s]["trq_alloc_total"])
        ax.plot(t, trq_csv, 'k-', alpha=0.5, label="Input")
        ax.plot(t, c_alloc, 'b--', label="Output")
        ax.set_title(s); ax.grid(True, alpha=0.3)
        if i==0: ax.legend()
    plt.tight_layout()

    # --- Fig 2: Répartition ---
    fig2, axs2 = plt.subplots(2, 2, figsize=(14, 8), sharex=True)
    fig2.canvas.manager.set_window_title("2. Répartition Essieux")
    for i, s in enumerate(strats):
        ax = axs2.ravel()[i]
        cav = np.array(results[s]["cav"])
        ctot = np.array(results[s]["trq_alloc_total"])
        ax.plot(t, 2*cav, 'b-', label="AV")
        ax.plot(t, ctot - 2*cav, 'r--', label="AR")
        ax.set_title(s); ax.grid(True, alpha=0.3)
        if i==0: ax.legend()
    plt.tight_layout()

    # --- Fig 3: Cos Phi ---
    fig3, axs3 = plt.subplots(2, 2, figsize=(14, 8), sharex=True)
    fig3.canvas.manager.set_window_title("3. CosPhi")
    for i, s in enumerate(strats):
        ax = axs3.ravel()[i]
        cf = np.nan_to_num(np.array(results[s]["cos_f"]), nan=0.0)
        cr = np.nan_to_num(np.array(results[s]["cos_r"]), nan=0.0)
        ax.plot(t, cf, 'b-', label="AV")
        ax.plot(t, cr, 'r--', alpha=0.7, label="AR")
        ax.set_title(s); ax.set_ylim(0, 1.05); ax.grid(True, alpha=0.3)
        if i==0: ax.legend(loc="lower right")
    plt.tight_layout()

    # --- Fig 4: Stats ---
    print("\n--- STATS ---")
    fig4, (ax_std, ax_mean) = plt.subplots(1, 2, figsize=(12, 6))
    fig4.canvas.manager.set_window_title("4. Stats")
    
    stds, means = [], []
    for s in strats:
        all_c = np.concatenate([results[s]["cos_f"], results[s]["cos_r"]])
        all_c = all_c[np.isfinite(all_c)]
        # On ignore les valeurs nulles (moteurs éteints) pour la stat de qualité
        all_c_active = all_c[all_c > 0.05] 
        
        if len(all_c_active) > 0:
            v_std = np.std(all_c_active)
            v_mean = np.mean(all_c_active)
        else:
            v_std, v_mean = 0, 0
        
        print(f"{s:<10} | Moyenne (Actif): {v_mean:.3f} | Ecart-Type: {v_std:.3f}")
        stds.append(v_std); means.append(v_mean)

    ax_std.bar(strats, stds, alpha=0.7, color='purple')
    ax_std.set_title("Stabilité (Écart-Type)")
    ax_mean.bar(strats, means, alpha=0.7, color='orange')
    ax_mean.set_title("Efficacité Moyenne (en fonctionnement)")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_simulation()