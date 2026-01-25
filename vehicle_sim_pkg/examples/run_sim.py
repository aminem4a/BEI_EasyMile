from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from vehicle_sim.config import SimulationConfig, VehicleConfig, AllocationConfig, ControllerConfig
from vehicle_sim.simulation_open_loop import SimulationOpenLoop


def build_speed_profile(dt: float, duration: float) -> np.ndarray:
    """Exemple simple: ramp + palier + oscillation."""
    t = np.arange(0.0, duration + 1e-12, dt)
    v = np.empty_like(t)

    for i, ti in enumerate(t):
        if ti < 5.0:
            v[i] = 0.0 + 20.0 * (ti / 5.0)         # 0 -> 20 m/s
        elif ti < 10.0:
            v[i] = 20.0                            # palier
        else:
            v[i] = 18.0 + 2.0 * np.sin(2*np.pi*(ti-10.0)/3.0)  # oscillation

    return t, v


def main():
    # 1) Racine projet + chemin data (robuste)
    ROOT = Path(__file__).resolve().parents[1]  # vehicle_sim_pkg/
    csv_path = ROOT / "data" / "raw" / "data_map_C_V_cosphi.csv"

    # 2) Configs (c’est normal que run_sim.py définisse les paramètres du run)
    sim_cfg = SimulationConfig(dt=0.1, duration=20.0)
    veh_cfg = VehicleConfig(mass=1200.0, wheel_radius=0.30)

    ctrl_cfg = ControllerConfig(kp=120.0, ki=2.0, kd=0.0)

    alloc_cfg = AllocationConfig(
        a1=0.7, a2=0.3,
        allow_regen=True,
        lambda1=5e-3,
        lambda2=5e-4,
        dC_max=5.0,
        clip_01=False,
        # si tu veux, tu peux laisser la simu le déduire via la data
        Cmax_per_wheel=None,
    )

    # 3) Scénario de référence (profil vitesse)
    t, v_ref = build_speed_profile(sim_cfg.dt, sim_cfg.duration)

    # 4) Lancer simulation
    sim = Simulation(sim_cfg, veh_cfg, ctrl_cfg, alloc_cfg, csv_path=csv_path)
    results = sim.run(speed_ref=v_ref)

    # 5) Plots
    plt.figure(figsize=(10, 4))
    plt.plot(t, v_ref, label="v_ref")
    plt.plot(results["time"], results["speed"], label="v")
    plt.grid(True, alpha=0.3)
    plt.title("Suivi vitesse")
    plt.xlabel("Temps (s)")
    plt.ylabel("Vitesse (m/s)")
    plt.legend()

    plt.figure(figsize=(10, 4))
    plt.plot(results["time"], results["torque_total"], label="Couple total")
    plt.grid(True, alpha=0.3)
    plt.title("Couple total commandé")
    plt.xlabel("Temps (s)")
    plt.ylabel("Nm")
    plt.legend()

    plt.figure(figsize=(10, 4))
    plt.plot(results["time"], results["Cav"], label="Cav (par roue)")
    plt.plot(results["time"], results["Car"], label="Car (par roue)")
    plt.grid(True, alpha=0.3)
    plt.title("Allocation AV/AR")
    plt.xlabel("Temps (s)")
    plt.ylabel("Nm")
    plt.legend()

    plt.show()


if __name__ == "__main__":
    main()
