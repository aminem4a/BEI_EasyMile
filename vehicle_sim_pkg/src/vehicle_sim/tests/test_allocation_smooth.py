from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from traitement_data.mapping_cosphi import load_map_data, build_mapping_two_sides
from classes.torque_allocator_smooth import TorqueAllocatorSmooth


def generate_test_reference(S_data, duration_s=60.0, dt=0.1, torque_peak_total=360.0):
    t = np.arange(0.0, duration_s + 1e-12, dt)
    speed_min = float(np.min(S_data))
    speed_max = float(np.max(S_data))

    speed = np.empty_like(t)
    torque = np.empty_like(t)

    for i, ti in enumerate(t):
        if ti < 15:
            speed[i] = speed_min + (speed_max - speed_min) * (ti / 15.0)
        elif ti < 30:
            speed[i] = speed_max
        elif ti < 45:
            speed[i] = speed_max - 0.65 * (speed_max - speed_min) * ((ti - 30.0) / 15.0)
        else:
            base = speed_min + 0.35 * (speed_max - speed_min)
            amp = 0.12 * (speed_max - speed_min)
            speed[i] = base + amp * np.sin(2.0 * np.pi * (ti - 45.0) / 6.0)

        if ti < 12:
            torque[i] = torque_peak_total * (ti / 12.0)
        elif ti < 28:
            torque[i] = 0.35 * torque_peak_total
        elif ti < 44:
            torque[i] = 0.35 * torque_peak_total - 0.90 * torque_peak_total * ((ti - 28.0) / 16.0)
        else:
            torque[i] = 0.10 * torque_peak_total * np.sin(2.0 * np.pi * (ti - 44.0) / 4.0)

    return t, speed, torque


def plot_series(t, series, title, ylabel, xlabel="Temps (s)"):
    fig, ax = plt.subplots(figsize=(11, 5), constrained_layout=True)
    for y, label in series:
        ax.plot(t, y, label=label)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend()
    return fig, ax


def main():
    csv_path = "data_map_C_V_cos(phi).csv"  # adapte le chemin si besoin
    dt = 0.1

    T_data, S_data, Z_data = load_map_data(csv_path)
    f_final = build_mapping_two_sides(T_data, S_data, Z_data, clip_01=False, verbose=True)

    t, speed_rpm, C_total = generate_test_reference(S_data, duration_s=60.0, dt=dt, torque_peak_total=360.0)

    Cmax_per_wheel = float(np.max(np.abs(T_data)))
    allocator = TorqueAllocatorSmooth(
        cosphi_map=f_final,
        Cmax_per_wheel=Cmax_per_wheel,
        a1=0.7, a2=0.3,
        allow_regen=True,
        lambda1=5e-3,
        lambda2=5e-4,
        dC_max=5.0,
    )

    Cav = np.zeros_like(t)
    Car = np.zeros_like(t)
    eta_f = np.zeros_like(t)
    eta_r = np.zeros_like(t)
    score = np.zeros_like(t)
    status = np.empty(t.shape, dtype=object)

    Cav_prev = None
    Cav_prev2 = None

    for k in range(len(t)):
        out = allocator.allocate(C_total[k], speed_rpm[k], Cav_prev=Cav_prev, Cav_prev2=Cav_prev2)
        Cav[k] = out["Cav"]
        Car[k] = out["Car"]
        eta_f[k] = out["eta_front"]
        eta_r[k] = out["eta_rear"]
        score[k] = out["score"]
        status[k] = out["status"]
        Cav_prev2 = Cav_prev
        Cav_prev = Cav[k]

    C_rec = 2.0 * Cav + 2.0 * Car
    err_constraint = C_rec - C_total

    plot_series(t, [(C_total, "C_total demandé (Nm)")], "Consigne de couple total", "Couple total (Nm)")
    plot_series(t, [(speed_rpm, "Vitesse (rpm)")], "Consigne de vitesse", "Vitesse (rpm)")
    plot_series(t, [(Cav, "Cav (par roue)"), (Car, "Car (par roue)")], "Couples alloués (par roue)", "Couple (Nm)")
    plot_series(t, [(err_constraint, "Erreur contrainte")], "Erreur de contrainte", "Nm")
    plot_series(t, [(eta_f, "cos(phi) AV"), (eta_r, "cos(phi) AR")], "cos(phi) via la map", "[-]")
    plot_series(t, [(score, "Score optimisé")], "Score d'optimisation", "[-]")

    plt.show()


if __name__ == "__main__":
    main()
