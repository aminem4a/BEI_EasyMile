"""
Code complet demandé :
1) Mapping cos(phi) = f(Couple, Vitesse) via tes 2 ajustements (positif / négatif) + collage piecewise
2) Classe d'optimisation (allocation AV/AR) qui utilise f_final
3) Génération d'une consigne (couple_total(t), vitesse_rpm(t)) pour tester l'optimisation
4) Plots de vérification
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from scipy.optimize import minimize_scalar


# =============================================================================
# Donnees (T=couple, S=vitesse, Z=cos(phi))
# =============================================================================
T_data = np.array([16.583, 18.988, 29.45, 7.644, 32.495, 38.412, 33.2, 15.582, 40.85, 49.164,
                   13.289, 12.272, 60.795, 53.025, 117.075, 47.7, 28.035, 2.5, 10.735, 61.824,
                   23.381, 6.666, 36.181, 57.018, 10.094, 19.488, 48.609, 22.698, 7.505, 31.6,
                   17.85, 28.987, 53.346, 53.53, 52.38, 5.05, 9.595, 20.37, 23.808, 27.371,
                   69.319, 14.847, 24.096, 36.865, 72, 24.735, 31.824, 40.788, 44.352, 46.272,
                   62.304, 76.1, 76.538, 87.87, 83.125, 99.716, 28.684, 87.138, 13.802,
                   37.595, 42.33, 42.874, 52, 63.08, 67.32, 81.279, 86.632, 87.87, 95.76,
                   13.39, 20.055, 22.374, 23.816, 45.374, 44.175, 51.1, 62.296, 73.632,
                   83.712, 18.236, 31.556, 45.186, 46.53, 58.092, 65.52, 77, 93.66, 98.098,
                   29.498, 32.11, 38.178, 54.136, 61.74, 68.02, 16.463, 48.609, 39.382])

S_data = np.array([2852.85, 1468.7, 1230, 2347.2, 956.87, 1260.72, 1607.55, 1890.05, 894.34,
                   596.55, 2195.96, 2371.65, 1432.6, 958.65, 1183.35, 1467.61, 1919.4, 4425.2,
                   2630.4, 938.08, 2238.6, 4092.43, 1821, 1187.76, 2979.2, 2493.63, 1795,
                   4122.82, 3636, 2126.05, 2834, 2802.8, 2609.84, 2687.36, 1705.25, 4382.65,
                   4425.2, 4066.02, 3392.64, 3498.66, 2209.66, 4309.52, 2710, 2126.05,
                   1183.05, 2882.88, 3473.91, 3079.65, 2269.44, 2558.78, 1783, 887, 1426.56,
                   1403.15, 1709.73, 1116.47, 2530.5, 1451.38, 3573.9, 2469.94, 2470,
                   2754.05, 2163.84, 1942.75, 1564.16, 1233.44, 1819.65, 550.2, 1517.19,
                   4387.76, 3763.2, 3991.55, 3063.06, 2096, 2922.3, 1975.05, 1842.67,
                   1926.6, 875.67, 4058.48, 2573.76, 2744.56, 2849, 2020.76, 2209.66,
                   2075.84, 1225.12, 1392.96, 3074.55, 2907.66, 2586.99, 2271.74, 2309,
                   1757, 3983.04, 2376.53, 2546.88])

Z_data = np.array([0.2037, 0.4264, 0.5562, 0.5916, 0.5723, 0.5917, 0.6014, 0.6534, 0.627,
                   0.6767, 0.735, 0.6745, 0.7056, 0.7154, 0.7008, 0.7178, 0.7828, 0.8085,
                   0.78, 0.8, 0.7872, 0.83, 0.83, 0.8051, 0.798, 0.8484, 0.8148, 0.867,
                   0.8772, 0.8858, 0.8787, 0.87, 0.9135, 0.8439, 0.8526, 0.836, 0.88, 0.88,
                   0.8976, 0.8976, 0.88, 0.8633, 0.8633, 0.9167, 0.8544, 0.855, 0.927, 0.855,
                   0.9, 0.927, 0.873, 0.918, 0.918, 0.927, 0.945, 0.945, 0.9464, 0.9373,
                   0.9476, 0.966, 0.966, 0.9292, 0.8832, 0.9108, 0.966, 0.9384, 0.8832,
                   0.9384, 0.92, 0.9021, 0.9021, 0.93, 0.9021, 0.9579, 0.9486, 0.9579,
                   0.9207, 0.9765, 0.8835, 0.893, 0.987, 0.9494, 0.893, 0.94, 0.9588, 0.987,
                   0.9024, 0.893, 0.912, 0.912, 0.9792, 0.96, 0.9312, 0.96, 0.9409, 0.9409,
                   0.9595])


# =============================================================================
# Mapping "full" :
# Z = a*T^2 + b*S^2 + c*T*S + d*T + e*S + f
# + 2 fits (T>0 et T<0 miroir) + collage piecewise
# =============================================================================
def build_A_full(T, S):
    T = np.asarray(T).ravel()
    S = np.asarray(S).ravel()
    return np.column_stack([T**2, S**2, T*S, T, S, np.ones_like(T)])

def fit_full(T, S, Z):
    A = build_A_full(T, S)
    coeffs, _, _, _ = np.linalg.lstsq(A, Z, rcond=None)
    Z_hat = A @ coeffs
    err = Z_hat - Z
    rmse = float(np.sqrt(np.mean(err**2)))
    mae  = float(np.mean(np.abs(err)))
    denom = float(np.sum((Z - np.mean(Z))**2))
    r2 = float(1.0 - (np.sum(err**2) / denom)) if denom > 0 else float("nan")
    return coeffs, rmse, mae, r2

def predict_full(coeffs, T, S):
    T = np.asarray(T)
    S = np.asarray(S)
    if T.shape != S.shape:
        raise ValueError("T et S doivent avoir la meme forme.")
    A = build_A_full(T, S)
    return (A @ coeffs).reshape(T.shape)

def print_fit(name, coeffs, rmse, mae, r2):
    labels = ["a(T^2)", "b(S^2)", "c(TS)", "d(T)", "e(S)", "f(1)"]
    print("\n" + "=" * 72)
    print(name)
    print("=" * 72)
    for lab, val in zip(labels, coeffs):
        print(f"{lab:<8s} = {val: .6e}")
    print("-" * 72)
    print(f"RMSE = {rmse:.6f} | MAE = {mae:.6f} | R2 = {r2:.6f}")
    print("=" * 72)

def build_mapping_two_sides(T_pos, S_pos, Z_pos, grid_n=130, clip_01=False, show_map_plot=True):
    """
    Retourne f_final(T,S) + (coeff_pos, coeff_neg).
    """
    # Fit coté positif
    coeff_pos, rmse_pos, mae_pos, r2_pos = fit_full(T_pos, S_pos, Z_pos)

    # Fit coté negatif via miroir
    T_neg = -T_pos
    coeff_neg, rmse_neg, mae_neg, r2_neg = fit_full(T_neg, S_pos, Z_pos)

    print_fit("AJUSTEMENT #1 (Couples positifs)", coeff_pos, rmse_pos, mae_pos, r2_pos)
    print_fit("AJUSTEMENT #2 (Couples negatifs - donnees miroir)", coeff_neg, rmse_neg, mae_neg, r2_neg)

    def f_final(T, S):
        T = np.asarray(T)
        S = np.asarray(S)
        Zp = predict_full(coeff_pos, T, S)
        Zn = predict_full(coeff_neg, T, S)
        Z = np.where(T >= 0, Zp, Zn)
        if clip_01:
            Z = np.clip(Z, 0.0, 1.0)
        return Z

    if show_map_plot:
        Tmax = float(np.max(np.abs(T_pos)))
        t_grid = np.linspace(-Tmax, Tmax, grid_n)
        s_grid = np.linspace(float(np.min(S_pos)), float(np.max(S_pos)), grid_n)
        Tm, Sm = np.meshgrid(t_grid, s_grid)
        Zm = f_final(Tm, Sm)

        fig = plt.figure(figsize=(11, 7), constrained_layout=True)
        ax = fig.add_subplot(111, projection="3d")
        surf = ax.plot_surface(Tm, Sm, Zm, cmap="viridis", alpha=0.62, edgecolor="none")
        ax.plot_wireframe(Tm, Sm, Zm, rstride=10, cstride=10, linewidth=0.35, alpha=0.25)

        ax.scatter(T_pos,  S_pos, Z_pos, s=18, color="red",    label="Mesures (T>0)")
        ax.scatter(-T_pos, S_pos, Z_pos, s=18, color="orange", label="Miroir (T<0)")

        ax.set_xlabel("Couple (Nm)")
        ax.set_ylabel("Vitesse (rpm)")
        ax.set_zlabel("Cos(phi)")
        ax.set_title("Mapping final (collage fit + / fit -)")
        ax.view_init(elev=22, azim=-55)
        cbar = fig.colorbar(surf, ax=ax, shrink=0.62, pad=0.06)
        cbar.set_label("Cos(phi)")
        ax.legend(loc="upper right")
        plt.show()

    return f_final, coeff_pos, coeff_neg


# =============================================================================
# Classe optimisation : allocation AV/AR (par roue)
# =============================================================================
@dataclass
class TorqueAllocator:
    """
    Allocation sur un véhicule avec symétrie gauche/droite :
      2*Cav + 2*Car = C_total

    Cav = couple par roue avant, Car = couple par roue arrière.
    Objectif (max):
      J = a1*cosphi(Cav,V) + a2*cosphi(Car,V), a1>a2, a1+a2=1
    """
    cosphi_map: callable           # f_final(T,S)
    Cmax_per_wheel: float          # borne couple par roue
    a1: float = 0.7
    a2: float = 0.3
    allow_regen: bool = True
    smooth_lambda: float = 0.0     # pénalité (Cav-Cav_prev)^2

    def __post_init__(self):
        if self.a1 <= 0 or self.a2 <= 0:
            raise ValueError("a1 et a2 doivent être > 0.")
        s = self.a1 + self.a2
        self.a1 /= s
        self.a2 /= s

    def _feasible_interval_for_Cav(self, C_total: float):
        """
        Car = C_total/2 - Cav
        Contraintes:
          allow_regen=True  : Cav,Car ∈ [-Cmax, +Cmax]
          allow_regen=False : Cav,Car ∈ [0, +Cmax]
        """
        Cmax = self.Cmax_per_wheel
        half = 0.5 * C_total

        if self.allow_regen:
            lo = max(-Cmax, half - Cmax)
            hi = min(+Cmax, half + Cmax)
        else:
            lo = max(0.0, half - Cmax)
            hi = min(Cmax, half)

        if lo > hi:
            return None
        return lo, hi

    def allocate(self, C_total: float, speed_rpm: float, Cav_prev: float | None = None):
        interval = self._feasible_interval_for_Cav(C_total)
        Cmax = self.Cmax_per_wheel

        if interval is None:
            Cav = np.clip(C_total / 4.0, -Cmax if self.allow_regen else 0.0, Cmax)
            Car = np.clip(C_total / 4.0, -Cmax if self.allow_regen else 0.0, Cmax)
            eta_f = float(self.cosphi_map(Cav, speed_rpm))
            eta_r = float(self.cosphi_map(Car, speed_rpm))
            score = self.a1 * eta_f + self.a2 * eta_r
            return {"Cav": Cav, "Car": Car, "eta_front": eta_f, "eta_rear": eta_r, "score": score, "status": "SATURATED"}

        lo, hi = interval

        def objective(Cav):
            Car = 0.5 * C_total - Cav
            eta_f = float(self.cosphi_map(Cav, speed_rpm))
            eta_r = float(self.cosphi_map(Car, speed_rpm))
            score = self.a1 * eta_f + self.a2 * eta_r

            if self.smooth_lambda > 0.0 and Cav_prev is not None:
                score -= self.smooth_lambda * (Cav - Cav_prev) ** 2

            return -score

        res = minimize_scalar(objective, bounds=(lo, hi), method="bounded")
        Cav_opt = float(res.x)
        Car_opt = float(0.5 * C_total - Cav_opt)

        eta_f = float(self.cosphi_map(Cav_opt, speed_rpm))
        eta_r = float(self.cosphi_map(Car_opt, speed_rpm))
        score = self.a1 * eta_f + self.a2 * eta_r

        return {"Cav": Cav_opt, "Car": Car_opt, "eta_front": eta_f, "eta_rear": eta_r, "score": score, "status": "OK"}


# =============================================================================
# Consignes test (couple_total(t), vitesse_rpm(t))
# =============================================================================
def generate_test_reference(duration_s=60.0, dt=0.1, torque_peak_total=360.0,
                            speed_min=None, speed_max=None):
    """
    Couple_total : accel -> maintien -> freinage (regen) -> oscillation faible
    Vitesse_rpm  : montée -> palier -> descente -> oscillation
    """
    t = np.arange(0.0, duration_s + 1e-12, dt)

    if speed_min is None:
        speed_min = float(np.min(S_data))
    if speed_max is None:
        speed_max = float(np.max(S_data))

    speed = np.empty_like(t)
    torque = np.empty_like(t)

    for i, ti in enumerate(t):
        # speed
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

        # torque total
        if ti < 12:
            torque[i] = torque_peak_total * (ti / 12.0)
        elif ti < 28:
            torque[i] = 0.35 * torque_peak_total
        elif ti < 44:
            torque[i] = 0.35 * torque_peak_total - 0.90 * torque_peak_total * ((ti - 28.0) / 16.0)
        else:
            torque[i] = 0.10 * torque_peak_total * np.sin(2.0 * np.pi * (ti - 44.0) / 4.0)

    return t, speed, torque


# =============================================================================
# MAIN : mapping + optimisation + test
# =============================================================================
def main():
    # 1) Mapping f_final (ton mapping LS 2 fits)
    f_final, coeff_pos, coeff_neg = build_mapping_two_sides(
        T_data, S_data, Z_data,
        grid_n=130,
        clip_01=False,
        show_map_plot=False  # mets True si tu veux revoir la surface
    )

    # 2) Optimiseur (borne couple par roue)
    Cmax_per_wheel = float(np.max(np.abs(T_data)))  # ~117 Nm (cohérent avec tes données)
    allocator = TorqueAllocator(
        cosphi_map=f_final,
        Cmax_per_wheel=Cmax_per_wheel,
        a1=0.7,
        a2=0.3,
        allow_regen=True,
        smooth_lambda=0.0
    )

    # 3) Consignes test
    t, speed_rpm, C_total = generate_test_reference(duration_s=60.0, dt=0.1, torque_peak_total=360.0)

    # 4) Boucle d'optimisation
    Cav = np.zeros_like(t)
    Car = np.zeros_like(t)
    eta_f = np.zeros_like(t)
    eta_r = np.zeros_like(t)
    score = np.zeros_like(t)
    status = np.empty(t.shape, dtype=object)

    Cav_prev = None
    for k in range(len(t)):
        out = allocator.allocate(C_total[k], speed_rpm[k], Cav_prev=Cav_prev)
        Cav[k] = out["Cav"]
        Car[k] = out["Car"]
        eta_f[k] = out["eta_front"]
        eta_r[k] = out["eta_rear"]
        score[k] = out["score"]
        status[k] = out["status"]
        Cav_prev = Cav[k]

    # 5) Check contrainte
    C_rec = 2.0 * Cav + 2.0 * Car
    max_err = float(np.max(np.abs(C_rec - C_total)))
    print("\n" + "=" * 88)
    print("TEST OPTIMISATION — Résumé")
    print("=" * 88)
    print(f"Cmax_per_wheel = {Cmax_per_wheel:.3f} Nm | allow_regen = {allocator.allow_regen}")
    print(f"Poids: a1={allocator.a1:.3f}, a2={allocator.a2:.3f}")
    print(f"Erreur max sur 2*Cav+2*Car=C_total : {max_err:.6e} Nm")
    print(f"Nb saturations : {int(np.sum(status == 'SATURATED'))} / {len(t)}")
    print("=" * 88 + "\n")

    # 6) Plots
    fig1, ax1 = plt.subplots(figsize=(11, 5), constrained_layout=True)
    ax1.plot(t, C_total, label="C_total demandé (Nm)")
    ax1.plot(t, 2*Cav, label="2*Cav (avant total)")
    ax1.plot(t, 2*Car, label="2*Car (arrière total)")
    ax1.set_title("Allocation de couple (demandé vs réparti)")
    ax1.set_xlabel("Temps (s)")
    ax1.set_ylabel("Couple (Nm)")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    fig2, ax2 = plt.subplots(figsize=(11, 5), constrained_layout=True)
    ax2.plot(t, speed_rpm, label="Vitesse (rpm)")
    ax2.set_title("Consigne de vitesse (rpm)")
    ax2.set_xlabel("Temps (s)")
    ax2.set_ylabel("rpm")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    fig3, ax3 = plt.subplots(figsize=(11, 5), constrained_layout=True)
    ax3.plot(t, eta_f, label="cos(phi) AV")
    ax3.plot(t, eta_r, label="cos(phi) AR")
    ax3.plot(t, score, label="score pondéré")
    ax3.set_title("cos(phi) et score (optimisation)")
    ax3.set_xlabel("Temps (s)")
    ax3.set_ylabel("[-]")
    ax3.grid(True, alpha=0.3)
    ax3.legend()

    # 7) Plot 3D : surface finale + points utilisés (AV / AR)
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    Tmax = float(np.max(np.abs(T_data)))
    t_grid = np.linspace(-Tmax, Tmax, 90)
    s_grid = np.linspace(float(np.min(S_data)), float(np.max(S_data)), 90)
    Tm, Sm = np.meshgrid(t_grid, s_grid)
    Zm = f_final(Tm, Sm)

    fig4 = plt.figure(figsize=(11, 7), constrained_layout=True)
    ax4 = fig4.add_subplot(111, projection="3d")
    surf = ax4.plot_surface(Tm, Sm, Zm, cmap="viridis", alpha=0.55, edgecolor="none")

    ax4.scatter(Cav, speed_rpm, f_final(Cav, speed_rpm), s=10, color="red", label="Points AV utilisés")
    ax4.scatter(Car, speed_rpm, f_final(Car, speed_rpm), s=10, color="orange", label="Points AR utilisés")

    ax4.set_title("Mapping final (collage) + points utilisés par l'optimisation")
    ax4.set_xlabel("Couple (Nm)")
    ax4.set_ylabel("Vitesse (rpm)")
    ax4.set_zlabel("Cos(phi)")
    ax4.view_init(elev=22, azim=-55)
    cbar = fig4.colorbar(surf, ax=ax4, shrink=0.62, pad=0.06)
    cbar.set_label("Cos(phi)")
    ax4.legend(loc="upper right")

    plt.show()


if __name__ == "__main__":
    main()
