"""
CODE COMPLET — Mapping (2 ajustements + collage) + Allocation LISSÉE + Scénario test + Analyse/Plots
--------------------------------------------------------------------------------------------------
Ce script contient :
1) Mapping cos(phi)=f(Couple,Vitesse) EXACTEMENT via tes moindres carrés :
   - Fit #1 sur T>0
   - Fit #2 sur T<0 (miroir)
   - Mapping final piecewise f_final(T,S)

2) Allocation de couple AV/AR (par roue) avec continuité (smooth) :
   - Maximisation : a1*cosphi(Cav,V) + a2*cosphi(Car,V)
   - Contrainte : 2*Cav + 2*Car = C_total
   - Continuité : pénalité (Cav - Cav_prev)^2
   - "Dérivabilité" (lissage de la dérivée) : pénalité (Cav - 2*Cav_prev + Cav_prev2)^2
   - Rate limiter : |Cav(k)-Cav(k-1)| <= dC_max

3) Génération de consignes test (couple_total(t), vitesse_rpm(t))

4) Courbes d'analyse (performance/observation) :
   - Consignes : C_total(t), speed(t)
   - Allocation : Cav(t), Car(t), totaux avant/arrière, reconstruction contrainte
   - "Smoothness" : dCav/dt (discret), d2Cav/dt2 (discret), idem pour Car
   - Qualité : cosphi avant/arrière, score pondéré
   - Activité des contraintes : saturation / rate-limited counts
   - Exploitation de la map : nuage (|C|,V,cosphi) utilisé pendant le test + histogrammes

"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from scipy.optimize import minimize_scalar


# =============================================================================
# 1) Donnees (T=couple, S=vitesse, Z=cos(phi))
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
                   29.498, 32.11, 38.178, 54.136, 61.74, 68.02, 16.463, 48.609, 39.382], dtype=float)

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
                   1757, 3983.04, 2376.53, 2546.88], dtype=float)

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
                   0.9595], dtype=float)


# =============================================================================
# 2) Mapping (2 fits + collage)
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
    mae  = float(np.mean(np.abs(err**2)))
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

def build_mapping_two_sides(T_pos, S_pos, Z_pos, clip_01=False):
    coeff_pos, rmse_pos, mae_pos, r2_pos = fit_full(T_pos, S_pos, Z_pos)
    coeff_neg, rmse_neg, mae_neg, r2_neg = fit_full(-T_pos, S_pos, Z_pos)

    print("\n" + "=" * 80)
    print(f"Fit + : RMSE={rmse_pos:.6f} | MAE={mae_pos:.6f} | R2={r2_pos:.6f}")
    print(f"Fit - : RMSE={rmse_neg:.6f} | MAE={mae_neg:.6f} | R2={r2_neg:.6f}")
    print("=" * 80 + "\n")

    def f_final(T, S):
        T = np.asarray(T)
        S = np.asarray(S)
        Zp = predict_full(coeff_pos, T, S)
        Zn = predict_full(coeff_neg, T, S)
        Z = np.where(T >= 0, Zp, Zn)
        if clip_01:
            Z = np.clip(Z, 0.0, 1.0)
        return Z

    return f_final


def plot_map_surface(f_final, grid_n=120, clip_01=False, title="MAP cos(phi)=f(Couple,Vitesse)"):
    Tmax = float(np.max(np.abs(T_data)))
    t_grid = np.linspace(-Tmax, Tmax, grid_n)
    s_grid = np.linspace(float(np.min(S_data)), float(np.max(S_data)), grid_n)
    Tm, Sm = np.meshgrid(t_grid, s_grid)
    Zm = f_final(Tm, Sm)
    if clip_01:
        Zm = np.clip(Zm, 0.0, 1.0)

    fig = plt.figure(figsize=(11, 7), constrained_layout=True)
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(Tm, Sm, Zm, cmap="viridis", alpha=0.62, edgecolor="none")

    ax.scatter(T_data, S_data, Z_data, s=18, color="red", label="Données (T>0)")
    ax.scatter(-T_data, S_data, Z_data, s=18, color="orange", label="Miroir (T<0)")

    ax.set_xlabel("Couple (Nm)")
    ax.set_ylabel("Vitesse (rpm)")
    ax.set_zlabel("Cos(phi)")
    ax.set_title(title)
    ax.view_init(elev=22, azim=-55)

    cbar = fig.colorbar(surf, ax=ax, shrink=0.62, pad=0.06)
    cbar.set_label("Cos(phi)")
    ax.legend(loc="upper right")
    plt.show()


# =============================================================================
# 3) Allocation smooth
# =============================================================================
@dataclass
class TorqueAllocatorSmooth:
    cosphi_map: callable
    Cmax_per_wheel: float
    a1: float = 0.7
    a2: float = 0.3
    allow_regen: bool = True

    lambda1: float = 5e-3
    lambda2: float = 5e-4
    dC_max: float = 5.0

    def __post_init__(self):
        s = self.a1 + self.a2
        self.a1 /= s
        self.a2 /= s

    def _feasible_interval_for_Cav(self, C_total: float):
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

    def allocate(self, C_total: float, speed_rpm: float,
                 Cav_prev: float | None,
                 Cav_prev2: float | None):
        interval0 = self._feasible_interval_for_Cav(C_total)
        Cmax = self.Cmax_per_wheel

        if interval0 is None:
            Cav = np.clip(C_total / 4.0, -Cmax if self.allow_regen else 0.0, Cmax)
            Car = np.clip(C_total / 4.0, -Cmax if self.allow_regen else 0.0, Cmax)
            eta_f = float(self.cosphi_map(Cav, speed_rpm))
            eta_r = float(self.cosphi_map(Car, speed_rpm))
            score = self.a1 * eta_f + self.a2 * eta_r
            return {"Cav": Cav, "Car": Car, "eta_front": eta_f, "eta_rear": eta_r, "score": score, "status": "SATURATED"}

        lo0, hi0 = interval0
        lo, hi = lo0, hi0

        if Cav_prev is not None and self.dC_max is not None and self.dC_max > 0.0:
            lo = max(lo, Cav_prev - self.dC_max)
            hi = min(hi, Cav_prev + self.dC_max)
            if lo > hi:
                Cav = float(np.clip(Cav_prev, lo0, hi0))
                Car = float(0.5 * C_total - Cav)
                eta_f = float(self.cosphi_map(Cav, speed_rpm))
                eta_r = float(self.cosphi_map(Car, speed_rpm))
                score = self.a1 * eta_f + self.a2 * eta_r
                return {"Cav": Cav, "Car": Car, "eta_front": eta_f, "eta_rear": eta_r, "score": score, "status": "RATE_LIMITED"}

        def objective(Cav):
            Car = 0.5 * C_total - Cav
            eta_f = float(self.cosphi_map(Cav, speed_rpm))
            eta_r = float(self.cosphi_map(Car, speed_rpm))
            score = self.a1 * eta_f + self.a2 * eta_r

            if Cav_prev is not None and self.lambda1 > 0.0:
                score -= self.lambda1 * (Cav - Cav_prev) ** 2

            if Cav_prev is not None and Cav_prev2 is not None and self.lambda2 > 0.0:
                dd = Cav - 2.0 * Cav_prev + Cav_prev2
                score -= self.lambda2 * (dd ** 2)

            return -score

        res = minimize_scalar(objective, bounds=(lo, hi), method="bounded")
        Cav_opt = float(res.x)
        Car_opt = float(0.5 * C_total - Cav_opt)

        eta_f = float(self.cosphi_map(Cav_opt, speed_rpm))
        eta_r = float(self.cosphi_map(Car_opt, speed_rpm))
        score = self.a1 * eta_f + self.a2 * eta_r
        return {"Cav": Cav_opt, "Car": Car_opt, "eta_front": eta_f, "eta_rear": eta_r, "score": score, "status": "OK"}


# =============================================================================
# 4) Consignes test
# =============================================================================
def generate_test_reference(duration_s=60.0, dt=0.1, torque_peak_total=360.0):
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


# =============================================================================
# 5) MAIN : allocation smooth + analyses + tracés de map
# =============================================================================
def main():
    dt = 0.1

    # Mapping final
    f_final = build_mapping_two_sides(T_data, S_data, Z_data, clip_01=False)

    # Tracé map (surface + points originaux) — demandé
    plot_map_surface(f_final, grid_n=120, clip_01=False, title="Map cos(phi)=f(Couple,Vitesse) (surface + points)")

    # Scénario test
    t, speed_rpm, C_total = generate_test_reference(duration_s=60.0, dt=dt, torque_peak_total=360.0)

    # Optimiseur
    Cmax_per_wheel = float(np.max(np.abs(T_data)))  # ~117 Nm
    allocator = TorqueAllocatorSmooth(
        cosphi_map=f_final,
        Cmax_per_wheel=Cmax_per_wheel,
        a1=0.7, a2=0.3,
        allow_regen=True,
        lambda1=5e-3,
        lambda2=5e-4,
        dC_max=5.0
    )

    # Boucle allocation
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

    # Indicateurs
    C_rec = 2.0 * Cav + 2.0 * Car
    err_constraint = C_rec - C_total
    max_err = float(np.max(np.abs(err_constraint)))
    sat_count = int(np.sum(status == "SATURATED"))
    rate_count = int(np.sum(status == "RATE_LIMITED"))

    dCav = np.gradient(Cav, dt)
    dCar = np.gradient(Car, dt)
    ddCav = np.gradient(dCav, dt)
    ddCar = np.gradient(dCar, dt)

    eps = 1e-9
    front_share = (2.0 * Cav) / (np.abs(C_total) + eps)

    eta_mean_weighted = allocator.a1 * eta_f + allocator.a2 * eta_r

    print("\n" + "=" * 96)
    print("RÉSUMÉ PERFORMANCE — Allocation smooth")
    print("=" * 96)
    print(f"dt = {dt:.3f} s | Cmax_per_wheel = {Cmax_per_wheel:.3f} Nm | allow_regen = {allocator.allow_regen}")
    print(f"Poids : a1={allocator.a1:.3f}, a2={allocator.a2:.3f}")
    print(f"Lissage : lambda1={allocator.lambda1:.2e}, lambda2={allocator.lambda2:.2e}, dC_max={allocator.dC_max:.3f} Nm/pas")
    print(f"Erreur max contrainte 2*Cav+2*Car=C_total : {max_err:.6e} Nm")
    print(f"SATURATED : {sat_count} / {len(t)} | RATE_LIMITED : {rate_count} / {len(t)}")
    print(f"eta pondéré : min={float(np.min(eta_mean_weighted)):.4f} | moy={float(np.mean(eta_mean_weighted)):.4f} | max={float(np.max(eta_mean_weighted)):.4f}")
    print("=" * 96 + "\n")

    # PLOTS — consignes
    fig1, ax1 = plt.subplots(figsize=(11, 5), constrained_layout=True)
    ax1.plot(t, C_total, label="C_total demandé (Nm)")
    ax1.set_title("Consigne de couple total")
    ax1.set_xlabel("Temps (s)")
    ax1.set_ylabel("Couple total (Nm)")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    fig2, ax2 = plt.subplots(figsize=(11, 5), constrained_layout=True)
    ax2.plot(t, speed_rpm, label="Vitesse (rpm)")
    ax2.set_title("Consigne de vitesse")
    ax2.set_xlabel("Temps (s)")
    ax2.set_ylabel("Vitesse (rpm)")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # PLOTS — allocation + contrainte
    fig3, ax3 = plt.subplots(figsize=(11, 5), constrained_layout=True)
    ax3.plot(t, Cav, label="Cav (par roue)")
    ax3.plot(t, Car, label="Car (par roue)")
    ax3.set_title("Couples alloués (par roue)")
    ax3.set_xlabel("Temps (s)")
    ax3.set_ylabel("Couple (Nm)")
    ax3.grid(True, alpha=0.3)
    ax3.legend()

    fig4, ax4 = plt.subplots(figsize=(11, 5), constrained_layout=True)
    ax4.plot(t, C_total, label="C_total demandé")
    ax4.plot(t, 2.0*Cav, label="2*Cav (avant total)")
    ax4.plot(t, 2.0*Car, label="2*Car (arrière total)")
    ax4.plot(t, C_rec, "--", label="2*Cav+2*Car (reconstruit)", alpha=0.8)
    ax4.set_title("Répartition avant/arrière et reconstruction de la contrainte")
    ax4.set_xlabel("Temps (s)")
    ax4.set_ylabel("Couple (Nm)")
    ax4.grid(True, alpha=0.3)
    ax4.legend()

    fig5, ax5 = plt.subplots(figsize=(11, 4), constrained_layout=True)
    ax5.plot(t, err_constraint, label="Erreur contrainte (reconstruit - demandé)")
    ax5.set_title("Erreur de contrainte")
    ax5.set_xlabel("Temps (s)")
    ax5.set_ylabel("Nm")
    ax5.grid(True, alpha=0.3)
    ax5.legend()

    # PLOTS — smoothness
    fig6, ax6 = plt.subplots(figsize=(11, 5), constrained_layout=True)
    ax6.plot(t, dCav, label="dCav/dt (Nm/s)")
    ax6.plot(t, dCar, label="dCar/dt (Nm/s)")
    ax6.set_title("Vitesse de variation des couples")
    ax6.set_xlabel("Temps (s)")
    ax6.set_ylabel("Nm/s")
    ax6.grid(True, alpha=0.3)
    ax6.legend()

    fig7, ax7 = plt.subplots(figsize=(11, 5), constrained_layout=True)
    ax7.plot(t, ddCav, label="d²Cav/dt² (Nm/s²)")
    ax7.plot(t, ddCar, label="d²Car/dt² (Nm/s²)")
    ax7.set_title("Seconde dérivée discrète (proxy de dérivabilité)")
    ax7.set_xlabel("Temps (s)")
    ax7.set_ylabel("Nm/s²")
    ax7.grid(True, alpha=0.3)
    ax7.legend()

    # PLOTS — cosphi / score / stratégie
    fig8, ax8 = plt.subplots(figsize=(11, 5), constrained_layout=True)
    ax8.plot(t, eta_f, label="cos(phi) AV")
    ax8.plot(t, eta_r, label="cos(phi) AR")
    ax8.plot(t, eta_mean_weighted, label="cos(phi) pondéré")
    ax8.set_title("cos(phi) obtenu via la map")
    ax8.set_xlabel("Temps (s)")
    ax8.set_ylabel("[-]")
    ax8.grid(True, alpha=0.3)
    ax8.legend()

    fig9, ax9 = plt.subplots(figsize=(11, 4), constrained_layout=True)
    ax9.plot(t, score, label="Score optimisé")
    ax9.set_title("Score d'optimisation")
    ax9.set_xlabel("Temps (s)")
    ax9.set_ylabel("[-]")
    ax9.grid(True, alpha=0.3)
    ax9.legend()

    fig10, ax10 = plt.subplots(figsize=(11, 4), constrained_layout=True)
    ax10.plot(t, front_share, label="Part avant = (2*Cav)/|C_total|")
    ax10.set_title("Part de couple envoyée à l'avant")
    ax10.set_xlabel("Temps (s)")
    ax10.set_ylabel("[-]")
    ax10.grid(True, alpha=0.3)
    ax10.legend()

    fig11, ax11 = plt.subplots(figsize=(11, 3.5), constrained_layout=True)
    code = np.zeros_like(t, dtype=float)
    code[status == "OK"] = 0.0
    code[status == "RATE_LIMITED"] = 1.0
    code[status == "SATURATED"] = 2.0
    ax11.plot(t, code, label="Status: 0=OK, 1=RATE_LIMITED, 2=SATURATED")
    ax11.set_title("Activation des contraintes (diagnostic)")
    ax11.set_xlabel("Temps (s)")
    ax11.set_ylabel("Code")
    ax11.set_yticks([0, 1, 2])
    ax11.grid(True, alpha=0.3)
    ax11.legend()

    # --- Nouveau : MAP + trajectoire utilisée (points d'exploitation)
    absC_used = np.abs(np.concatenate([Cav, Car]))
    V_used = np.concatenate([speed_rpm, speed_rpm])
    eta_used = np.concatenate([eta_f, eta_r])

    fig12 = plt.figure(figsize=(11, 7), constrained_layout=True)
    ax12 = fig12.add_subplot(111, projection="3d")

    # Surface map en fonction de |C| uniquement (projection utile)
    Cgrid = np.linspace(0.0, float(np.max(np.abs(T_data))), 80)
    Vgrid = np.linspace(float(np.min(S_data)), float(np.max(S_data)), 80)
    Cm, Vm = np.meshgrid(Cgrid, Vgrid)
    # On évalue f_final avec T>=0 car abs(C) ici
    Zm = f_final(Cm, Vm)

    surf2 = ax12.plot_surface(Cm, Vm, Zm, cmap="viridis", alpha=0.45, edgecolor="none")
    ax12.scatter(absC_used, V_used, eta_used, s=10, label="Points utilisés (AV+AR)")

    ax12.set_title("Map (|Couple|, Vitesse) + trajectoire réellement exploitée")
    ax12.set_xlabel("|Couple| (Nm)")
    ax12.set_ylabel("Vitesse (rpm)")
    ax12.set_zlabel("cos(phi)")
    ax12.view_init(elev=22, azim=-55)
    cbar2 = fig12.colorbar(surf2, ax=ax12, shrink=0.62, pad=0.06)
    cbar2.set_label("cos(phi)")
    ax12.legend(loc="upper right")

    plt.show()


if __name__ == "__main__":
    main()
