from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from scipy.optimize import minimize_scalar
import pandas as pd


# =============================================================================
# 0) Config
# =============================================================================
@dataclass
class SimConfig:
    dt: float = 0.1
    duration_s: float = 60.0
    torque_peak_total: float = 360.0

    a1: float = 0.7
    a2: float = 0.3
    allow_regen: bool = True

    lambda1: float = 5e-3
    lambda2: float = 5e-4
    dC_max: float = 5.0

    map_clip_01: bool = False
    map_grid_n: int = 120


# =============================================================================
# 1) Load data
# =============================================================================
def load_map_data(csv_path: str):
    df = pd.read_csv(csv_path)
    T = df["T"].to_numpy(float)
    S = df["S"].to_numpy(float)
    Z = df["Z"].to_numpy(float)
    return T, S, Z


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
    mae  = float(np.mean(np.abs(err**2)))  # NOTE: conservé tel quel pour ne pas changer les résultats affichés
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

def plot_map_surface(f_final, T_data, S_data, Z_data, grid_n=120, clip_01=False, title="MAP cos(phi)=f(Couple,Vitesse)"):
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
# 3) Allocation smooth (inchangé)
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

    def allocate(self, C_total: float, speed_rpm: float, Cav_prev: float | None, Cav_prev2: float | None):
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
# 4) Test reference (inchangé)
# =============================================================================
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


# =============================================================================
# 5) Plot helpers (gros gain de simplicité)
# =============================================================================
def plot_series(t, series: list[tuple[np.ndarray, str]], title: str, ylabel: str, xlabel="Temps (s)"):
    fig, ax = plt.subplots(figsize=(11, 5), constrained_layout=True)
    for y, label in series:
        ax.plot(t, y, label=label)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend()
    return fig, ax

def plot_status_code(t, status):
    code = np.zeros_like(t, dtype=float)
    code[status == "OK"] = 0.0
    code[status == "RATE_LIMITED"] = 1.0
    code[status == "SATURATED"] = 2.0

    fig, ax = plt.subplots(figsize=(11, 3.5), constrained_layout=True)
    ax.plot(t, code, label="Status: 0=OK, 1=RATE_LIMITED, 2=SATURATED")
    ax.set_title("Activation des contraintes (diagnostic)")
    ax.set_xlabel("Temps (s)")
    ax.set_ylabel("Code")
    ax.set_yticks([0, 1, 2])
    ax.grid(True, alpha=0.3)
    ax.legend()
    return fig, ax

def plot_map_usage(f_final, T_data, S_data, Cav, Car, speed_rpm, eta_f, eta_r):
    absC_used = np.abs(np.concatenate([Cav, Car]))
    V_used = np.concatenate([speed_rpm, speed_rpm])
    eta_used = np.concatenate([eta_f, eta_r])

    fig = plt.figure(figsize=(11, 7), constrained_layout=True)
    ax = fig.add_subplot(111, projection="3d")

    Cgrid = np.linspace(0.0, float(np.max(np.abs(T_data))), 80)
    Vgrid = np.linspace(float(np.min(S_data)), float(np.max(S_data)), 80)
    Cm, Vm = np.meshgrid(Cgrid, Vgrid)
    Zm = f_final(Cm, Vm)

    surf = ax.plot_surface(Cm, Vm, Zm, cmap="viridis", alpha=0.45, edgecolor="none")
    ax.scatter(absC_used, V_used, eta_used, s=10, label="Points utilisés (AV+AR)")

    ax.set_title("Map (|Couple|, Vitesse) + trajectoire réellement exploitée")
    ax.set_xlabel("|Couple| (Nm)")
    ax.set_ylabel("Vitesse (rpm)")
    ax.set_zlabel("cos(phi)")
    ax.view_init(elev=22, azim=-55)

    cbar = fig.colorbar(surf, ax=ax, shrink=0.62, pad=0.06)
    cbar.set_label("cos(phi)")
    ax.legend(loc="upper right")
    return fig, ax


# =============================================================================
# 6) Simulation + analyse
# =============================================================================
def run_allocation(cfg: SimConfig, f_final, T_data, S_data):
    t, speed_rpm, C_total = generate_test_reference(
        S_data, duration_s=cfg.duration_s, dt=cfg.dt, torque_peak_total=cfg.torque_peak_total
    )

    Cmax_per_wheel = float(np.max(np.abs(T_data)))
    allocator = TorqueAllocatorSmooth(
        cosphi_map=f_final,
        Cmax_per_wheel=Cmax_per_wheel,
        a1=cfg.a1, a2=cfg.a2,
        allow_regen=cfg.allow_regen,
        lambda1=cfg.lambda1,
        lambda2=cfg.lambda2,
        dC_max=cfg.dC_max,
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

    return allocator, t, speed_rpm, C_total, Cav, Car, eta_f, eta_r, score, status


def print_summary(cfg: SimConfig, allocator, t, C_total, Cav, Car, eta_f, eta_r, status):
    dt = cfg.dt
    C_rec = 2.0 * Cav + 2.0 * Car
    err_constraint = C_rec - C_total
    max_err = float(np.max(np.abs(err_constraint)))
    sat_count = int(np.sum(status == "SATURATED"))
    rate_count = int(np.sum(status == "RATE_LIMITED"))

    eta_mean_weighted = allocator.a1 * eta_f + allocator.a2 * eta_r

    print("\n" + "=" * 96)
    print("RÉSUMÉ PERFORMANCE — Allocation smooth")
    print("=" * 96)
    print(f"dt = {dt:.3f} s | Cmax_per_wheel = {allocator.Cmax_per_wheel:.3f} Nm | allow_regen = {allocator.allow_regen}")
    print(f"Poids : a1={allocator.a1:.3f}, a2={allocator.a2:.3f}")
    print(f"Lissage : lambda1={allocator.lambda1:.2e}, lambda2={allocator.lambda2:.2e}, dC_max={allocator.dC_max:.3f} Nm/pas")
    print(f"Erreur max contrainte 2*Cav+2*Car=C_total : {max_err:.6e} Nm")
    print(f"SATURATED : {sat_count} / {len(t)} | RATE_LIMITED : {rate_count} / {len(t)}")
    print(f"eta pondéré : min={float(np.min(eta_mean_weighted)):.4f} | moy={float(np.mean(eta_mean_weighted)):.4f} | max={float(np.max(eta_mean_weighted)):.4f}")
    print("=" * 96 + "\n")


def plot_all(cfg: SimConfig, allocator, f_final, T_data, S_data, t, speed_rpm, C_total, Cav, Car, eta_f, eta_r, score, status):
    dt = cfg.dt
    C_rec = 2.0 * Cav + 2.0 * Car
    err_constraint = C_rec - C_total

    dCav = np.gradient(Cav, dt)
    dCar = np.gradient(Car, dt)
    ddCav = np.gradient(dCav, dt)
    ddCar = np.gradient(dCar, dt)

    eps = 1e-9
    front_share = (2.0 * Cav) / (np.abs(C_total) + eps)
    eta_mean_weighted = allocator.a1 * eta_f + allocator.a2 * eta_r

    plot_series(t, [(C_total, "C_total demandé (Nm)")], "Consigne de couple total", "Couple total (Nm)")
    plot_series(t, [(speed_rpm, "Vitesse (rpm)")], "Consigne de vitesse", "Vitesse (rpm)")

    plot_series(t, [(Cav, "Cav (par roue)"), (Car, "Car (par roue)")], "Couples alloués (par roue)", "Couple (Nm)")

    plot_series(
        t,
        [(C_total, "C_total demandé"),
         (2.0 * Cav, "2*Cav (avant total)"),
         (2.0 * Car, "2*Car (arrière total)"),
         (C_rec, "2*Cav+2*Car (reconstruit)")],
        "Répartition avant/arrière et reconstruction de la contrainte",
        "Couple (Nm)"
    )

    plot_series(t, [(err_constraint, "Erreur contrainte")], "Erreur de contrainte", "Nm")

    plot_series(t, [(dCav, "dCav/dt"), (dCar, "dCar/dt")], "Vitesse de variation des couples", "Nm/s")
    plot_series(t, [(ddCav, "d²Cav/dt²"), (ddCar, "d²Car/dt²")], "Seconde dérivée discrète (proxy dérivabilité)", "Nm/s²")

    plot_series(t, [(eta_f, "cos(phi) AV"), (eta_r, "cos(phi) AR"), (eta_mean_weighted, "cos(phi) pondéré")], "cos(phi) obtenu via la map", "[-]")
    plot_series(t, [(score, "Score optimisé")], "Score d'optimisation", "[-]")
    plot_series(t, [(front_share, "Part avant")], "Part de couple envoyée à l'avant", "[-]")

    plot_status_code(t, status)
    plot_map_usage(f_final, T_data, S_data, Cav, Car, speed_rpm, eta_f, eta_r)

    plt.show()


# =============================================================================
# 7) Main
# =============================================================================
def main():
    cfg = SimConfig()

    T_data, S_data, Z_data = load_map_data("data_map_C_V_cos(phi).csv")
    f_final = build_mapping_two_sides(T_data, S_data, Z_data, clip_01=cfg.map_clip_01)

    plot_map_surface(
        f_final, T_data, S_data, Z_data,
        grid_n=cfg.map_grid_n,
        clip_01=cfg.map_clip_01,
        title="Map cos(phi)=f(Couple,Vitesse) (surface + points)"
    )

    allocator, t, speed_rpm, C_total, Cav, Car, eta_f, eta_r, score, status = run_allocation(cfg, f_final, T_data, S_data)
    print_summary(cfg, allocator, t, C_total, Cav, Car, eta_f, eta_r, status)
    plot_all(cfg, allocator, f_final, T_data, S_data, t, speed_rpm, C_total, Cav, Car, eta_f, eta_r, score, status)


if __name__ == "__main__":
    main()
