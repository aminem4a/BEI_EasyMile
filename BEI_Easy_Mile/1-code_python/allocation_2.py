from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Tuple, Dict, Any

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import minimize_scalar


# =============================================================================
# Données (T=couple, S=vitesse, Z=cos(phi))
# =============================================================================
df = pd.read_csv("data_map_C_V_cos(phi).csv")

T_data = df["T"].to_numpy(float)
S_data = df["S"].to_numpy(float)
Z_data = df["Z"].to_numpy(float)

# =============================================================================
# Mapping model: Z = a*T^2 + b*S^2 + c*T*S + d*T + e*S + f
# Fit côté T>=0 + fit côté T<0 (miroir) puis collage piecewise
# =============================================================================
def _build_A(T: np.ndarray, S: np.ndarray) -> np.ndarray:
    T = np.asarray(T).ravel()
    S = np.asarray(S).ravel()
    return np.column_stack([T**2, S**2, T*S, T, S, np.ones_like(T)])


def _fit_ls(T: np.ndarray, S: np.ndarray, Z: np.ndarray) -> Tuple[np.ndarray, Dict[str, float]]:
    A = _build_A(T, S)
    coeffs, _, _, _ = np.linalg.lstsq(A, Z, rcond=None)
    Z_hat = A @ coeffs
    err = Z_hat - Z

    rmse = float(np.sqrt(np.mean(err**2)))
    mae = float(np.mean(np.abs(err)))
    denom = float(np.sum((Z - np.mean(Z))**2))
    r2 = float(1.0 - (np.sum(err**2) / denom)) if denom > 0 else float("nan")

    return coeffs, {"rmse": rmse, "mae": mae, "r2": r2}


def _predict(coeffs: np.ndarray, T: np.ndarray, S: np.ndarray) -> np.ndarray:
    T = np.asarray(T)
    S = np.asarray(S)
    if T.shape != S.shape:
        raise ValueError("T et S doivent avoir la même forme.")
    A = _build_A(T, S)
    return (A @ coeffs).reshape(T.shape)


@dataclass(frozen=True)
class CosPhiMap:
    coeff_pos: np.ndarray
    coeff_neg: np.ndarray
    clip_01: bool = True

    def __call__(self, T: np.ndarray | float, S: np.ndarray | float) -> np.ndarray:
        T_arr = np.asarray(T)
        S_arr = np.asarray(S)
        Zp = _predict(self.coeff_pos, T_arr, S_arr)
        Zn = _predict(self.coeff_neg, T_arr, S_arr)
        Z = np.where(T_arr >= 0, Zp, Zn)
        if self.clip_01:
            Z = np.clip(Z, 0.0, 1.0)
        return Z


def build_mapping_two_sides(
    T_pos: np.ndarray,
    S_pos: np.ndarray,
    Z_pos: np.ndarray,
    clip_01: bool = True,
) -> Tuple[CosPhiMap, Dict[str, Dict[str, float]]]:
    # Fit côté T>0 (données)
    coeff_pos, met_pos = _fit_ls(T_pos, S_pos, Z_pos)

    # Fit côté T<0 (miroir)
    coeff_neg, met_neg = _fit_ls(-T_pos, S_pos, Z_pos)

    mapping = CosPhiMap(coeff_pos=coeff_pos, coeff_neg=coeff_neg, clip_01=clip_01)
    metrics = {"pos": met_pos, "neg": met_neg}
    return mapping, metrics


# =============================================================================
# Allocation AV/AR (par roue), véhicule symétrique :
# 2*Cav + 2*Car = C_total
# =============================================================================
@dataclass
class TorqueAllocator:
    cosphi_map: Callable[[float, float], np.ndarray]
    Cmax_per_wheel: float
    a1: float = 0.7
    a2: float = 0.3
    allow_regen: bool = True
    smooth_lambda: float = 0.0

    def __post_init__(self) -> None:
        if self.a1 <= 0 or self.a2 <= 0:
            raise ValueError("a1 et a2 doivent être > 0.")
        s = self.a1 + self.a2
        self.a1 /= s
        self.a2 /= s

    def _interval_for_Cav(self, C_total: float) -> Optional[Tuple[float, float]]:
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

    def allocate(self, C_total: float, speed_rpm: float, Cav_prev: Optional[float] = None) -> Dict[str, Any]:
        interval = self._interval_for_Cav(C_total)
        Cmax = self.Cmax_per_wheel

        def clip_c(x: float) -> float:
            lo = -Cmax if self.allow_regen else 0.0
            return float(np.clip(x, lo, Cmax))

        # Fallback saturation si impossible
        if interval is None:
            Cav = clip_c(C_total / 4.0)
            Car = clip_c(C_total / 4.0)
            eta_f = float(self.cosphi_map(Cav, speed_rpm))
            eta_r = float(self.cosphi_map(Car, speed_rpm))
            score = self.a1 * eta_f + self.a2 * eta_r
            return {
                "Cav": Cav, "Car": Car,
                "C_front_total": 2.0 * Cav, "C_rear_total": 2.0 * Car,
                "eta_front": eta_f, "eta_rear": eta_r,
                "score": score, "status": "SATURATED"
            }

        lo, hi = interval

        def objective(Cav: float) -> float:
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

        return {
            "Cav": Cav_opt, "Car": Car_opt,
            "C_front_total": 2.0 * Cav_opt, "C_rear_total": 2.0 * Car_opt,
            "eta_front": eta_f, "eta_rear": eta_r,
            "score": score, "status": "OK"
        }


# =============================================================================
# Consignes test (couple_total(t), vitesse_rpm(t))
# =============================================================================
def generate_test_reference(
    duration_s: float = 60.0,
    dt: float = 0.1,
    torque_peak_total: float = 360.0,
    speed_min: Optional[float] = None,
    speed_max: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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
# Simulation : optimisation vs baseline uniforme
# =============================================================================
def simulate(
    allocator: TorqueAllocator,
    t: np.ndarray,
    speed_rpm: np.ndarray,
    C_total: np.ndarray,
) -> Dict[str, np.ndarray]:
    n = len(t)
    Cav = np.zeros(n)
    Car = np.zeros(n)
    eta_f = np.zeros(n)
    eta_r = np.zeros(n)
    score = np.zeros(n)
    status = np.empty(n, dtype=object)

    Cav_prev = None
    for k in range(n):
        out = allocator.allocate(float(C_total[k]), float(speed_rpm[k]), Cav_prev=Cav_prev)
        Cav[k] = out["Cav"]
        Car[k] = out["Car"]
        eta_f[k] = out["eta_front"]
        eta_r[k] = out["eta_rear"]
        score[k] = out["score"]
        status[k] = out["status"]
        Cav_prev = Cav[k]

    C_front_total = 2.0 * Cav
    C_rear_total = 2.0 * Car
    C_rec = C_front_total + C_rear_total
    return {
        "Cav": Cav, "Car": Car,
        "C_front_total": C_front_total,
        "C_rear_total": C_rear_total,
        "eta_f": eta_f, "eta_r": eta_r,
        "score": score,
        "status": status,
        "C_rec": C_rec,
    }


def baseline_uniform(mapping: Callable[[float, float], np.ndarray],
                     C_total: np.ndarray,
                     speed_rpm: np.ndarray) -> Dict[str, np.ndarray]:
    # uniforme: 2*Cav = 2*Car = C_total/2  => Cav=Car=C_total/4
    Cav = C_total / 4.0
    Car = C_total / 4.0
    eta_f = mapping(Cav, speed_rpm).astype(float)
    eta_r = mapping(Car, speed_rpm).astype(float)
    score = 0.7 * eta_f + 0.3 * eta_r  # uniquement pour comparaison visuelle
    return {
        "Cav": Cav, "Car": Car,
        "C_front_total": 2.0 * Cav,
        "C_rear_total": 2.0 * Car,
        "eta_f": eta_f, "eta_r": eta_r,
        "score": score,
    }


# =============================================================================
# Plots
# =============================================================================
def plot_results(t: np.ndarray, speed_rpm: np.ndarray, C_total: np.ndarray,
                 opt: Dict[str, np.ndarray], uni: Dict[str, np.ndarray]) -> None:
    # Couples (optim vs uniforme)
    fig1, ax1 = plt.subplots(figsize=(11, 5), constrained_layout=True)
    ax1.plot(t, C_total, label="C_total demandé (Nm)")
    ax1.plot(t, opt["C_front_total"], label="Avant total (opti)")
    ax1.plot(t, opt["C_rear_total"], label="Arrière total (opti)")
    ax1.plot(t, uni["C_front_total"], "--", label="Avant total (uniforme)")
    ax1.plot(t, uni["C_rear_total"], "--", label="Arrière total (uniforme)")
    ax1.set_title("Allocation de couple (demandé vs réparti)")
    ax1.set_xlabel("Temps (s)")
    ax1.set_ylabel("Couple (Nm)")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Vitesse
    fig2, ax2 = plt.subplots(figsize=(11, 5), constrained_layout=True)
    ax2.plot(t, speed_rpm, label="Vitesse (rpm)")
    ax2.set_title("Consigne de vitesse")
    ax2.set_xlabel("Temps (s)")
    ax2.set_ylabel("rpm")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # cos(phi)
    fig3, ax3 = plt.subplots(figsize=(11, 5), constrained_layout=True)
    ax3.plot(t, opt["eta_f"], label="cos(phi) AV (opti)")
    ax3.plot(t, opt["eta_r"], label="cos(phi) AR (opti)")
    ax3.plot(t, uni["eta_f"], "--", label="cos(phi) AV (uniforme)")
    ax3.plot(t, uni["eta_r"], "--", label="cos(phi) AR (uniforme)")
    ax3.set_title("cos(phi) AV/AR : optimisation vs uniforme")
    ax3.set_xlabel("Temps (s)")
    ax3.set_ylabel("[-]")
    ax3.grid(True, alpha=0.3)
    ax3.legend()

    plt.show()


# =============================================================================
# MAIN
# =============================================================================
def main() -> None:
    # 1) Mapping
    mapping, metrics = build_mapping_two_sides(T_data, S_data, Z_data, clip_01=True)
    print("Fit metrics:", metrics)

    # 2) Allocateur
    Cmax_per_wheel = float(np.max(np.abs(T_data)))  # ~117 Nm
    allocator = TorqueAllocator(
        cosphi_map=mapping,
        Cmax_per_wheel=Cmax_per_wheel,
        a1=0.7, a2=0.3,
        allow_regen=True,
        smooth_lambda=0.0,
    )

    # 3) Consignes
    t, speed_rpm, C_total = generate_test_reference(duration_s=60.0, dt=0.1, torque_peak_total=360.0)

    # 4) Simulations
    opt = simulate(allocator, t, speed_rpm, C_total)
    uni = baseline_uniform(mapping, C_total, speed_rpm)

    # 5) Vérif contrainte
    max_err = float(np.max(np.abs(opt["C_rec"] - C_total)))
    nb_sat = int(np.sum(opt["status"] == "SATURATED"))
    print("\n" + "=" * 88)
    print("TEST OPTIMISATION — Résumé")
    print("=" * 88)
    print(f"Cmax_per_wheel = {Cmax_per_wheel:.3f} Nm | allow_regen = {allocator.allow_regen}")
    print(f"Poids: a1={allocator.a1:.3f}, a2={allocator.a2:.3f}")
    print(f"Erreur max sur 2*Cav+2*Car=C_total : {max_err:.6e} Nm")
    print(f"Nb saturations : {nb_sat} / {len(t)}")
    print("=" * 88 + "\n")

    # 6) Plots
    plot_results(t, speed_rpm, C_total, opt, uni)


if __name__ == "__main__":
    main()
