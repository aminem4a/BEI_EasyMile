# -*- coding: utf-8 -*-
"""
BEI EasyMile — Simulation en boucle ouverte (comparaison de stratégies d'allocation)

Objectif
--------
- Lire une consigne de couple d'essieu depuis `nominal_driving_8kmh_unloaded.csv`
- Convertir en couple véhicule: C_vehicle = 2 * C_axle (hypothèse validée: consigne par essieu)
- Appliquer 3 stratégies d'allocation AV/AR (2 roues par essieu)
- Appliquer un modèle véhicule longitudinal (Class_vehiculeoff.Vehicule)
- OPTION 1: pas de marche arrière (saturation des consignes négatives à l'arrêt, vitesse bornée >= 0)

Important
---------
- cos(phi) est obtenu UNIQUEMENT par interpolation des points (Torque, Speed_rpm, cosphi)
  fournis au début de `allocation_couple.py` (aucun fit polynomiale / surface LS).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
from scipy.optimize import minimize_scalar

# ============================================================================
# FICHIERS DU PROJET
# ============================================================================
THIS_DIR = Path(__file__).resolve().parent

CSV_CONSIGNE = THIS_DIR / "nominal_driving_8kmh_unloaded.csv"
ALLOC_DATA_PY = THIS_DIR / "allocation_couple.py"
VEHICLE_PY = THIS_DIR / "Class_vehiculeoff.py"

# ============================================================================
# PARAMÈTRES SIMULATION
# ============================================================================
TARGET_DT_S = 0.5            # resampling du CSV (s) — à ajuster si besoin
SLOPE_RAD = 0.0              # pente (rad)
V_MAX_MPS = 15.0 / 3.6       # 15 km/h (fiche véhicule)
V_STOP_MPS = 0.01            # seuil "à l'arrêt" pour Option 1

# Conversion vitesse véhicule -> vitesse moteur (rpm)
GEAR_RATIO = 26.0            # mention "Transmission 26:1" dans Class_vehiculeoff.py
EPS = 1e-12

# Pondération objectif (favoriser l'essieu avant)
W_FRONT = 0.65
W_REAR = 0.35

# Stratégie 1 (répartition fixe)
FRONT_SHARE_FIXED = 0.65

# Stratégie 3 (lissage)
DX_MAX = 0.10                # variation max du ratio AV par pas de temps
LAMBDA_SMOOTH = 0.40         # pénalité de variation (plus grand = plus lisse)


# ============================================================================
# OUTILS — IMPORT DYNAMIQUE (évite d'imposer le projet comme package)
# ============================================================================
def load_module_from_path(path: Path, name: str):
    import importlib.util
    import sys

    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Impossible de charger le module: {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# ============================================================================
# INTERPOLATION cos(phi) À PARTIR DES POINTS FOURNIS
# ============================================================================
class CosphiInterpolator:
    """Interpolation (Torque, Speed_rpm) -> cosphi, avec fallback nearest.

    - Utilise les points fournis dans allocation_couple.py:
      TORQUE_DATA, SPEED_DATA, COSPHI_DATA
    - Pour T < 0: cosphi(|T|, rpm) (miroir)
    - Clip cosphi dans [0, 1]
    """

    def __init__(self, torque_nm: np.ndarray, speed_rpm: np.ndarray, cosphi: np.ndarray):
        torque_nm = np.asarray(torque_nm, dtype=float).ravel()
        speed_rpm = np.asarray(speed_rpm, dtype=float).ravel()
        cosphi = np.asarray(cosphi, dtype=float).ravel()

        if not (len(torque_nm) == len(speed_rpm) == len(cosphi)):
            raise ValueError("Les tableaux TORQUE/SPEED/COSPHI doivent avoir la même taille.")

        pts = np.column_stack([torque_nm, speed_rpm])

        # Interpolation linéaire sur nuage + fallback nearest hors enveloppe convexe
        self._lin = LinearNDInterpolator(pts, cosphi, fill_value=np.nan)
        self._near = NearestNDInterpolator(pts, cosphi)

        # Bornes utiles (debug / clipping doux)
        self.t_min, self.t_max = float(np.min(torque_nm)), float(np.max(torque_nm))
        self.s_min, self.s_max = float(np.min(speed_rpm)), float(np.max(speed_rpm))

    def __call__(self, torque_nm: float, speed_rpm: float) -> float:
        if not np.isfinite(torque_nm) or not np.isfinite(speed_rpm):
            return 0.0

        t = abs(float(torque_nm))  # miroir pour freinage
        s = float(speed_rpm)

        z = self._lin(t, s)
        if np.isnan(z):
            z = self._near(t, s)

        z = float(z)
        if not np.isfinite(z):
            z = 0.0

        # Saturation physique
        return float(np.clip(z, 0.0, 1.0))


def vehicle_speed_to_motor_rpm(v_mps: float, wheel_radius_m: float) -> float:
    """v (m/s) -> rpm moteur (hypothèse: ratio fixe + pas de glissement)."""
    omega_w = v_mps / max(wheel_radius_m, EPS)          # rad/s
    omega_m = omega_w * GEAR_RATIO
    rpm = omega_m * 60.0 / (2.0 * math.pi)
    return float(max(rpm, 0.0))


# ============================================================================
# STRATÉGIES D'ALLOCATION
# ============================================================================
@dataclass
class AllocationOutput:
    # Couples par roue (Nm)
    c_front_wheel: float
    c_rear_wheel: float

    # cos(phi) associés
    cosphi_front: float
    cosphi_rear: float

    # ratio AV (sur couple véhicule) en valeur absolue
    front_share: float


class StrategyBase:
    name: str = "base"

    def reset(self):
        pass

    def allocate(self, c_vehicle: float, rpm: float, cosphi_map: CosphiInterpolator) -> AllocationOutput:
        raise NotImplementedError


class StrategyFixedSplit(StrategyBase):
    name = "S1 — Fixed split"

    def __init__(self, front_share: float = FRONT_SHARE_FIXED):
        self.front_share = float(np.clip(front_share, 0.0, 1.0))

    def allocate(self, c_vehicle: float, rpm: float, cosphi_map: CosphiInterpolator) -> AllocationOutput:
        x = self.front_share
        c_front_total = x * c_vehicle
        c_rear_total = (1.0 - x) * c_vehicle

        c_fw = 0.5 * c_front_total
        c_rw = 0.5 * c_rear_total

        zf = cosphi_map(c_fw, rpm)
        zr = cosphi_map(c_rw, rpm)

        return AllocationOutput(c_fw, c_rw, zf, zr, x)


class StrategyUniformSplit(StrategyBase):
    """Répartition uniforme AV/AR, sans optimisation (baseline)."""

    name = "S0 — Uniform (sans opti)"

    def allocate(self, c_vehicle: float, rpm: float, cosphi_map: CosphiInterpolator) -> AllocationOutput:
        x = 0.5
        c_front_total = x * c_vehicle
        c_rear_total = (1.0 - x) * c_vehicle

        c_fw = 0.5 * c_front_total
        c_rw = 0.5 * c_rear_total

        zf = cosphi_map(c_fw, rpm)
        zr = cosphi_map(c_rw, rpm)

        return AllocationOutput(c_fw, c_rw, zf, zr, x)


class StrategyInstantOpt(StrategyBase):
    name = "S2 — Instant opt"

    def __init__(self, w_front: float = W_FRONT, w_rear: float = W_REAR):
        self.w_front = float(w_front)
        self.w_rear = float(w_rear)

    def allocate(self, c_vehicle: float, rpm: float, cosphi_map: CosphiInterpolator) -> AllocationOutput:
        # variable: x = fraction envoyée à l'avant
        def objective(x: float) -> float:
            x = float(np.clip(x, 0.0, 1.0))
            c_fw = 0.5 * (x * c_vehicle)
            c_rw = 0.5 * ((1.0 - x) * c_vehicle)
            zf = cosphi_map(c_fw, rpm)
            zr = cosphi_map(c_rw, rpm)
            # on MINIMISE => négatif du score
            return -(self.w_front * zf + self.w_rear * zr)

        res = minimize_scalar(objective, bounds=(0.0, 1.0), method="bounded", options={"xatol": 1e-3})
        x = float(np.clip(res.x if res.success else 0.5, 0.0, 1.0))

        c_fw = 0.5 * (x * c_vehicle)
        c_rw = 0.5 * ((1.0 - x) * c_vehicle)
        zf = cosphi_map(c_fw, rpm)
        zr = cosphi_map(c_rw, rpm)
        return AllocationOutput(c_fw, c_rw, zf, zr, x)


class StrategySmoothOpt(StrategyBase):
    name = "S3 — Smooth opt"

    def __init__(self, dx_max: float = DX_MAX, lam: float = LAMBDA_SMOOTH,
                 w_front: float = W_FRONT, w_rear: float = W_REAR):
        self.dx_max = float(abs(dx_max))
        self.lam = float(max(lam, 0.0))
        self.w_front = float(w_front)
        self.w_rear = float(w_rear)
        self._x_prev = FRONT_SHARE_FIXED

    def reset(self):
        self._x_prev = FRONT_SHARE_FIXED

    def allocate(self, c_vehicle: float, rpm: float, cosphi_map: CosphiInterpolator) -> AllocationOutput:
        x0 = float(np.clip(self._x_prev, 0.0, 1.0))
        lo = float(np.clip(x0 - self.dx_max, 0.0, 1.0))
        hi = float(np.clip(x0 + self.dx_max, 0.0, 1.0))

        def objective(x: float) -> float:
            x = float(np.clip(x, 0.0, 1.0))
            c_fw = 0.5 * (x * c_vehicle)
            c_rw = 0.5 * ((1.0 - x) * c_vehicle)
            zf = cosphi_map(c_fw, rpm)
            zr = cosphi_map(c_rw, rpm)
            score = (self.w_front * zf + self.w_rear * zr)
            smooth_penalty = self.lam * (x - x0) ** 2
            return -(score) + smooth_penalty

        # Si lo==hi (cas limite) on ne cherche pas
        if abs(hi - lo) < 1e-9:
            x = lo
        else:
            res = minimize_scalar(objective, bounds=(lo, hi), method="bounded", options={"xatol": 1e-3})
            x = float(np.clip(res.x if res.success else x0, 0.0, 1.0))

        self._x_prev = x

        c_fw = 0.5 * (x * c_vehicle)
        c_rw = 0.5 * ((1.0 - x) * c_vehicle)
        zf = cosphi_map(c_fw, rpm)
        zr = cosphi_map(c_rw, rpm)
        return AllocationOutput(c_fw, c_rw, zf, zr, x)


# ============================================================================
# LECTURE + RESAMPLING DU CSV DE CONSIGNE
# ============================================================================
def load_and_resample_csv(path: Path, dt_target_s: float) -> pd.DataFrame:
    """Charge le CSV et resample à dt_target_s via interpolation linéaire.

    Colonnes attendues:
      - Time(ms)
      - Axle_Torque_Setpoint(N.m)
      - Speed_Feedback(m/s) (optionnel, utilisé pour comparaison)
    """
    df = pd.read_csv(path, sep=None, engine="python")
    required = ["Time(ms)", "Axle_Torque_Setpoint(N.m)"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Colonne manquante dans {path.name}: {col}")

    t = df["Time(ms)"].to_numpy(dtype=float) * 1e-3
    c_axle = df["Axle_Torque_Setpoint(N.m)"].to_numpy(dtype=float)

    # Optionnel
    v_fb = df["Speed_Feedback(m/s)"].to_numpy(dtype=float) if "Speed_Feedback(m/s)" in df.columns else None

    # Grille cible uniforme
    t0, t1 = float(t[0]), float(t[-1])
    n = int(math.floor((t1 - t0) / dt_target_s)) + 1
    t_u = t0 + dt_target_s * np.arange(n)

    c_axle_u = np.interp(t_u, t, c_axle)
    out = {
        "t_s": t_u,
        "c_axle_set_nm": c_axle_u,
        "c_vehicle_set_nm": 2.0 * c_axle_u,   # Hypothèse A: consigne par essieu
    }
    if v_fb is not None:
        out["v_feedback_mps"] = np.interp(t_u, t, v_fb)

    return pd.DataFrame(out)


# ============================================================================
# SIMULATION OPEN-LOOP
# ============================================================================
@dataclass
class SimLog:
    t_s: np.ndarray
    c_vehicle_set: np.ndarray
    v_mps: np.ndarray
    a_mps2: np.ndarray

    c_front_total: np.ndarray
    c_rear_total: np.ndarray
    cosphi_front: np.ndarray
    cosphi_rear: np.ndarray
    front_share: np.ndarray

    eta_weighted: np.ndarray


def simulate_open_loop(df_cmd: pd.DataFrame,
                       vehicle,
                       strategy: StrategyBase,
                       cosphi_map: CosphiInterpolator) -> SimLog:
    """Simule une stratégie sur toute la consigne resamplée."""

    t = df_cmd["t_s"].to_numpy(dtype=float)
    dt = float(np.median(np.diff(t)))

    n = len(t)
    v = np.zeros(n)
    a = np.zeros(n)

    c_set = df_cmd["c_vehicle_set_nm"].to_numpy(dtype=float)

    c_front_total = np.zeros(n)
    c_rear_total = np.zeros(n)
    zf = np.zeros(n)
    zr = np.zeros(n)
    x = np.zeros(n)
    eta = np.zeros(n)

    strategy.reset()

    for k in range(n):
        # OPTION 1: pas de marche arrière (cycle nominal)
        c_req = float(c_set[k])
        if vehicle.v <= V_STOP_MPS and c_req < 0.0:
            c_req = 0.0

        # rpm "équivalent" (hypothèse ratio fixe)
        rpm = vehicle_speed_to_motor_rpm(vehicle.v, vehicle.r)

        out = strategy.allocate(c_req, rpm, cosphi_map)

        # Couples par essieu (2 roues)
        cF = 2.0 * out.c_front_wheel
        cR = 2.0 * out.c_rear_wheel

        # (Option) on peut clipper si besoin — ici on conserve tel quel
        torques_wheels = [out.c_front_wheel, out.c_front_wheel, out.c_rear_wheel, out.c_rear_wheel]

        # Modèle véhicule (longitudinal)
        vk, ak, _ = vehicle.update(torques_wheels, dt, slope_rad=SLOPE_RAD)

        # OPTION 1: vitesse non négative + limite vmax
        vehicle.v = float(np.clip(vehicle.v, 0.0, V_MAX_MPS))
        vk = vehicle.v

        # Log
        v[k] = vk
        a[k] = ak

        c_front_total[k] = cF
        c_rear_total[k] = cR
        zf[k] = out.cosphi_front
        zr[k] = out.cosphi_rear
        x[k] = out.front_share
        eta[k] = (W_FRONT * out.cosphi_front + W_REAR * out.cosphi_rear)

    return SimLog(
        t_s=t,
        c_vehicle_set=c_set,
        v_mps=v,
        a_mps2=a,
        c_front_total=c_front_total,
        c_rear_total=c_rear_total,
        cosphi_front=zf,
        cosphi_rear=zr,
        front_share=x,
        eta_weighted=eta,
    )


# ============================================================================
# PLOTS (axes + unités + légendes propres)
# ============================================================================
def plot_summary(logs: Dict[str, SimLog], out_png: Path):
    fig = plt.figure(figsize=(12, 10))

    # 1) Consigne couple
    ax1 = fig.add_subplot(4, 1, 1)
    for name, lg in logs.items():
        ax1.plot(lg.t_s, lg.c_vehicle_set, label=name)
    ax1.set_title("Couple véhicule demandé")
    ax1.set_xlabel("Temps (s)")
    ax1.set_ylabel("Couple véhicule (N·m)")
    ax1.grid(True)
    ax1.legend()

    # 2) Vitesse
    ax2 = fig.add_subplot(4, 1, 2)
    for name, lg in logs.items():
        ax2.plot(lg.t_s, lg.v_mps, label=name)
    ax2.set_title("Vitesse véhicule (boucle ouverte)")
    ax2.set_xlabel("Temps (s)")
    ax2.set_ylabel("Vitesse (m/s)")
    ax2.grid(True)
    ax2.legend()

    # 3) Couples essieux
    ax3 = fig.add_subplot(4, 1, 3)
    for name, lg in logs.items():
        ax3.plot(lg.t_s, lg.c_front_total, label=f"{name} — Essieu AV")
        ax3.plot(lg.t_s, lg.c_rear_total, linestyle="--", label=f"{name} — Essieu AR")
    ax3.set_title("Répartition du couple (par essieu)")
    ax3.set_xlabel("Temps (s)")
    ax3.set_ylabel("Couple essieu (N·m)")
    ax3.grid(True)
    ax3.legend(ncols=2)

    # 4) Score (rendement pondéré)
    ax4 = fig.add_subplot(4, 1, 4)
    for name, lg in logs.items():
        ax4.plot(lg.t_s, lg.eta_weighted, label=name)
    ax4.set_title("Score de rendement pondéré (W_FRONT*cosφ_AV + W_REAR*cosφ_AR)")
    ax4.set_xlabel("Temps (s)")
    ax4.set_ylabel("Score (—)")
    ax4.grid(True)
    ax4.legend()

    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def plot_details(logs: Dict[str, SimLog], out_png: Path):
    fig = plt.figure(figsize=(12, 10))

    # 1) Couples par roue
    ax1 = fig.add_subplot(3, 1, 1)
    for name, lg in logs.items():
        ax1.plot(lg.t_s, 0.5 * lg.c_front_total, label=f"{name} — Cav (par roue)")
        ax1.plot(lg.t_s, 0.5 * lg.c_rear_total, linestyle="--", label=f"{name} — Car (par roue)")
    ax1.set_title("Couples par roue")
    ax1.set_xlabel("Temps (s)")
    ax1.set_ylabel("Couple roue (N·m)")
    ax1.grid(True)
    ax1.legend(ncols=2)

    # 2) cos(phi)
    ax2 = fig.add_subplot(3, 1, 2)
    for name, lg in logs.items():
        ax2.plot(lg.t_s, lg.cosphi_front, label=f"{name} — cosφ AV")
        ax2.plot(lg.t_s, lg.cosphi_rear, linestyle="--", label=f"{name} — cosφ AR")
    ax2.set_title("cos(φ) interpolé (cartographie)")
    ax2.set_xlabel("Temps (s)")
    ax2.set_ylabel("cos(φ) (—)")
    ax2.grid(True)
    ax2.legend(ncols=2)

    # 3) Part AV
    ax3 = fig.add_subplot(3, 1, 3)
    for name, lg in logs.items():
        ax3.plot(lg.t_s, lg.front_share, label=name)
    ax3.set_title("Part du couple envoyée à l'avant")
    ax3.set_xlabel("Temps (s)")
    ax3.set_ylabel("Part AV (—)")
    ax3.grid(True)
    ax3.legend()

    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def plot_cosphi_uniform_vs_opti(log_uniform: SimLog, log_opti: SimLog,
                                label_uniform: str, label_opti: str,
                                out_front_png: Path, out_rear_png: Path):
    """Nouveaux plots demandés:
    - cos(phi) AV uniquement (uniforme vs opti)
    - cos(phi) AR uniquement (uniforme vs opti)
    """

    # --- Front ---
    fig = plt.figure(figsize=(10, 4.5))
    plt.plot(log_uniform.t_s, log_uniform.cosphi_front, label=label_uniform)
    plt.plot(log_opti.t_s, log_opti.cosphi_front, label=label_opti)
    plt.title("cos(φ) — Essieu avant (comparaison)")
    plt.xlabel("Temps (s)")
    plt.ylabel("cos(φ) avant (—)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_front_png, dpi=200)
    plt.close(fig)

    # --- Rear ---
    fig = plt.figure(figsize=(10, 4.5))
    plt.plot(log_uniform.t_s, log_uniform.cosphi_rear, label=label_uniform)
    plt.plot(log_opti.t_s, log_opti.cosphi_rear, label=label_opti)
    plt.title("cos(φ) — Essieu arrière (comparaison)")
    plt.xlabel("Temps (s)")
    plt.ylabel("cos(φ) arrière (—)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_rear_png, dpi=200)
    plt.close(fig)


# ============================================================================
# MAIN
# ============================================================================
def main():
    # 1) Charger les données cosphi depuis allocation_couple.py
    alloc_mod = load_module_from_path(ALLOC_DATA_PY, "allocation_couple_data")
    torque_data = np.asarray(alloc_mod.TORQUE_DATA, dtype=float)
    speed_data = np.asarray(alloc_mod.SPEED_DATA, dtype=float)
    cosphi_data = np.asarray(alloc_mod.COSPHI_DATA, dtype=float)

    cosphi_map = CosphiInterpolator(torque_data, speed_data, cosphi_data)

    # 2) Charger + resampler la consigne CSV
    df_cmd = load_and_resample_csv(CSV_CONSIGNE, TARGET_DT_S)

    # 3) Instancier modèle véhicule
    veh_mod = load_module_from_path(VEHICLE_PY, "vehicle_model")
    # On crée une instance par stratégie pour comparaison "fair" (même état initial)
    def new_vehicle():
        return veh_mod.Vehicule()

    # 4) Stratégies
    strategies = [
        StrategyFixedSplit(front_share=FRONT_SHARE_FIXED),
        StrategyInstantOpt(w_front=W_FRONT, w_rear=W_REAR),
        StrategySmoothOpt(dx_max=DX_MAX, lam=LAMBDA_SMOOTH, w_front=W_FRONT, w_rear=W_REAR),
    ]

    logs: Dict[str, SimLog] = {}
    for strat in strategies:
        veh = new_vehicle()
        logs[strat.name] = simulate_open_loop(df_cmd, veh, strat, cosphi_map)

    # 5) Figures
    out1 = THIS_DIR / "comparatif_strategies_openloop_interp_clean.png"
    out2 = THIS_DIR / "comparatif_strategies_openloop_detail_interp_clean.png"
    plot_summary(logs, out1)
    plot_details(logs, out2)


    # 6) Nouveaux plots demandés: cos(phi) séparé AV et AR, sans opti vs avec opti
    #    - Sans opti: répartition uniforme 50/50
    #    - Avec opti: stratégie lissée (SmoothOpt)
    uniform = StrategyUniformSplit()
    opti = StrategySmoothOpt(dx_max=DX_MAX, lam=LAMBDA_SMOOTH, w_front=W_FRONT, w_rear=W_REAR)

    log_uniform = simulate_open_loop(df_cmd, new_vehicle(), uniform, cosphi_map)
    log_opti = simulate_open_loop(df_cmd, new_vehicle(), opti, cosphi_map)

    out_front = THIS_DIR / "cosphi_avant_sans_vs_avec_opti.png"
    out_rear = THIS_DIR / "cosphi_arriere_sans_vs_avec_opti.png"
    plot_cosphi_uniform_vs_opti(log_uniform, log_opti,
                                label_uniform=uniform.name,
                                label_opti=opti.name,
                                out_front_png=out_front,
                                out_rear_png=out_rear)

    print("OK — Figures générées :")
    print(f"- {out1}")
    print(f"- {out2}")
    print(f"- {out_front}")
    print(f"- {out_rear}")


if __name__ == "__main__":
    main()
