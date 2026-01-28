# src/vehicle_sim/control/allocation.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import os
import numpy as np

# SciPy optionnel
try:
    from scipy.optimize import minimize_scalar  # type: ignore
    _HAVE_SCIPY = True
except Exception:
    _HAVE_SCIPY = False

# Pour accéder proprement aux fichiers data/ dans un package
try:
    from importlib.resources import files as ir_files  # py>=3.9
except Exception:
    ir_files = None


# =============================================================================
# 1) Map cosphi via LS (2 fits)
# =============================================================================

@dataclass
class CosPhiMapLS:
    coeff_pos: np.ndarray  # (6,)
    coeff_neg: np.ndarray  # (6,)
    T_min: float
    T_max: float
    rpm_min: float
    rpm_max: float
    clamp_01: bool = True

    def __call__(self, T: float, rpm: float) -> float:
        T = float(T)
        rpm = float(abs(rpm))

        # hors map => 0
        if (T < self.T_min) or (T > self.T_max) or (rpm < self.rpm_min) or (rpm > self.rpm_max):
            return 0.0

        c = self.coeff_pos if T >= 0.0 else self.coeff_neg
        a, b, cTr, d, e, f0 = [float(x) for x in c]
        val = a*T*T + b*rpm*rpm + cTr*T*rpm + d*T + e*rpm + f0

        if self.clamp_01:
            val = float(np.clip(val, 0.0, 1.0))
        return val


def _A(T: np.ndarray, rpm: np.ndarray) -> np.ndarray:
    return np.column_stack([T**2, rpm**2, T*rpm, T, rpm, np.ones_like(T)])


def _fit_two_sides(T: np.ndarray, rpm: np.ndarray, cosphi: np.ndarray) -> CosPhiMapLS:
    T = np.asarray(T, dtype=float).ravel()
    R = np.asarray(rpm, dtype=float).ravel()
    Z = np.asarray(cosphi, dtype=float).ravel()

    if not (len(T) == len(R) == len(Z)):
        raise ValueError("Données invalides: T, rpm, cosphi doivent avoir la même longueur.")

    Rabs = np.abs(R)

    coeff_pos, *_ = np.linalg.lstsq(_A(T, Rabs), Z, rcond=None)
    coeff_neg, *_ = np.linalg.lstsq(_A(-T, Rabs), Z, rcond=None)

    return CosPhiMapLS(
        coeff_pos=np.asarray(coeff_pos, dtype=float).reshape(6,),
        coeff_neg=np.asarray(coeff_neg, dtype=float).reshape(6,),
        T_min=float(np.min(T)),
        T_max=float(np.max(T)),
        rpm_min=float(np.min(Rabs)),
        rpm_max=float(np.max(Rabs)),
        clamp_01=True,
    )


# =============================================================================
# 2) Chargement dataset + cache (fit une seule fois)
# =============================================================================

# Tu peux override le chemin via variable d'env si tu veux:
#   VEHICLE_SIM_COSPHI_CSV=/chemin/absolu/mon_fichier.csv
ENV_CSV = "VEHICLE_SIM_COSPHI_CSV"

# Nom par défaut du fichier dans data/
DEFAULT_CSV_NAME = "efficiency_map_clean.csv"

# Indices colonnes CSV : [torque, rpm, cosphi]
TORQUE_COL = 0
RPM_COL = 1
COSPHI_COL = 2
CSV_DELIM = ","
CSV_SKIP_HEADER = 1

_COSPHI_MAP: Optional[CosPhiMapLS] = None


def _resolve_default_csv_path() -> str:
    # 1) override env
    env = os.environ.get(ENV_CSV, "").strip()
    if env:
        return env

    # 2) chemin via importlib.resources (propre en package)
    # suppose que ton fichier est dans: src/vehicle_sim/data/efficiency_map_clean.csv
    if ir_files is not None:
        try:
            p = ir_files("vehicle_sim").joinpath("data").joinpath(DEFAULT_CSV_NAME)
            return str(p)
        except Exception:
            pass

    # 3) fallback relatif (si tu lances depuis racine)
    return os.path.join("data", DEFAULT_CSV_NAME)


def _load_cloud_csv(path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Nuage cosphi introuvable: {path}\n"
            f"- Mets {DEFAULT_CSV_NAME} dans src/vehicle_sim/data/\n"
            f"- ou définis {ENV_CSV} vers ton CSV."
        )

    data = np.genfromtxt(path, delimiter=CSV_DELIM, skip_header=CSV_SKIP_HEADER)
    if data.ndim == 1 or data.shape[1] <= max(TORQUE_COL, RPM_COL, COSPHI_COL):
        raise ValueError(
            f"CSV invalide: {path}\n"
            f"Attendu au moins 3 colonnes: torque,rpm,cosphi."
        )

    T = data[:, TORQUE_COL]
    R = data[:, RPM_COL]
    Z = data[:, COSPHI_COL]
    return T, R, Z


def get_cosphi_map() -> CosPhiMapLS:
    """
    Singleton: map LS construite une seule fois.
    """
    global _COSPHI_MAP
    if _COSPHI_MAP is not None:
        return _COSPHI_MAP

    csv_path = _resolve_default_csv_path()
    T, R, Z = _load_cloud_csv(csv_path)
    _COSPHI_MAP = _fit_two_sides(T, R, Z)
    return _COSPHI_MAP


# =============================================================================
# 3) Allocateur (4 stratégies)
# =============================================================================

@dataclass
class _SmoothState:
    prev_Cav: Optional[float] = None
    prev2_Cav: Optional[float] = None


class TorqueAllocator:
    """
    Contrainte:
      2*Cav + 2*Car = T_total   (Cav/Car = couple par moteur/roue)
      Car = T_total/2 - Cav
    """

    def __init__(
        self,
        a1: float = 0.7,
        a2: float = 0.3,
        allow_regen: bool = True,
        Cmax_per_motor: float = 120.0,
        lambda1: float = 5e-3,
        lambda2: float = 5e-4,
        dC_max: float = 5.0,
    ):
        self.map = get_cosphi_map()

        s = float(a1 + a2)
        if s <= 0:
            raise ValueError("a1+a2 doit être > 0.")
        self.a1 = float(a1 / s)
        self.a2 = float(a2 / s)

        self.allow_regen = bool(allow_regen)
        self.Cmax = float(Cmax_per_motor)

        self.lambda1 = float(lambda1)
        self.lambda2 = float(lambda2)
        self.dC_max = float(dC_max)

        self._st = _SmoothState()

    def _interval_Cav(self, T_total: float) -> Optional[Tuple[float, float]]:
        half = 0.5 * float(T_total)
        if self.allow_regen:
            lo = max(-self.Cmax, half - self.Cmax)
            hi = min(+self.Cmax, half + self.Cmax)
        else:
            lo = max(0.0, half - self.Cmax)
            hi = min(self.Cmax, half)
        if lo > hi:
            return None
        return lo, hi

    def _score(self, Cav: float, Car: float, rpm: float) -> float:
        return self.a1 * self.map(Cav, rpm) + self.a2 * self.map(Car, rpm)

    def _obj(self, Cav: float, T_total: float, rpm: float, smooth: bool) -> float:
        Car = 0.5 * float(T_total) - float(Cav)
        score = self._score(Cav, Car, rpm)

        if smooth:
            if self._st.prev_Cav is not None:
                score -= self.lambda1 * (Cav - self._st.prev_Cav) ** 2
            if self._st.prev_Cav is not None and self._st.prev2_Cav is not None:
                dd = Cav - 2.0 * self._st.prev_Cav + self._st.prev2_Cav
                score -= self.lambda2 * (dd ** 2)

        return -score

    def _solve_1d(self, lo: float, hi: float, fun) -> float:
        if _HAVE_SCIPY:
            res = minimize_scalar(fun, bounds=(lo, hi), method="bounded")
            return float(res.x)
        xs = np.linspace(lo, hi, 61)
        vals = [fun(float(x)) for x in xs]
        return float(xs[int(np.argmin(vals))])

    def _out(self, Cav: float, Car: float, rpm: float, method: str, status: str) -> Dict:
        Cav = float(Cav)
        Car = float(Car)

        Tf = 2.0 * Cav
        Tr = 2.0 * Car
        T_total = Tf + Tr

        cos_f = float(self.map(Cav, rpm))
        cos_r = float(self.map(Car, rpm))

        eps = 1e-9
        ratio_front = float(np.clip(Tf / (T_total + eps), 0.0, 1.0))

        return {
            "method": method,
            "status": status,
            "rpm": float(rpm),

            "Cav": Cav,
            "Car": Car,
            "T_front": Tf,
            "T_rear": Tr,
            "T_wheels": np.array([Cav, Cav, Car, Car], dtype=float),

            "cosphi_front": cos_f,
            "cosphi_rear": cos_r,
            "ratio_front": ratio_front,
            "score": float(self.a1 * cos_f + self.a2 * cos_r),
        }

    def _update_state(self, Cav: float) -> None:
        self._st.prev2_Cav = self._st.prev_Cav
        self._st.prev_Cav = float(Cav)

    def _inverse(self, T_total: float, rpm: float) -> Dict:
        Cav = 0.25 * float(T_total)
        Cav = float(np.clip(Cav, -self.Cmax if self.allow_regen else 0.0, self.Cmax))
        Car = Cav
        self._update_state(Cav)
        return self._out(Cav, Car, rpm, "inverse", "OK")

    def _piecewise(self, T_total: float, rpm: float) -> Dict:
        itv = self._interval_Cav(T_total)
        if itv is None:
            return self._inverse(T_total, rpm)
        lo, hi = itv
        Cav = self._solve_1d(lo, hi, lambda x: self._obj(x, T_total, rpm, smooth=False))
        Car = 0.5 * float(T_total) - Cav
        self._update_state(Cav)
        return self._out(Cav, Car, rpm, "piecewise", "OK")

    def _smooth(self, T_total: float, rpm: float) -> Dict:
        itv0 = self._interval_Cav(T_total)
        if itv0 is None:
            return self._inverse(T_total, rpm)

        lo0, hi0 = itv0
        lo, hi = lo0, hi0

        if self._st.prev_Cav is not None and self.dC_max > 0.0:
            lo = max(lo, self._st.prev_Cav - self.dC_max)
            hi = min(hi, self._st.prev_Cav + self.dC_max)
            if lo > hi:
                Cav = float(np.clip(self._st.prev_Cav, lo0, hi0))
                Car = 0.5 * float(T_total) - Cav
                self._update_state(Cav)
                return self._out(Cav, Car, rpm, "smooth", "RATE_LIMITED")

        Cav = self._solve_1d(lo, hi, lambda x: self._obj(x, T_total, rpm, smooth=True))
        Car = 0.5 * float(T_total) - Cav
        self._update_state(Cav)
        return self._out(Cav, Car, rpm, "smooth", "OK")

    def _quadratic(self, T_total: float, rpm: float) -> Dict:
        itv = self._interval_Cav(T_total)
        if itv is None:
            return self._inverse(T_total, rpm)
        lo, hi = itv

        Cav0 = self._st.prev_Cav
        if Cav0 is None:
            Cav0 = float(np.clip(0.25 * float(T_total), lo, hi))

        def J(Cav: float) -> float:
            Car = 0.5 * float(T_total) - Cav
            return self._score(Cav, Car, rpm)

        h = max(1e-3, 0.01 * self.Cmax)
        x1 = float(np.clip(Cav0 - h, lo, hi))
        x2 = float(np.clip(Cav0, lo, hi))
        x3 = float(np.clip(Cav0 + h, lo, hi))

        f1, f2, f3 = J(x1), J(x2), J(x3)
        g1 = (f3 - f1) / (2.0 * h) if abs(x3 - x1) > 1e-12 else 0.0
        g2 = (f3 - 2.0 * f2 + f1) / (h * h) if abs(h) > 1e-12 else 0.0

        if abs(g2) < 1e-9:
            return self._piecewise(T_total, rpm)

        Cav = float(np.clip(Cav0 - g1 / g2, lo, hi))
        Car = 0.5 * float(T_total) - Cav
        self._update_state(Cav)
        return self._out(Cav, Car, rpm, "quadratic", "OK")

    def optimize(self, strategy: str, T_total: float, rpm: float) -> Dict:
        rpm = float(rpm)
        if abs(rpm) < 1.0:
            rpm = 1.0

        s = (strategy or "").strip().lower()
        if s in ("", "inverse", "pinv", "pseudo_inverse", "pseudoinverse"):
            return self._inverse(T_total, rpm)
        if "piece" in s:
            return self._piecewise(T_total, rpm)
        if "smooth" in s:
            return self._smooth(T_total, rpm)
        if "quad" in s:
            return self._quadratic(T_total, rpm)
        return self._inverse(T_total, rpm)
