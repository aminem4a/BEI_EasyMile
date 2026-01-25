from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.optimize import minimize_scalar


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
