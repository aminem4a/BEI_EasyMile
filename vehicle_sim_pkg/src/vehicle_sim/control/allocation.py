import numpy as np
from scipy.optimize import minimize_scalar
from ..utils.data_loader import DataLoader

class TorqueAllocator:
    def __init__(self, Cmax=None):
        self.cosphi_map = DataLoader.build_efficiency_map(clip_01=True)
        if Cmax is None:
            self.Cmax = getattr(self.cosphi_map, 'Cmax_data', 117.0)
        else:
            self.Cmax = Cmax

    def _eval(self, c, rpm):
        """Retourne un pur float (pas un numpy array)"""
        val = self.cosphi_map(c, rpm)
        return float(val)

    def solve_allocation(self, strategy_name, C_total, rpm, **kwargs):
        if strategy_name == "Inverse":
            return self._solve_alloc2(C_total, rpm)
        elif strategy_name == "Smooth":
            return self._solve_alloc3(C_total, rpm, kwargs.get("prev_cav"), kwargs.get("prev_cav2"))
        elif strategy_name == "Quadratic":
            return self._solve_quadratic(C_total, rpm)
        elif strategy_name == "Piecewise":
            return self._solve_piecewise(C_total, rpm)
        else:
            return self._solve_alloc2(C_total, rpm)

    def _solve_alloc2(self, C_total, rpm):
        a1, a2 = 0.7, 0.3
        half = 0.5 * C_total
        lo = max(-self.Cmax, half - self.Cmax)
        hi = min(+self.Cmax, half + self.Cmax)
        
        if lo > hi:
            val = self.Cmax if C_total > 0 else -self.Cmax
            return val, val, self._eval(val, rpm), self._eval(val, rpm), "SAT"
        
        def obj(Cav):
            Car = 0.5 * C_total - Cav
            return -(a1*self.cosphi_map(Cav, rpm) + a2*self.cosphi_map(Car, rpm))

        res = minimize_scalar(obj, bounds=(lo, hi), method="bounded")
        Cav = float(res.x)
        Car = 0.5*C_total - Cav
        return Cav, Car, self._eval(Cav, rpm), self._eval(Car, rpm), "OK"

    def _solve_alloc3(self, C_total, rpm, prev_cav, prev_cav2):
        a1, a2 = 0.7, 0.3
        lam1, lam2 = 5e-3, 5e-4
        half = 0.5 * C_total
        lo = max(-self.Cmax, half - self.Cmax)
        hi = min(+self.Cmax, half + self.Cmax)

        if lo > hi:
            val = self.Cmax if C_total > 0 else -self.Cmax
            return val, val, self._eval(val, rpm), self._eval(val, rpm), "SAT"

        def obj(Cav):
            Car = 0.5 * C_total - Cav
            score = a1*self.cosphi_map(Cav, rpm) + a2*self.cosphi_map(Car, rpm)
            if prev_cav is not None:
                score -= lam1 * (Cav - prev_cav)**2
            if prev_cav is not None and prev_cav2 is not None:
                score -= lam2 * (Cav - 2*prev_cav + prev_cav2)**2
            return -score

        res = minimize_scalar(obj, bounds=(lo, hi), method="bounded")
        Cav = float(res.x)
        Car = 0.5*C_total - Cav
        return Cav, Car, self._eval(Cav, rpm), self._eval(Car, rpm), "OK"

    def _solve_quadratic(self, C_total, rpm):
        c_50 = C_total / 4.0
        score_50 = 0.7*self.cosphi_map(c_50, rpm) + 0.3*self.cosphi_map(c_50, rpm)
        
        if (C_total/2.0) <= self.Cmax:
            c_100 = C_total / 2.0
            score_100 = 0.7*self.cosphi_map(c_100, rpm) + 0.3*self.cosphi_map(0, rpm)
        else:
            score_100 = -999.0
            c_100 = c_50
            
        if score_100 > score_50: Cav = c_100
        else: Cav = c_50
            
        Cav = np.clip(Cav, -self.Cmax, self.Cmax)
        Car = np.clip(0.5*C_total - Cav, -self.Cmax, self.Cmax)
        return Cav, Car, self._eval(Cav, rpm), self._eval(Car, rpm), "OK"

    def _solve_piecewise(self, C_total, rpm):
        ratios = np.linspace(0, 1, 31)
        best_s = -999.0
        best_cav = np.clip(C_total/4.0, -self.Cmax, self.Cmax)
        for r in ratios:
            cav_try = (C_total * r) / 2.0
            car_try = (C_total * (1-r)) / 2.0
            if abs(cav_try) <= self.Cmax and abs(car_try) <= self.Cmax:
                s = 0.7*self.cosphi_map(cav_try, rpm) + 0.3*self.cosphi_map(car_try, rpm)
                if s > best_s:
                    best_s = s
                    best_cav = cav_try
        Car = 0.5*C_total - best_cav
        return best_cav, Car, self._eval(best_cav, rpm), self._eval(Car, rpm), "OK"