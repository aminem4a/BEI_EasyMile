# -*- coding: utf-8 -*-
import numpy as np
from typing import List
from scipy.optimize import minimize_scalar

# Import relatif propre
from ..utils import DataLoader 

try:
    import cvxpy as cp
    CVXPY_AVAILABLE = True
except ImportError:
    CVXPY_AVAILABLE = False

class TorqueAllocator:
    def __init__(self, mode: str = "smooth", data_loader: DataLoader = None):
        self.mode = mode
        self.loader = data_loader
        
        # Paramètres
        self.alpha = 0.005
        self.max_rate = 300.0
        self.last_tf = 0.0
        self.prev_torques = np.zeros(4)
        
        self.T_nom = 180.0 
        self.lambda_P = 1.0
        self.lambda_S = 0.6

    def _get_phys_limits(self, rpm):
        if self.loader: return self.loader.get_max_torque(rpm)
        return self.T_nom

    def _estimate_cosphi(self, T, omega_rad):
        if self.loader:
            rpm = abs(omega_rad) * 9.5493
            return self.loader.get_cosphi(T, rpm)
        return max(0.1, min(0.98, 0.2 + 0.75 * min(1.0, abs(T)/180.0 * 1.5)))

    def allocate(self, total_cmd: float, rpm: float, dt: float) -> List[float]:
        t_max = self._get_phys_limits(rpm)
        omega = rpm * 0.1047
        half_total = total_cmd / 2.0

        # 1. INVERSE
        if self.mode == "inverse":
            split = np.clip(total_cmd / 4.0, -t_max, t_max)
            return [split] * 4

        # 2. PIECEWISE
        if self.mode == "piecewise":
            low, high = 0, half_total
            low, high = min(low, high), max(low, high)

            def cost_pw(tf):
                tr = half_total - tf
                return -(self._estimate_cosphi(tf, omega) + self._estimate_cosphi(tr, omega))

            res = minimize_scalar(cost_pw, bounds=(low, high), method='bounded')
            tf = res.x
            self.last_tf = tf
            return [tf/2, tf/2, (half_total-tf)/2, (half_total-tf)/2]

        # 3. SMOOTH
        if self.mode == "smooth":
            d_max = self.max_rate * dt
            low = max(-t_max, self.last_tf - d_max)
            high = min(t_max, self.last_tf + d_max)
            
            # Intersection avec bornes absolues (simplifié positif)
            if total_cmd >= 0:
                low = max(low, 0)
                high = min(high, half_total)
            
            low, high = min(low, high), max(low, high)

            def cost_sm(tf):
                tr = half_total - tf
                eff = -(self._estimate_cosphi(tf, omega) + self._estimate_cosphi(tr, omega))
                return eff + self.alpha * (tf - self.last_tf)**2

            res = minimize_scalar(cost_sm, bounds=(low, high), method='bounded')
            tf = res.x
            self.last_tf = tf
            return [tf/2, tf/2, (half_total-tf)/2, (half_total-tf)/2]

        # 4. QUADRATIC
        if self.mode == "quadratic":
            if CVXPY_AVAILABLE: return self._solve_qp(total_cmd, omega)
            return [total_cmd/4.0]*4

        return [total_cmd/4.0]*4

    def _solve_qp(self, T_global, omega) -> List[float]:
        delta = 5.0
        H_diag, f_lin = [], []
        
        def local_obj(t):
            cp_val = self._estimate_cosphi(t, omega)
            eff = max(0.1, cp_val * 1.1) if cp_val < 0.85 else cp_val
            P = (t * omega) / eff if eff > 0 else 0
            S = P / cp_val if cp_val > 0.05 else 0
            return self.lambda_P * P + self.lambda_S * S

        for i in range(4):
            t0 = self.prev_torques[i]
            y0 = local_obj(t0)
            hess = max(1e-4, (local_obj(t0+delta) - 2*y0 + local_obj(t0-delta)) / delta**2)
            grad = (local_obj(t0+delta) - local_obj(t0-delta)) / (2*delta)
            H_diag.append(hess)
            f_lin.append(grad - hess * t0)

        x = cp.Variable(4)
        prob = cp.Problem(cp.Minimize(0.5 * cp.quad_form(x, np.diag(H_diag)) + np.array(f_lin).T @ x),
                          [cp.sum(x) == T_global, x >= -self.T_nom, x <= self.T_nom])
        try:
            prob.solve(solver=cp.OSQP, verbose=False)
            res = x.value if prob.status in ["optimal", "optimal_inaccurate"] else [T_global/4]*4
        except: res = [T_global/4]*4
        self.prev_torques = res
        return list(res)# -*- coding: utf-8 -*-
import numpy as np
from typing import List
from scipy.optimize import minimize_scalar

# Import relatif propre
from ..utils import DataLoader 

try:
    import cvxpy as cp
    CVXPY_AVAILABLE = True
except ImportError:
    CVXPY_AVAILABLE = False

class TorqueAllocator:
    def __init__(self, mode: str = "smooth", data_loader: DataLoader = None):
        self.mode = mode
        self.loader = data_loader
        
        # Paramètres
        self.alpha = 0.005
        self.max_rate = 300.0
        self.last_tf = 0.0
        self.prev_torques = np.zeros(4)
        
        self.T_nom = 180.0 
        self.lambda_P = 1.0
        self.lambda_S = 0.6

    def _get_phys_limits(self, rpm):
        if self.loader: return self.loader.get_max_torque(rpm)
        return self.T_nom

    def _estimate_cosphi(self, T, omega_rad):
        """Debug intégré pour voir pourquoi c'est constant."""
        rpm = abs(omega_rad) * 9.5493
        t_abs = abs(T)
        
        if self.loader:
            val = self.loader.get_cosphi(T, rpm)
            
            # --- DEBUG TEMPORAIRE (Affiche 1 fois sur 1000 pour ne pas spammer) ---
            # Activez ceci si vous voulez voir les valeurs dans la console
            if np.random.rand() < 0.001: 
                print(f"DEBUG MAP -> Demande: {t_abs:.1f}Nm @ {rpm:.0f}rpm  >>> Sortie CosPhi: {val:.3f}")
            # ----------------------------------------------------------------------
            
            return val
        
        # Fallback
        return max(0.1, min(0.98, 0.2 + 0.75 * min(1.0, abs(T)/180.0 * 1.5)))

    def allocate(self, total_cmd: float, rpm: float, dt: float) -> List[float]:
        t_max = self._get_phys_limits(rpm)
        omega = rpm * 0.1047
        half_total = total_cmd / 2.0

        # 1. INVERSE
        if self.mode == "inverse":
            split = np.clip(total_cmd / 4.0, -t_max, t_max)
            return [split] * 4

        # 2. PIECEWISE
        if self.mode == "piecewise":
            low, high = 0, half_total
            low, high = min(low, high), max(low, high)

            def cost_pw(tf):
                tr = half_total - tf
                return -(self._estimate_cosphi(tf, omega) + self._estimate_cosphi(tr, omega))

            res = minimize_scalar(cost_pw, bounds=(low, high), method='bounded')
            tf = res.x
            self.last_tf = tf
            return [tf/2, tf/2, (half_total-tf)/2, (half_total-tf)/2]

        # 3. SMOOTH
        if self.mode == "smooth":
            d_max = self.max_rate * dt
            low = max(-t_max, self.last_tf - d_max)
            high = min(t_max, self.last_tf + d_max)
            
            # Intersection avec bornes absolues (simplifié positif)
            if total_cmd >= 0:
                low = max(low, 0)
                high = min(high, half_total)
            
            low, high = min(low, high), max(low, high)

            def cost_sm(tf):
                tr = half_total - tf
                eff = -(self._estimate_cosphi(tf, omega) + self._estimate_cosphi(tr, omega))
                return eff + self.alpha * (tf - self.last_tf)**2

            res = minimize_scalar(cost_sm, bounds=(low, high), method='bounded')
            tf = res.x
            self.last_tf = tf
            return [tf/2, tf/2, (half_total-tf)/2, (half_total-tf)/2]

        # 4. QUADRATIC
        if self.mode == "quadratic":
            if CVXPY_AVAILABLE: return self._solve_qp(total_cmd, omega)
            return [total_cmd/4.0]*4

        return [total_cmd/4.0]*4

    def _solve_qp(self, T_global, omega) -> List[float]:
        delta = 5.0
        H_diag, f_lin = [], []
        
        def local_obj(t):
            cp_val = self._estimate_cosphi(t, omega)
            eff = max(0.1, cp_val * 1.1) if cp_val < 0.85 else cp_val
            P = (t * omega) / eff if eff > 0 else 0
            S = P / cp_val if cp_val > 0.05 else 0
            return self.lambda_P * P + self.lambda_S * S

        for i in range(4):
            t0 = self.prev_torques[i]
            y0 = local_obj(t0)
            hess = max(1e-4, (local_obj(t0+delta) - 2*y0 + local_obj(t0-delta)) / delta**2)
            grad = (local_obj(t0+delta) - local_obj(t0-delta)) / (2*delta)
            H_diag.append(hess)
            f_lin.append(grad - hess * t0)

        x = cp.Variable(4)
        prob = cp.Problem(cp.Minimize(0.5 * cp.quad_form(x, np.diag(H_diag)) + np.array(f_lin).T @ x),
                          [cp.sum(x) == T_global, x >= -self.T_nom, x <= self.T_nom])
        try:
            prob.solve(solver=cp.OSQP, verbose=False)
            res = x.value if prob.status in ["optimal", "optimal_inaccurate"] else [T_global/4]*4
        except: res = [T_global/4]*4
        self.prev_torques = res
        return list(res)