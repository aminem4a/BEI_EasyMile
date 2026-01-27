# -*- coding: utf-8 -*-
import numpy as np
from typing import List
from scipy.optimize import minimize_scalar
import cvxpy as cp 
from ..utils import DataLoader 

class TorqueAllocator:
    def __init__(self, mode: str = "smooth", data_loader: DataLoader = None):
        self.mode = mode
        self.loader = data_loader
        
        # Paramètres
        self.lambda_smooth = 0.005 
        self.max_rate = 300.0
        self.last_tf = 0.0
        self.prev_torques = np.zeros(4)
        
        self.T_nom = 180.0       # Limite par moteur
        self.T_GLOBAL_MAX = 500.0 # Limite GLOBALE (Sécurité Projet)

    def _estimate_cosphi(self, T, omega_rad):
        if self.loader:
            rpm = abs(omega_rad) * 9.5493
            if np.isnan(T) or np.isnan(rpm): return 0.5
            return self.loader.get_cosphi(T, rpm)
        return 0.5

    def allocate(self, total_cmd: float, rpm: float, dt: float) -> List[float]:
        # --- SÉCURITÉ : CLAMPING GLOBAL 500 Nm ---
        # On force la commande à rester dans [-500, +500] quoi qu'il arrive
        total_cmd = np.clip(total_cmd, -self.T_GLOBAL_MAX, self.T_GLOBAL_MAX)
        
        omega = rpm * 0.1047
        
        # --- 1. INVERSE ---
        if self.mode == "inverse":
            split = total_cmd / 4.0
            return [split] * 4

        # --- 2. PIECEWISE / SMOOTH ---
        if self.mode in ["piecewise", "smooth"]:
            low = min(0, total_cmd)
            high = max(0, total_cmd)
            
            # Inversion des bornes si freinage
            if total_cmd < 0:
                low, high = high, low 

            # Lissage temporel
            if self.mode == "smooth":
                d_max = self.max_rate * dt * 2.0
                low = max(low, self.last_tf - d_max)
                high = min(high, self.last_tf + d_max)
                if low > high: low, high = high, low

            # Optimisation
            if abs(total_cmd) < 1e-3:
                self.last_tf = 0.0
                return [0.0] * 4

            def cost_func(t_front_axle):
                t_rear_axle = total_cmd - t_front_axle
                eff_f = self._estimate_cosphi(t_front_axle/2.0, omega)
                eff_r = self._estimate_cosphi(t_rear_axle/2.0, omega)
                J = -(eff_f + eff_r)
                
                if self.mode == "smooth":
                    J += self.lambda_smooth * (t_front_axle - self.last_tf)**2
                return J

            try:
                res = minimize_scalar(cost_func, bounds=(low, high), method='bounded')
                if res.success:
                    tf_axle = res.x
                    self.last_tf = tf_axle
                    tr_axle = total_cmd - tf_axle
                    return [tf_axle/2, tf_axle/2, tr_axle/2, tr_axle/2]
            except:
                pass

        # --- 3. QUADRATIC ---
        if self.mode == "quadratic":
            return self._solve_qp_robust(total_cmd, omega)

        # Fallback
        return [total_cmd/4.0] * 4

    def _solve_qp_robust(self, T_global, omega) -> List[float]:
        try:
            effs = []
            for t_prev in self.prev_torques:
                e = max(0.01, self._estimate_cosphi(t_prev, omega))
                effs.append(e)

            x = cp.Variable(4)
            # Minimisation des pertes Joules pondérées
            Q_weights = [1.0/e for e in effs]
            obj = cp.Minimize(cp.sum([Q_weights[i] * x[i]**2 for i in range(4)]))
            
            constraints = [
                cp.sum(x) == T_global,
                x >= -self.T_nom,
                x <= self.T_nom
            ]
            
            prob = cp.Problem(obj, constraints)
            prob.solve(solver=cp.OSQP, verbose=False)

            if prob.status in ["optimal", "optimal_inaccurate"]:
                res = x.value
                self.prev_torques = res
                return list(res)
        except:
            pass
        return [T_global/4.0]*4

    def _get_phys_limits(self, rpm):
        if self.loader: return self.loader.get_max_torque(rpm)
        return self.T_nom