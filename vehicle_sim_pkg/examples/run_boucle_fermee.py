# -*- coding: utf-8 -*-
"""
SIMULATION BOUCLE FERMÉE — PROFIL SYNTHÉTIQUE
---------------------------------------------
Ce script :
1. Génère un profil de vitesse complet (Accélération -> Palier -> Sinusoïde -> Arrêt).
2. Utilise le Pilote Robuste (Feed-Forward + PID) et la Physique Corrigée (tau=0.05s).
3. Affiche les performances et l'efficacité avec des graphiques clairs.
"""

from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from scipy.optimize import minimize_scalar
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# QP optionnel
try:
    import cvxpy as cp
    _HAS_CVXPY = True
except Exception:
    _HAS_CVXPY = False

# =============================================================================
# 0) DONNÉES MAP (INTEGRÉES)
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
# 1) OUTILS & MAP
# =============================================================================
def build_A_full(T, S):
    T, S = np.asarray(T).ravel(), np.asarray(S).ravel()
    return np.column_stack([T**2, S**2, T*S, T, S, np.ones_like(T)])

def fit_full(T, S, Z):
    A = build_A_full(T, S)
    c, _, _, _ = np.linalg.lstsq(A, Z, rcond=None)
    return c

def predict_full(coeffs, T, S):
    A = build_A_full(T, S)
    return (A @ coeffs).reshape(np.asarray(T).shape)

def build_mapping_two_sides(clip_01=False):
    c_pos = fit_full(T_data, S_data, Z_data)
    c_neg = fit_full(-T_data, S_data, Z_data)
    def cosphi_map(T, S):
        T = np.asarray(T); S = np.asarray(S)
        val = np.where(T >= 0, predict_full(c_pos, T, S), predict_full(c_neg, T, S))
        return np.clip(val, 0.1, 1.0)
    return cosphi_map, c_pos, c_neg

def d_cosphi_dT(coeffs, T, S):
    a, _, c, d, _, _ = coeffs
    return 2.0 * a * T + c * S + d

# =============================================================================
# 2) GÉNÉRATION CONSIGNE (SYNTHÉTIQUE)
# =============================================================================
def generate_speed_reference(duration_s=60.0, dt=0.01):
    t = np.arange(0.0, duration_s + 1e-12, dt)
    # Vitesse max cible = 15 km/h = 4.16 m/s
    v_max_kmh = 15.0
    v_max = v_max_kmh / 3.6
    
    speed = np.zeros_like(t)
    for i, ti in enumerate(t):
        if ti < 10: 
            # Accélération linéaire
            speed[i] = v_max * (ti / 10.0)
        elif ti < 25: 
            # Palier constant
            speed[i] = v_max
        elif ti < 40: 
            # Variation sinusoïdale autour d'une valeur moyenne
            # On descend un peu et on oscille
            base = v_max * 0.8
            amp = v_max * 0.2
            speed[i] = base + amp * np.sin(2.0 * np.pi * (ti - 25.0) / 7.5)
        elif ti < 50:
            # Freinage
            v_start_brake = speed[i-1] if i > 0 else v_max
            speed[i] = v_start_brake * (1 - (ti - 40.0)/10.0)
        else:
            # Arrêt
            speed[i] = 0.0
            
    return t, speed

# =============================================================================
# 3) PHYSIQUE & PILOTE (Robustes)
# =============================================================================
@dataclass
class VehicleConfig:
    mass: float = 4900.0
    wheel_radius: float = 0.24
    ratio_reduction: float = 26.0
    surface_frontale: float = 3.0
    cx: float = 0.7
    crr: float = 0.015
    friction_force: float = 80.0
    motor_max_torque: float = 200.0
    tau1: float = 0.05  # Moteur réactif (50ms)
    r: float = 1.0

class Engine:
    def __init__(self, max_torque, tau1, r):
        self.max_torque = max_torque
        self.tau1, self.r = tau1, r
        self.x1, self.x2 = 0.0, 0.0
    def step(self, cmd, dt):
        u = np.clip(cmd, -self.max_torque, self.max_torque)
        dx1 = self.x2
        dx2 = (u - (2.0*self.r*self.tau1)*self.x2 - self.x1) / (self.tau1**2)
        self.x1 += dx1 * dt
        self.x2 += dx2 * dt
        return self.x1

class Vehicle:
    def __init__(self, config, dt=0.01):
        self.config = config; self.dt = dt
        self.engines = [Engine(config.motor_max_torque, config.tau1, config.r) for _ in range(4)]
        self.velocity = 0.0; self.position = 0.0
    def step(self, cmds, dt=None):
        dt = dt if dt else self.dt
        act_torques = np.array([e.step(c, dt) for e, c in zip(self.engines, cmds)])
        drive_F = (np.sum(act_torques)*self.config.ratio_reduction)/self.config.wheel_radius
        v = self.velocity
        sign = np.sign(v) if abs(v)>0.01 else (np.sign(drive_F) if abs(drive_F)>10 else 0)
        res_F = (0.5*1.225*self.config.surface_frontale*self.config.cx*(v**2) + 
                 self.config.mass*9.81*self.config.crr + self.config.friction_force)*sign
        if abs(v)<0.01 and abs(drive_F)<=abs(res_F): self.velocity = 0.0
        else: self.velocity += ((drive_F - res_F)/self.config.mass)*dt
        return self.velocity, act_torques

class Driver:
    def __init__(self, cfg, kp=800.0, ki=5.0, dt=0.01):
        self.cfg, self.kp, self.ki, self.dt = cfg, kp, ki, dt
        self.integral, self.max_tot = 0.0, 4*cfg.motor_max_torque
    def compute_cmd(self, v_ref, v_meas):
        # Feed-Forward
        sign = np.sign(v_ref)
        res_F = (0.5*1.225*self.cfg.surface_frontale*self.cfg.cx*(v_ref**2) + 
                 self.cfg.mass*9.81*self.cfg.crr + self.cfg.friction_force)*sign
        C_ff = (res_F * self.cfg.wheel_radius) / self.cfg.ratio_reduction
        # PID
        err = v_ref - v_meas
        self.integral += err * self.dt
        lim = self.max_tot/self.ki if self.ki>0 else 0
        self.integral = np.clip(self.integral, -lim, lim)
        return np.clip(C_ff + self.kp*err + self.ki*self.integral, -self.max_tot, self.max_tot)

# =============================================================================
# 4) ALLOCATEURS
# =============================================================================
@dataclass
class AllocatorInverse:
    Cmax: float
    def allocate(self, C, rpm, **k):
        per = np.clip(C/4, -self.Cmax, self.Cmax)
        return {"Cav": per, "Car": per}

@dataclass
class AllocatorPiecewise:
    map: callable; Cmax: float
    def allocate(self, C, rpm, **k):
        lo = max(-self.Cmax, 0.5*C - self.Cmax); hi = min(self.Cmax, 0.5*C + self.Cmax)
        if lo > hi: return {"Cav": C/4, "Car": C/4}
        obj = lambda c: -(0.7*self.map(c, rpm) + 0.3*self.map(0.5*C-c, rpm))
        res = minimize_scalar(obj, bounds=(lo, hi), method="bounded")
        return {"Cav": res.x, "Car": 0.5*C - res.x}

@dataclass
class AllocatorSmooth:
    map: callable; Cmax: float
    def allocate(self, C, rpm, prev=None, **k):
        lo = max(-self.Cmax, 0.5*C - self.Cmax); hi = min(self.Cmax, 0.5*C + self.Cmax)
        if lo > hi: return {"Cav": C/4, "Car": C/4}
        tgt = prev if prev is not None else C/4
        obj = lambda c: -(0.7*self.map(c, rpm) + 0.3*self.map(0.5*C-c, rpm)) + 1e-5*(c-tgt)**2
        res = minimize_scalar(obj, bounds=(lo, hi), method="bounded")
        return {"Cav": res.x, "Car": 0.5*C - res.x}

@dataclass
class AllocatorQuadratic:
    c_pos: any; c_neg: any; Cmax: float
    def allocate(self, C, rpm, prev=None, **k):
        if not _HAS_CVXPY: return {"Cav": C/4, "Car": C/4}
        lo = max(-self.Cmax, 0.5*C - self.Cmax); hi = min(self.Cmax, 0.5*C + self.Cmax)
        if lo > hi: return {"Cav": C/4, "Car": C/4}
        Cav, Car = cp.Variable(), cp.Variable()
        prev_val = prev if prev is not None else C/4
        obj = cp.Minimize((Cav - prev_val)**2 + 1e-3*(Cav**2 + Car**2))
        const = [2*Cav + 2*Car == C, Cav >= lo, Cav <= hi, Car >= -self.Cmax, Car <= self.Cmax]
        try:
            cp.Problem(obj, const).solve(solver=cp.OSQP, verbose=False)
            return {"Cav": Cav.value, "Car": Car.value}
        except: return {"Cav": C/4, "Car": C/4}

# =============================================================================
# 5) MAIN & PLOTS
# =============================================================================
def main():
    dt_sim = 0.01
    # Génération du scénario synthétique
    t_sim, v_ref_ms = generate_speed_reference(duration_s=60.0, dt=dt_sim)
    
    map_func, cp, cn = build_mapping_two_sides(clip_01=True)
    Cmax = 200.0
    cfg = VehicleConfig()
    
    allocs = {
        "Inverse": AllocatorInverse(Cmax),
        "Piecewise": AllocatorPiecewise(map_func, Cmax),
        "Smooth": AllocatorSmooth(map_func, Cmax),
        "Quadratic": AllocatorQuadratic(cp, cn, Cmax)
    }
    
    results = {}

    print("Début de simulation (Profil Synthétique)...")
    for name, strat in allocs.items():
        veh = Vehicle(cfg, dt_sim)
        drv = Driver(cfg, kp=800, ki=5, dt=dt_sim)
        
        v_hist, c_cmd_hist, c_real_hist = [], [], []
        cav_hist, car_hist = [], []
        cos_f, cos_r = [], []
        prev = 0
        
        for k in range(len(t_sim)):
            cmd_tot = drv.compute_cmd(v_ref_ms[k], veh.velocity)
            
            rpm = (veh.velocity / cfg.wheel_radius) * cfg.ratio_reduction * (60/2/np.pi)
            rpm_safe = max(abs(rpm), 1.0) 
            res = strat.allocate(cmd_tot, rpm_safe, prev=prev)
            cav, car = float(res["Cav"]), float(res["Car"])
            prev = cav
            
            v, real_torques = veh.step([cav, cav, car, car])
            
            v_hist.append(v)
            c_cmd_hist.append(cmd_tot)
            c_real_hist.append(np.sum(real_torques))
            cav_hist.append(cav); car_hist.append(car)
            cos_f.append(map_func(cav, rpm_safe))
            cos_r.append(map_func(car, rpm_safe))
            
        results[name] = {
            "v": np.array(v_hist), "c_cmd": np.array(c_cmd_hist), "c_real": np.array(c_real_hist),
            "cav": np.array(cav_hist), "car": np.array(car_hist),
            "cf": np.array(cos_f), "cr": np.array(cos_r)
        }

    # --- PLOTS AVEC AXES CLAIRS ---
    fig, axs = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)
    
    # 1. VITESSE
    ax = axs[0,0]
    ax.plot(t_sim, v_ref_ms*3.6, 'k--', lw=2, label="Consigne")
    for n, r in results.items():
        rmse = np.sqrt(np.mean((v_ref_ms - r["v"])**2)) * 3.6
        ax.plot(t_sim, r["v"]*3.6, label=f"{n} (RMSE={rmse:.2f} km/h)")
    ax.set_title("Suivi de Vitesse (Consigne vs Réel)", fontsize=12, fontweight='bold')
    ax.set_xlabel("Temps [s]", fontsize=10)
    ax.set_ylabel("Vitesse [km/h]", fontsize=10)
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, alpha=0.4)
    
    # 2. COUPLE TOTAL
    ax = axs[0,1]
    for n, r in results.items():
        if n == "Inverse": # Un seul tracé pour ne pas surcharger
            ax.plot(t_sim, r["c_cmd"], 'r--', label="PID Commande")
            ax.plot(t_sim, r["c_real"], 'b', label="Réel Moteurs", alpha=0.6)
    ax.set_title("Couple Total Véhicule (Commande vs Réel)", fontsize=12, fontweight='bold')
    ax.set_xlabel("Temps [s]", fontsize=10)
    ax.set_ylabel("Couple Total [Nm]", fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.4)
    
    # 3. RÉPARTITION COUPLE
    ax = axs[1,0]
    r = results["Smooth"] # Zoom sur une méthode intéressante
    ax.plot(t_sim, 2*r["cav"], label="Essieu Avant (2x Cav)", color='#1f77b4')
    ax.plot(t_sim, 2*r["car"], label="Essieu Arrière (2x Car)", color='#ff7f0e', linestyle='--')
    ax.set_title("Répartition Couple par Essieu (Méthode Smooth)", fontsize=12, fontweight='bold')
    ax.set_xlabel("Temps [s]", fontsize=10)
    ax.set_ylabel("Couple Essieu [Nm]", fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.4)
    
    # 4. HISTOGRAMME COSPHI
    ax = axs[1,1]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    for i, (n, r) in enumerate(results.items()):
        vals = np.concatenate([r["cf"], r["cr"]])
        # Filtre : on ne garde que les points où les moteurs forcent (>5Nm)
        couple_actifs = np.abs(np.concatenate([r["cav"], r["car"]])) > 5.0
        valid_vals = vals[couple_actifs]
        
        if len(valid_vals) > 0:
            ax.hist(valid_vals, bins=20, alpha=0.4, label=n, density=True, color=colors[i])
            
    ax.set_title("Distribution Efficacité (Cos $\phi$) en charge", fontsize=12, fontweight='bold')
    ax.set_xlabel("Valeur Cos $\phi$ (sans unité)", fontsize=10)
    ax.set_ylabel("Densité de probabilité", fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.4)
    
    plt.show()

if __name__ == "__main__":
    main()