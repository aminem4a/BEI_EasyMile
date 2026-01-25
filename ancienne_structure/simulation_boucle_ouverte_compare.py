# -*- coding: utf-8 -*-

"""Simulation en boucle ouverte : comparaison de 3 stratégies d'allocation.

Ce script :
- utilise le modèle véhicule calibré (Class_vehiculeoff.Vehicule)
- modélise chaque moteur par un 2nd ordre (Motor2ndOrder)
- compare 3 stratégies d'allocation (issues des fichiers fournis)
- génère des réponses temporelles avec axes/légendes correctement renseignés

Sorties :
- comparatif_strategies_openloop.png
- comparatif_strategies_openloop_detail.png
"""

from __future__ import annotations

import math
import os
import sys
from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt

# Assure l'import depuis le dossier du script
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)

import Class_vehiculeoff as vehmod
import allocation_couple as alloc_ls
import allocation_2 as alloc_pw
import allocation_3 as alloc_smooth


# =============================================================================
# Paramètres communs simulation
# =============================================================================
GEAR_RATIO = 26.0                 # Transmission : moteur -> roue
RPM_PER_RAD_S = 60.0 / (2.0 * math.pi)  # 9.549...

# Limites véhicule (sujet EasyMile) : Vmax = 15 km/h (=> 4.167 m/s)
V_MAX_MPS = 15.0 / 3.6


# =============================================================================
# Modèle moteur 2nd ordre (simple)
# =============================================================================
class Motor2ndOrder:
    """Modèle discret d'un 2nd ordre sur le couple moteur.

    Équivalent à : x¨ + 2*zeta*wn*x˙ + wn²*x = wn²*u
    Ici, on choisit wn≈45 rad/s, zeta≈0.8 (valeurs proches du script initial).
    """

    def __init__(self, wn: float = 45.0, zeta: float = 0.8):
        self.wn = float(wn)
        self.zeta = float(zeta)
        # état : [couple, couple_dot]
        self.state = np.array([0.0, 0.0], dtype=float)

    def step(self, u: float, dt: float) -> float:
        x, xd = float(self.state[0]), float(self.state[1])
        wn = self.wn
        zeta = self.zeta
        # xdd = wn^2*(u - x) - 2*zeta*wn*xd
        xdd = (wn * wn) * (u - x) - (2.0 * zeta * wn) * xd
        x_new = x + xd * dt
        xd_new = xd + xdd * dt
        self.state[:] = (x_new, xd_new)
        return float(x_new)


# =============================================================================
# Profils de consigne
# =============================================================================

def torque_profile_total_motor(t: np.ndarray) -> np.ndarray:
    """Consigne C_total(t) (somme des 4 couples moteurs).

    Profil volontairement "riche" : rampe + palier + échelon + relâchement.
    On reste majoritairement en couple positif pour comparer proprement avec
    la stratégie LS (allocation_couple) qui ne gère pas le regen.
    """
    C = np.zeros_like(t, dtype=float)
    for k, tk in enumerate(t):
        if tk < 4.0:
            C[k] = 200.0 * (tk / 4.0)          # 0 -> 200
        elif tk < 7.0:
            C[k] = 200.0                        # palier
        elif tk < 10.0:
            C[k] = 520.0                        # gros échelon
        elif tk < 13.0:
            C[k] = 140.0                        # relâchement
        else:
            # petite ondulation (toujours positive)
            C[k] = 140.0 + 40.0 * math.sin(2.0 * math.pi * (tk - 13.0) / 2.5)
            C[k] = max(0.0, C[k])
    return C


# =============================================================================
# Wrappers stratégie (harmonisation interfaces)
# =============================================================================

@dataclass
class StrategyResult:
    name: str
    t: np.ndarray
    C_total_req: np.ndarray
    v_mps: np.ndarray
    a_mps2: np.ndarray
    x_m: np.ndarray
    Cav_motor: np.ndarray
    Car_motor: np.ndarray
    score: np.ndarray
    eta_front: np.ndarray
    eta_rear: np.ndarray


class StrategyBase:
    name: str

    @property
    def C_total_motor_limit(self) -> float:
        """Limite physique de la consigne totale (somme 4 moteurs).

        Par défaut, on considère que chaque stratégie connaît sa limite
        (typiquement 4 * Cmax_per_wheel). Si non surchargée, on renvoie +inf.
        """
        return float("inf")

    def reset(self) -> None:
        raise NotImplementedError

    def allocate(self, C_total_motor: float, speed_motor_rpm: float, k: int) -> tuple[list[float], float, float, float, float]:
        """Retour:
        - refs_motors: [C_fl, C_fr, C_rl, C_rr] (couples moteurs)
        - Cav (par roue avant), Car (par roue arrière)
        - eta_front, eta_rear (cosphi)
        - score
        """
        raise NotImplementedError


class Strat1_LS(StrategyBase):
    """Stratégie 1 : baseline LS (allocation_couple.py)."""

    def __init__(self, a1: float = 0.7, a2: float = 0.3, couple_max_moteur: float = 120.0):
        self.name = "1) LS (single fit, no regen)"
        self.a1 = float(a1)
        self.a2 = float(a2)
        self.couple_max_moteur = float(couple_max_moteur)
        self.reset()

    def reset(self) -> None:
        # EVTorqueOptimizerLS contient en dur (0.7/0.3) dans _cout.
        self.opt = alloc_ls.EVTorqueOptimizerLS(
            couple_data=alloc_ls.TORQUE_DATA,
            vitesse_data=alloc_ls.SPEED_DATA,
            cosphi_data=alloc_ls.COSPHI_DATA,
            couple_max_moteur=self.couple_max_moteur,
            cosphi_min=0.0,
            cosphi_max=1.0,
        )

    @property
    def C_total_motor_limit(self) -> float:
        # 4 roues, limite par roue
        return 4.0 * self.couple_max_moteur

    def allocate(self, C_total_motor: float, speed_motor_rpm: float, k: int):
        # Sécurité : stratégie LS ne gère pas le négatif.
        C_total_motor = float(max(0.0, C_total_motor))

        # Cohérence domaine cartographie : on évite une extrapolation extrême.
        v_rpm = float(np.clip(speed_motor_rpm, np.min(alloc_ls.SPEED_DATA), np.max(alloc_ls.SPEED_DATA)))

        # Si la consigne dépasse la limite (4*Cmax), on la sature.
        C_total_motor = float(np.clip(C_total_motor, 0.0, self.C_total_motor_limit))

        try:
            res = self.opt.calculer_repartition_optimale(consigne_globale=C_total_motor, vitesse=v_rpm)
        except alloc_ls.AllocationCoupleImpossible:
            # Fallback robuste : répartition égale saturée.
            Cav = float(np.clip(C_total_motor / 4.0, 0.0, self.couple_max_moteur))
            Car = Cav
            refs = [Cav, Cav, Car, Car]
            eta = float(self.opt.cosphi(Cav, v_rpm))
            score = float(0.7 * eta + 0.3 * eta)
            return refs, Cav, Car, eta, eta, score
        Cav = float(res.couple_roue_avant)
        Car = float(res.couple_roue_arriere)
        refs = [Cav, Cav, Car, Car]
        return refs, Cav, Car, float(res.cosphi_avant), float(res.cosphi_arriere), float(res.score)


class Strat2_Piecewise(StrategyBase):
    """Stratégie 2 : mapping piecewise +/- + optimisation instantanée (allocation_2.py)."""

    def __init__(self, a1: float = 0.7, a2: float = 0.3, allow_regen: bool = True):
        self.name = "2) Piecewise (+/-)"
        self.a1 = float(a1)
        self.a2 = float(a2)
        self.allow_regen = bool(allow_regen)
        self.reset()

    def reset(self) -> None:
        f_final, _, _ = alloc_pw.build_mapping_two_sides(
            alloc_pw.T_data, alloc_pw.S_data, alloc_pw.Z_data,
            grid_n=130,
            clip_01=False,
            show_map_plot=False,
        )
        Cmax = float(np.max(np.abs(alloc_pw.T_data)))
        self._Cmax = Cmax
        self.alloc = alloc_pw.TorqueAllocator(
            cosphi_map=f_final,
            Cmax_per_wheel=Cmax,
            a1=self.a1,
            a2=self.a2,
            allow_regen=self.allow_regen,
            smooth_lambda=0.0,
        )

    @property
    def C_total_motor_limit(self) -> float:
        return 4.0 * float(self._Cmax)

    def allocate(self, C_total_motor: float, speed_motor_rpm: float, k: int):
        out = self.alloc.allocate(float(C_total_motor), float(speed_motor_rpm), Cav_prev=None)
        Cav = float(out["Cav"])
        Car = float(out["Car"])
        refs = [Cav, Cav, Car, Car]
        return refs, Cav, Car, float(out["eta_front"]), float(out["eta_rear"]), float(out["score"])


class Strat3_Smooth(StrategyBase):
    """Stratégie 3 : lissée (pénalités + rate limiter) (allocation_3.py)."""

    def __init__(
        self,
        a1: float = 0.7,
        a2: float = 0.3,
        allow_regen: bool = True,
        lambda1: float = 5e-3,
        lambda2: float = 5e-4,
        dC_max: float = 5.0,
    ):
        self.name = "3) Smooth (penalties + rate limit)"
        self.a1 = float(a1)
        self.a2 = float(a2)
        self.allow_regen = bool(allow_regen)
        self.lambda1 = float(lambda1)
        self.lambda2 = float(lambda2)
        self.dC_max = float(dC_max)
        self.reset()

    def reset(self) -> None:
        f_final = alloc_smooth.build_mapping_two_sides(
            alloc_smooth.T_data, alloc_smooth.S_data, alloc_smooth.Z_data,
            clip_01=False,
        )
        Cmax = float(np.max(np.abs(alloc_smooth.T_data)))
        self._Cmax = Cmax
        self.alloc = alloc_smooth.TorqueAllocatorSmooth(
            cosphi_map=f_final,
            Cmax_per_wheel=Cmax,
            a1=self.a1,
            a2=self.a2,
            allow_regen=self.allow_regen,
            lambda1=self.lambda1,
            lambda2=self.lambda2,
            dC_max=self.dC_max,
        )
        self._Cav_prev = None
        self._Cav_prev2 = None

    @property
    def C_total_motor_limit(self) -> float:
        return 4.0 * float(self._Cmax)

    def allocate(self, C_total_motor: float, speed_motor_rpm: float, k: int):
        out = self.alloc.allocate(
            float(C_total_motor),
            float(speed_motor_rpm),
            Cav_prev=self._Cav_prev,
            Cav_prev2=self._Cav_prev2,
        )
        Cav = float(out["Cav"])
        Car = float(out["Car"])
        refs = [Cav, Cav, Car, Car]

        # mise à jour mémoire
        self._Cav_prev2 = self._Cav_prev
        self._Cav_prev = Cav

        return refs, Cav, Car, float(out["eta_front"]), float(out["eta_rear"]), float(out["score"])


# =============================================================================
# Simulation
# =============================================================================

def simulate(strategy: StrategyBase, t: np.ndarray, C_total_req: np.ndarray, dt: float) -> StrategyResult:
    strategy.reset()

    vehicle = vehmod.Vehicule()
    motors = [Motor2ndOrder() for _ in range(4)]

    v = np.zeros_like(t)
    a = np.zeros_like(t)
    x = np.zeros_like(t)

    Cav = np.zeros_like(t)
    Car = np.zeros_like(t)
    eta_f = np.zeros_like(t)
    eta_r = np.zeros_like(t)
    score = np.zeros_like(t)

    for k in range(len(t)):
        # Sature la consigne totale selon la limite connue de la stratégie.
        C_total_k = float(np.clip(C_total_req[k], -strategy.C_total_motor_limit, strategy.C_total_motor_limit))

        # vitesse moteur en rpm (approx) : omega_wheel = v/r, omega_motor = ratio*omega_wheel
        omega_motor_rpm = (vehicle.v / vehicle.r) * GEAR_RATIO * RPM_PER_RAD_S

        refs, Cav_k, Car_k, ef_k, er_k, sc_k = strategy.allocate(C_total_k, omega_motor_rpm, k)

        # dynamique moteur (couples moteurs réels)
        motor_torques = [mot.step(u, dt) for mot, u in zip(motors, refs)]

        # conversion moteur -> roue
        wheel_torques = [c * GEAR_RATIO for c in motor_torques]

        v_k, a_k, x_k = vehicle.update(wheel_torques, dt)

        # Le modèle longitudinal simplifié ne modélise pas explicitement la chute de couple
        # à haute vitesse (courbe couple/vitesse moteur). Pour éviter des vitesses
        # irréalistes en boucle ouverte, on applique la limite véhicule (15 km/h).
        if v_k > V_MAX_MPS:
            v_k = V_MAX_MPS
            vehicle.v = V_MAX_MPS
            vehicle.a = 0.0

        v[k] = v_k
        a[k] = a_k
        x[k] = x_k
        Cav[k] = Cav_k
        Car[k] = Car_k
        eta_f[k] = ef_k
        eta_r[k] = er_k
        score[k] = sc_k

    return StrategyResult(
        name=strategy.name,
        t=t,
        C_total_req=C_total_req,
        v_mps=v,
        a_mps2=a,
        x_m=x,
        Cav_motor=Cav,
        Car_motor=Car,
        score=score,
        eta_front=eta_f,
        eta_rear=eta_r,
    )


def summarize(results: list[StrategyResult], dt: float) -> None:
    print("\n" + "=" * 110)
    print("RÉSUMÉ COMPARATIF — Boucle ouverte")
    print("=" * 110)
    for r in results:
        dv = np.gradient(r.v_mps, dt)
        dCav = np.gradient(r.Cav_motor, dt)
        dCar = np.gradient(r.Car_motor, dt)

        print(f"\n{r.name}")
        print("-" * 110)
        print(f"V_finale = {r.v_mps[-1]:.3f} m/s | X_finale = {r.x_m[-1]:.1f} m")
        print(f"a_max = {float(np.max(r.a_mps2)):.3f} m/s² | a_min = {float(np.min(r.a_mps2)):.3f} m/s²")
        print(f"Score moyen = {float(np.mean(r.score)):.4f} | Score min/max = {float(np.min(r.score)):.4f}/{float(np.max(r.score)):.4f}")
        print(f"CosPhi (front) moyen = {float(np.mean(r.eta_front)):.4f} | (rear) moyen = {float(np.mean(r.eta_rear)):.4f}")
        print(f"Lissage (RMS dCav/dt) = {float(np.sqrt(np.mean(dCav**2))):.3f} Nm/s | (RMS dCar/dt) = {float(np.sqrt(np.mean(dCar**2))):.3f} Nm/s")

    print("\nNB: Les couples Cav/Car sont des couples *moteur* par roue (AVG=AVD=Cav, ARG=ARD=Car).")
    print("=" * 110 + "\n")


# =============================================================================
# Plots
# =============================================================================

def plot_results(results: list[StrategyResult], out_png_1: str, out_png_2: str) -> None:
    t = results[0].t

    # Figure 1 : vue synthèse
    fig, axes = plt.subplots(4, 1, figsize=(12, 12), sharex=True, constrained_layout=True)

    # 1) Consigne couple total
    axes[0].plot(t, results[0].C_total_req, label="C_total demandé")
    axes[0].set_title("Consigne de couple total (somme 4 moteurs)")
    axes[0].set_ylabel("Couple total moteur (Nm)")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    # 2) Vitesse véhicule
    for r in results:
        axes[1].plot(t, r.v_mps, label=r.name)
    axes[1].set_title("Vitesse véhicule")
    axes[1].set_ylabel("Vitesse (m/s)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    # 3) Répartition couple (par roue moteur)
    for r in results:
        axes[2].plot(t, 2.0 * r.Cav_motor, label=f"{r.name} — AV total (2*Cav)")
        axes[2].plot(t, 2.0 * r.Car_motor, label=f"{r.name} — AR total (2*Car)", linestyle="--")
    axes[2].set_title("Répartition couple moteur par essieu")
    axes[2].set_ylabel("Couple essieu (Nm)")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(ncol=2, fontsize=8)

    # 4) Score
    for r in results:
        axes[3].plot(t, r.score, label=r.name)
    axes[3].set_title("Performance instantanée (score pondéré cos(phi))")
    axes[3].set_xlabel("Temps (s)")
    axes[3].set_ylabel("Score (—)")
    axes[3].grid(True, alpha=0.3)
    axes[3].legend()

    fig.savefig(out_png_1, dpi=160)

    # Figure 2 : détails allocation + cosphi
    fig2, axes2 = plt.subplots(3, 1, figsize=(12, 10), sharex=True, constrained_layout=True)

    for r in results:
        axes2[0].plot(t, r.Cav_motor, label=f"{r.name} — Cav (par roue AV)")
        axes2[0].plot(t, r.Car_motor, label=f"{r.name} — Car (par roue AR)", linestyle="--")
    axes2[0].set_title("Couples moteurs par roue")
    axes2[0].set_ylabel("Couple moteur (Nm)")
    axes2[0].grid(True, alpha=0.3)
    axes2[0].legend(ncol=2, fontsize=8)

    for r in results:
        axes2[1].plot(t, r.eta_front, label=f"{r.name} — cos(phi) AV")
        axes2[1].plot(t, r.eta_rear, label=f"{r.name} — cos(phi) AR", linestyle="--")
    axes2[1].set_title("cos(phi) estimé (avant vs arrière)")
    axes2[1].set_ylabel("cos(phi) (—)")
    axes2[1].grid(True, alpha=0.3)
    axes2[1].legend(ncol=2, fontsize=8)

    for r in results:
        front_share = (2.0 * r.Cav_motor) / (np.maximum(1e-6, np.abs(r.C_total_req)))
        axes2[2].plot(t, front_share, label=r.name)
    axes2[2].set_title("Part de couple envoyée à l'avant")
    axes2[2].set_xlabel("Temps (s)")
    axes2[2].set_ylabel("(2*Cav) / |C_total| (—)")
    axes2[2].grid(True, alpha=0.3)
    axes2[2].legend()

    fig2.savefig(out_png_2, dpi=160)


def main() -> None:
    dt = 0.01
    t = np.arange(0.0, 16.0, dt)
    C_total = torque_profile_total_motor(t)

    strategies: list[StrategyBase] = [
        Strat1_LS(),
        Strat2_Piecewise(),
        Strat3_Smooth(),
    ]

    results = [simulate(s, t, C_total, dt) for s in strategies]

    summarize(results, dt)

    out1 = os.path.join(THIS_DIR, "comparatif_strategies_openloop.png")
    out2 = os.path.join(THIS_DIR, "comparatif_strategies_openloop_detail.png")
    plot_results(results, out1, out2)

    print(f"Figures générées :\n- {out1}\n- {out2}")


if __name__ == "__main__":
    main()
