from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from scipy.optimize import minimize_scalar


# =============================================================================
# Données
# =============================================================================
# Recommandation : stocker ces données dans un .npz ou .csv et les charger dans main().
# Ici, on les laisse inline pour le BE / reproductibilité.

TORQUE_DATA = np.array([
    16.583, 18.988, 29.45, 7.644, 32.495, 38.412, 33.2, 15.582, 40.85, 49.164,
    13.289, 12.272, 60.795, 53.025, 117.075, 47.7, 28.035, 2.5, 10.735, 61.824,
    23.381, 6.666, 36.181, 57.018, 10.094, 19.488, 48.609, 22.698, 7.505, 31.6,
    17.85, 28.987, 53.346, 53.53, 52.38, 5.05, 9.595, 20.37, 23.808, 27.371,
    69.319, 14.847, 24.096, 36.865, 72.0, 24.735, 31.824, 40.788, 44.352, 46.272,
    62.304, 76.1, 76.538, 87.87, 83.125, 99.716, 28.684, 87.138, 13.802, 37.595,
    42.33, 42.874, 52.0, 63.08, 67.32, 81.279, 86.632, 87.87, 95.76, 13.39,
    20.055, 22.374, 23.816, 45.374, 44.175, 51.1, 62.296, 73.632, 83.712, 18.236,
    31.556, 45.186, 46.53, 58.092, 65.52, 77.0, 93.66, 98.098, 29.498, 32.11,
    38.178, 54.136, 61.74, 68.02, 16.463, 48.609, 39.382
], dtype=float)

SPEED_DATA = np.array([
    2852.85, 1468.7, 1230.0, 2347.2, 956.87, 1260.72, 1607.55, 1890.05, 894.34, 596.55,
    2195.96, 2371.65, 1432.6, 958.65, 1183.35, 1467.61, 1919.4, 4425.2, 2630.4, 938.08,
    2238.6, 4092.43, 1821.0, 1187.76, 2979.2, 2493.63, 1795.0, 4122.82, 3636.0, 2126.05,
    2834.0, 2802.8, 2609.84, 2687.36, 1705.25, 4382.65, 4425.2, 4066.02, 3392.64, 3498.66,
    2209.66, 4309.52, 2710.0, 2126.05, 1183.05, 2882.88, 3473.91, 3079.65, 2269.44, 2558.78,
    1783.0, 887.0, 1426.56, 1403.15, 1709.73, 1116.47, 2530.5, 1451.38, 3573.9, 2469.94,
    2470.0, 2754.05, 2163.84, 1942.75, 1564.16, 1233.44, 1819.65, 550.2, 1517.19, 4387.76,
    3763.2, 3991.55, 3063.06, 2096.0, 2922.3, 1975.05, 1842.67, 1926.6, 875.67, 4058.48,
    2573.76, 2744.56, 2849.0, 2020.76, 2209.66, 2075.84, 1225.12, 1392.96, 3074.55, 2907.66,
    2586.99, 2271.74, 2309.0, 1757.0, 3983.04, 2376.53, 2546.88
], dtype=float)

COSPHI_DATA = np.array([
    0.2037, 0.4264, 0.5562, 0.5916, 0.5723, 0.5917, 0.6014, 0.6534, 0.627, 0.6767,
    0.735, 0.6745, 0.7056, 0.7154, 0.7008, 0.7178, 0.7828, 0.8085, 0.78, 0.8,
    0.7872, 0.83, 0.83, 0.8051, 0.798, 0.8484, 0.8148, 0.867, 0.8772, 0.8858,
    0.8787, 0.87, 0.9135, 0.8439, 0.8526, 0.836, 0.88, 0.88, 0.8976, 0.8976,
    0.88, 0.8633, 0.8633, 0.9167, 0.8544, 0.855, 0.927, 0.855, 0.9, 0.927,
    0.873, 0.918, 0.918, 0.927, 0.945, 0.945, 0.9464, 0.9373, 0.9476, 0.966,
    0.966, 0.9292, 0.8832, 0.9108, 0.966, 0.9384, 0.8832, 0.9384, 0.92, 0.9021,
    0.9021, 0.93, 0.9021, 0.9579, 0.9486, 0.9579, 0.9207, 0.9765, 0.8835, 0.893,
    0.987, 0.9494, 0.893, 0.94, 0.9588, 0.987, 0.9024, 0.893, 0.912, 0.912,
    0.9792, 0.96, 0.9312, 0.96, 0.9409, 0.9409, 0.9595
], dtype=float)


# =============================================================================
# Types et exceptions
# =============================================================================

@dataclass(frozen=True)
class AllocationCoupleResultat:
    """Résultat unique et typé : évite le mélange dict/erreur."""
    consigne_globale: float
    vitesse: float
    couple_roue_avant: float
    couple_roue_arriere: float
    cosphi_avant: float
    cosphi_arriere: float
    score: float  # J = Kp*cosphi_avant + cosphi_arriere


class AllocationCoupleImpossible(ValueError):
    """Problème impossible à cause des contraintes physiques."""
    pass


# =============================================================================
# Ajustement moindres carrés
# =============================================================================

def ajuster_surface_moindres_carres(
    couple: np.ndarray,
    vitesse: np.ndarray,
    cosphi: np.ndarray
) -> np.ndarray:
    """
    Ajuste la surface :
        cosphi = a*C² + b*V² + c*C*V + d*C + e*V + f
    Retour : coeffs = [a, b, c, d, e, f]
    """
    if not (len(couple) == len(vitesse) == len(cosphi)):
        raise ValueError("Les tableaux couple/vitesse/cosphi doivent avoir la même longueur.")

    A = np.column_stack([
        couple**2,
        vitesse**2,
        couple * vitesse,
        couple,
        vitesse,
        np.ones_like(couple)
    ])
    coeffs, *_ = np.linalg.lstsq(A, cosphi, rcond=None)
    return coeffs


# =============================================================================
# Optimiseur
# =============================================================================

class EVTorqueOptimizerLS:
    """
    Allocation optimale de couple (4 roues motrices, 2 essieux, gauche=droite)
    selon la stratégie du PDF :contentReference[oaicite:0]{index=0} mais avec une surface LS.

    Contrainte :
        2*Cav + 2*Car = Creq  =>  Car = (Creq/2) - Cav

    Objectif :
        Maximiser J(Cav) = a1*η(Cav,V) + a2*η(Car,V)
    """

    def __init__(
        self,
        couple_data: np.ndarray,
        vitesse_data: np.ndarray,
        cosphi_data: np.ndarray,
        couple_max_moteur: float = 120.0,
        cosphi_min: float = 0.0,
        cosphi_max: float = 1.0
    ):
        self.couple_max_moteur = float(couple_max_moteur)
        self.cosphi_min = float(cosphi_min)
        self.cosphi_max = float(cosphi_max)

        # Bornes des données (pour contrôle domaine/unité)
        self._v_min = float(np.min(vitesse_data))
        self._v_max = float(np.max(vitesse_data))
        self._c_min = float(np.min(couple_data))
        self._c_max = float(np.max(couple_data))

        # Ajustement surface LS
        self._coeffs = ajuster_surface_moindres_carres(couple_data, vitesse_data, cosphi_data)

    @property
    def coeffs(self) -> np.ndarray:
        """Copie des coefficients LS ."""
        return self._coeffs.copy()

    # -------------------------------------------------------------------------
    # Vérifications / warnings
    # -------------------------------------------------------------------------
    def _controler_vitesse(self, vitesse: float) -> None:
        """
        Réponse à la remarque : "comment vérifier l'hypothèse d'unité ?"
        On ne prouve pas l'unité, mais on détecte les incohérences :
        si vitesse hors plage de la cartographie, on alerte clairement.
        """
        if vitesse < self._v_min or vitesse > self._v_max:
            warnings.warn(
                f"Vitesse={vitesse:.3f} hors plage cartographie "
                f"[{self._v_min:.3f}, {self._v_max:.3f}]. "
                "Vérifier l'unité (rpm vs rad/s) et/ou risque d'extrapolation.",
                RuntimeWarning
            )

    # -------------------------------------------------------------------------
    # Cartographie cos(phi) via surface LS
    # -------------------------------------------------------------------------
    def cosphi(self, couple: float, vitesse: float) -> float:
        """
        Retourne cos(phi) estimé par LS, borné dans [cosphi_min, cosphi_max].

        Note : couple négatif (freinage) non géré => warning + retour 0.
        """
        if couple < 0.0:
            warnings.warn(
                "Couple négatif (freinage) non géré par cette cartographie : cos(phi)=0.",
                RuntimeWarning
            )
            return 0.0

        if couple <= 0.1:
            return 0.0

        a, b, c, d, e, f = self._coeffs
        val = (
            a * couple**2 +
            b * vitesse**2 +
            c * couple * vitesse +
            d * couple +
            e * vitesse +
            f
        )

        if not np.isfinite(val):
            return 0.0

        return float(np.clip(val, self.cosphi_min, self.cosphi_max))
    
    

    # -------------------------------------------------------------------------
    # Contraintes et optimisation
    # -------------------------------------------------------------------------
    def _bornes_couple_avant(self, consigne_globale: float) -> Tuple[float, float]:
        """
        Bornes admissibles sur Cav (couple d'une roue avant) :
          0 <= Cav <= Cmax
          0 <= Car <= Cmax, avec Car = (Creq/2) - Cav
        """
        min_avant = max(0.0, (consigne_globale - 2.0 * self.couple_max_moteur) / 2.0)
        max_avant = min(self.couple_max_moteur, consigne_globale / 2.0)
        return float(min_avant), float(max_avant)

    def _cout(self, couple_avant: float, consigne_globale: float, vitesse: float) -> float:
        """
        Coût à minimiser : -J.
        Pénalité forte si Car sort des limites.
        """
        couple_arriere = (consigne_globale / 2.0) - couple_avant

        if couple_arriere < 0.0 or couple_arriere > self.couple_max_moteur:
            return 1e6

        eta_avant = self.cosphi(couple_avant, vitesse)
        eta_arriere = self.cosphi(couple_arriere, vitesse)

        return -(0.7 * eta_avant + 0.3 * eta_arriere)

    def calculer_repartition_optimale(
        self,
        consigne_globale: float,
        vitesse: float,
    ) -> AllocationCoupleResultat:
        """
        Retourne la répartition optimale sous forme d'un résultat typé.
        Lève AllocationCoupleImpossible si le problème est infaisable.
        """
        consigne_globale = float(consigne_globale)
        vitesse = float(vitesse)
        self._controler_vitesse(vitesse)

        min_avant, max_avant = self._bornes_couple_avant(consigne_globale)
        if min_avant > max_avant:
            raise AllocationCoupleImpossible(
                "Couple demandé incompatible avec les limites moteurs (problème infaisable)."
            )

        opt_result = minimize_scalar(
            self._cout,
            bounds=(min_avant, max_avant),
            args=(consigne_globale, vitesse),
            method="bounded"
        )

        couple_avant_opt = float(opt_result.x)
        couple_arriere_opt = float((consigne_globale / 2.0) - couple_avant_opt)

        eta_avant = self.cosphi(couple_avant_opt, vitesse)
        eta_arriere = self.cosphi(couple_arriere_opt, vitesse)

        score = float(0.7 * eta_avant + 0.3 * eta_arriere)

        return AllocationCoupleResultat(
            consigne_globale=consigne_globale,
            vitesse=vitesse,
            couple_roue_avant=couple_avant_opt,
            couple_roue_arriere=couple_arriere_opt,
            cosphi_avant=eta_avant,
            cosphi_arriere=eta_arriere,
            score=score
        )
        


# =============================================================================
# Exemple d'utilisation (try/except)
# =============================================================================

def main() -> None:
    optimiseur = EVTorqueOptimizerLS(
        couple_data=TORQUE_DATA,
        vitesse_data=SPEED_DATA,
        cosphi_data=COSPHI_DATA,
        couple_max_moteur=120.0
    )

    print("--- TEST 1 : Faible Charge ---")
    try:
        r1 = optimiseur.calculer_repartition_optimale(consigne_globale=40.0, vitesse=2000.0)
        print(r1)
    except AllocationCoupleImpossible as e:
        print(f"Impossible: {e}")

    print("\n--- TEST 2 : Forte Charge ---")
    try:
        r2 = optimiseur.calculer_repartition_optimale(consigne_globale=300.0, vitesse=2000.0)
        print(r2)
    except AllocationCoupleImpossible as e:
        print(f"Impossible: {e}")


if __name__ == "__main__":
    main()
