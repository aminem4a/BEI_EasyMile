# -*- coding: utf-8 -*-
"""
Code BEI - Stratégie de Répartition du Couple pour l'EZDolly
Optimisation multi-objectif: Minimisation de la puissance active (P) et
maximisation du facteur de puissance (Cos(phi)) via minimisation de la puissance apparente (S).
"""

import cvxpy as cp
import numpy as np
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import LinearNDInterpolator 
import os

# ============================================================================
# 0. OUTILS DE CHARGEMENT ET PRÉPARATION DES DONNÉES (EXCEL)
# ============================================================================

def charger_carte_moteur(file_name: str) -> Dict:
    """
    Charge et prépare les données de la carte moteur (T, omega, CosPhi) à partir du fichier Excel.
    """
    
    print(f"\n[CHARGEMENT] Tentative de lecture du fichier: {file_name}")
    
    # Vérifie si le fichier existe
    if not os.path.exists(file_name):
        print(f"❌ Fichier non trouvé: {file_name}")
        print(f"   Répertoire courant: {os.getcwd()}")
        return None
    
    try:
        # Lecture du fichier Excel
        print(f"Lecture du fichier Excel...")
        df = pd.read_excel(file_name, engine='openpyxl')
        
        print(f"✅ Fichier lu avec succès. Dimensions: {df.shape}")
        print(f"Colonnes disponibles: {list(df.columns)}")
        
        # Affichage d'un aperçu
        print("\nAperçu des 5 premières lignes:")
        print(df.head())
        print("\nTypes de données:")
        print(df.dtypes)
        
        # Nettoyage des noms de colonnes
        df.columns = df.columns.astype(str).str.strip()
        
        # Recherche intelligente des colonnes
        column_mapping = {}
        
        # Dictionnaire de patterns pour identifier les colonnes
        patterns = {
            'SpeedRPM': ['speed', 'rpm', 'vitesse', 'n', 'omega', 'rotation', 'tr/min'],
            'TorqueNm': ['torque', 'couple', 't_', 'torq', 'nm', 'moment', 'ct'],
            'CosPhi': ['cos', 'phi', 'cosphi', 'cos_phi', 'facteur', 'pf', 'power_factor', 'cosφ']
        }
        
        print("\nRecherche des colonnes pertinentes...")
        for target_col, patterns_list in patterns.items():
            found = False
            for col in df.columns:
                col_lower = col.lower()
                for pattern in patterns_list:
                    if pattern in col_lower:
                        column_mapping[target_col] = col
                        print(f"  {target_col} → '{col}'")
                        found = True
                        break
                if found:
                    break
            
            if not found:
                print(f"  ❌ {target_col}: colonne non trouvée")
        
        # Vérification et fallback
        if len(column_mapping) < 3:
            print(f"\nColonnes manquantes. Utilisation des 3 premières colonnes...")
            if len(df.columns) >= 3:
                column_mapping['SpeedRPM'] = df.columns[0]
                column_mapping['TorqueNm'] = df.columns[1]
                column_mapping['CosPhi'] = df.columns[2]
                print(f"  Mapping automatique: {column_mapping}")
            else:
                print("  ❌ Pas assez de colonnes pour procéder.")
                return None
        
        # Extraction et nettoyage des données
        print(f"\nExtraction et nettoyage des données...")
        
        # Créer un DataFrame propre
        df_clean = pd.DataFrame()
        
        # Convertir les colonnes en numérique
        df_clean['SpeedRPM'] = pd.to_numeric(df[column_mapping['SpeedRPM']], errors='coerce')
        df_clean['TorqueNm'] = pd.to_numeric(df[column_mapping['TorqueNm']], errors='coerce')
        df_clean['CosPhi'] = pd.to_numeric(df[column_mapping['CosPhi']], errors='coerce')
        
        # Supprimer les lignes avec NaN
        initial_count = len(df_clean)
        df_clean.dropna(subset=['SpeedRPM', 'TorqueNm', 'CosPhi'], inplace=True)
        final_count = len(df_clean)
        
        print(f"  Points initiaux: {initial_count}")
        print(f"  Points après nettoyage: {final_count}")
        print(f"  Points supprimés (NaN): {initial_count - final_count}")
        
        if final_count < 4:
            print(f"  ❌ Trop peu de données valides ({final_count} points)")
            return None
        
        # Calcul des champs dérivés
        print(f"\nCalcul des champs dérivés...")
        df_clean['Omega_rad_s'] = df_clean['SpeedRPM'] * 2 * np.pi / 60
        df_clean['CosPhi'] = df_clean['CosPhi'].clip(lower=0.1, upper=1.0)  # Valeurs réalistes
        
        # Puissance apparente: S = P_mech / Cos(phi)
        df_clean['P_mech_W'] = df_clean['TorqueNm'] * df_clean['Omega_rad_s']
        df_clean['S_Apparent_W'] = df_clean['P_mech_W'] / df_clean['CosPhi']
        
        # Statistiques
        print(f"\n📊 Statistiques des données nettoyées:")
        print(f"  SpeedRPM: {df_clean['SpeedRPM'].min():.0f} - {df_clean['SpeedRPM'].max():.0f} RPM")
        print(f"  TorqueNm: {df_clean['TorqueNm'].min():.1f} - {df_clean['TorqueNm'].max():.1f} Nm")
        print(f"  CosPhi: {df_clean['CosPhi'].min():.3f} - {df_clean['CosPhi'].max():.3f}")
        print(f"  Omega: {df_clean['Omega_rad_s'].min():.1f} - {df_clean['Omega_rad_s'].max():.1f} rad/s")
        print(f"  Points: {len(df_clean)}")
        
        # Préparation pour interpolation
        print(f"\nPréparation de l'interpolateur...")
        points = df_clean[['TorqueNm', 'Omega_rad_s']].values
        values_cosphi = df_clean['CosPhi'].values
        
        try:
            # Création de l'interpolateur
            interpolator = LinearNDInterpolator(points, values_cosphi)
            
            # Test de l'interpolateur
            test_torque = df_clean['TorqueNm'].median()
            test_omega = df_clean['Omega_rad_s'].median()
            test_point = np.array([[test_torque, test_omega]])
            test_value = interpolator(test_point)[0]
            
            if np.isnan(test_value):
                print(f"  ⚠️ Interpolateur retourne NaN au point test ({test_torque:.1f} Nm, {test_omega:.1f} rad/s)")
                # Utiliser l'interpolation au plus proche voisin
                from scipy.interpolate import NearestNDInterpolator
                interpolator = NearestNDInterpolator(points, values_cosphi)
                test_value = interpolator(test_point)[0]
                print(f"  Utilisation de l'interpolateur au plus proche voisin")
            
            print(f"  ✅ Interpolateur créé avec succès")
            print(f"  Test interpolation: CosPhi({test_torque:.1f} Nm, {test_omega:.1f} rad/s) = {test_value:.3f}")
            
        except Exception as e:
            print(f"  ❌ Erreur création interpolateur: {e}")
            return None
        
        return {
            'interpolator_cosphi': interpolator,
            'dataframe': df_clean,
            'points': points,
            'values_cosphi': values_cosphi,
            'values_S': df_clean['S_Apparent_W'].values
        }
        
    except ImportError:
        print(f"❌ Bibliothèque 'openpyxl' manquante.")
        print(f"   Installer avec: pip install openpyxl")
        return None
        
    except Exception as e:
        print(f"❌ Erreur lors du chargement: {e}")
        import traceback
        traceback.print_exc()
        return None


# ============================================================================
# 1. OPTIMISEUR QUADRATIQUE MULTI-OBJECTIF (QP)
# ============================================================================

class QPConsumptionOptimizerEZDolly:
    def __init__(self, moteur_params: dict, motor_map: Dict = None, lambda_S: float = 0.0):
        self.params = moteur_params
        self.T_nom = moteur_params["couple_nominal"]
        self.omega_nom = moteur_params["vitesse_nominale"] * 2 * np.pi / 60
        self.gear_ratio = moteur_params["rapport_reduction"]
        self.wheel_radius = moteur_params["rayon_roue"]
        self.mass_max = moteur_params.get("masse_max", 11900)
        self.eta_nom = moteur_params.get("rendement", 0.95)
        self.num_moteurs = 4
        
        # Coefficients de pondération
        self.lambda_P = 1.0  # Pondération puissance active
        self.lambda_S = lambda_S  # Pondération puissance apparente (pour CosPhi)
        
        self.motor_map = motor_map
        
        # Calcul des contraintes
        self._calculate_constraints()
        
        print(f"\n[OPTIMISEUR QP] Initialisé avec:")
        print(f"  Couple nominal: {self.T_nom} Nm")
        print(f"  Rapport réduction: {self.gear_ratio}:1")
        print(f"  Lambda_S (pondération CosPhi): {self.lambda_S}")
        
    def _calculate_constraints(self):
        """Calcule les contraintes mécaniques"""
        # Couple max avec marge de sécurité
        self.T_max_motor = self.T_nom * 1.2  # 20% de marge
        self.T_min_motor = -self.T_nom * 1.2  # Freinage régénératif
        
        # Contrainte d'adhérence
        g = 9.81
        Fz_per_wheel = self.mass_max * g / 4  # Répartition uniforme
        self.mu = 0.8  # Coefficient d'adhérence route sèche
        T_max_adhesion_wheel = self.mu * Fz_per_wheel * self.wheel_radius
        self.T_max_adhesion_motor = T_max_adhesion_wheel / self.gear_ratio
        
        print(f"[CONTRAINTES] Couple max moteur: {self.T_max_motor:.1f} Nm")
        print(f"[CONTRAINTES] Adhérence max: {self.T_max_adhesion_motor:.1f} Nm")
        
    def get_motor_efficiency(self, T: float, omega: float) -> float:
        """Estime le rendement du moteur"""
        if abs(omega) < 0.1 or abs(T) < 0.1:
            return 0.1  # Rendement minimal à très faible charge
            
        # Normalisation
        T_norm = abs(T) / self.T_nom
        omega_norm = abs(omega) / self.omega_nom
        
        # Modèle simplifié: rendement dépend de la charge et de la vitesse
        load_factor = min(1.0, T_norm * 1.2)  # Plage 0-1.2
        speed_factor = 0.8 + 0.2 * min(1.0, omega_norm)  # Amélioration avec la vitesse
        
        eta = self.eta_nom * (0.7 + 0.3 * load_factor) * speed_factor
        return max(0.1, min(eta, 0.98))  # Bornes réalistes
        
    def calculate_power_consumption(self, T_motor: float, omega_motor: float) -> float:
        """Calcule la puissance électrique consommée"""
        eta = self.get_motor_efficiency(T_motor, omega_motor)
        if eta > 0:
            return (T_motor * omega_motor) / eta
        return 0.0

    def get_motor_cosphi(self, T: float, omega: float) -> float:
        """Obtient le Cos(Phi) par interpolation ou estimation"""
        
        # Si pas de carte moteur, utiliser un modèle simplifié
        if self.motor_map is None or not self.motor_map.get('interpolator_cosphi'):
            # Modèle d'estimation par défaut
            T_norm = abs(T) / self.T_nom
            # Cos(Phi) s'améliore avec la charge
            cosphi_estime = 0.5 + 0.45 * min(1.0, T_norm * 1.5) 
            return max(0.1, min(cosphi_estime, 0.95))

        # Interpolation depuis la carte moteur
        point = np.array([[abs(T), abs(omega)]])
        cosphi = self.motor_map['interpolator_cosphi'](point)[0]
        
        # Si interpolation échoue, prendre la valeur la plus proche
        if np.isnan(cosphi):
            points_map = self.motor_map['points']
            values_map = self.motor_map['values_cosphi']
            distances = np.linalg.norm(points_map - point, axis=1)
            closest_idx = np.argmin(distances)
            cosphi = values_map[closest_idx]
            print(f"  ⚠️ Interpolation NaN, utilisation du plus proche: CosPhi = {cosphi:.3f}")
            
        return max(0.1, min(cosphi, 0.98))
    
    def get_motor_apparent_power(self, T: float, omega: float) -> float:
        """Calcule la puissance apparente S = P_elec / Cos(Phi)"""
        if abs(omega) < 0.1 or abs(T) < 0.1:
            return 0.0
            
        # Puissance électrique
        P_elec = self.calculate_power_consumption(T, omega)
        
        # Cos(Phi)
        cosphi = self.get_motor_cosphi(T, omega)
        
        # Puissance apparente
        if cosphi > 0.05:
            return abs(P_elec / cosphi)
        else:
            return abs(P_elec) / 0.1  # Eviter division par zéro
        
    def optimize(self, T_global_wheel: float, v_vehicle: float, T_previous: List[float] = None) -> Dict:
        """Résout le problème d'optimisation QP"""
        
        print(f"\n[QP OPTIMIZATION] Début optimisation")
        print(f"  Couple global (roue): {T_global_wheel:.1f} Nm")
        print(f"  Vitesse véhicule: {v_vehicle:.2f} m/s")
        
        # Conversion couple roue → moteur
        T_global_motor = T_global_wheel / self.gear_ratio
        print(f"  Couple global (moteur): {T_global_motor:.1f} Nm")
        
        # Couples précédents (pour contrainte de taux)
        if T_previous is None:
            T_previous = [T_global_motor / 4] * 4
            
        # Calcul vitesse rotation moteur
        if abs(v_vehicle) > 0.1:
            omega_wheel = v_vehicle / self.wheel_radius
            omega_motor = omega_wheel * self.gear_ratio
        else:
            omega_motor = 0.0
            
        omega_current = [omega_motor] * 4
        print(f"  Vitesse rotation: {omega_motor:.1f} rad/s")
        
        # Préparation des coefficients pour la fonction objectif
        print(f"  Calcul des coefficients quadratiques...")
        
        H_diag_P, f_coeffs_P = [], []  # Pour puissance active
        H_diag_S, f_coeffs_S = [], []  # Pour puissance apparente (CosPhi)
        a_coeffs_P = []  # Termes constants (pour info)
        
        delta_T = 0.05 * self.T_nom  # Pas pour dérivées numériques
        
        for i in range(self.num_moteurs):
            T_i = T_previous[i]
            omega_i = omega_current[i]
            
            # === Coefficients pour Puissance Active (P) ===
            P_current = self.calculate_power_consumption(T_i, omega_i)
            P_plus = self.calculate_power_consumption(T_i + delta_T, omega_i)
            P_minus = self.calculate_power_consumption(T_i - delta_T, omega_i)
            
            # Dérivées premières et secondes (approximation quadratique)
            b_P = (P_plus - P_minus) / (2 * delta_T)  # Dérivée première
            c_P = max(0.001, (P_plus - 2*P_current + P_minus) / (delta_T**2))  # Dérivée seconde
            a_P = P_current - b_P * T_i - 0.5 * c_P * T_i**2  # Terme constant
            
            H_diag_P.append(c_P)
            f_coeffs_P.append(b_P)
            a_coeffs_P.append(a_P)
            
            # === Coefficients pour Puissance Apparente (S - pour CosPhi) ===
            S_current = self.get_motor_apparent_power(T_i, omega_i)
            S_plus = self.get_motor_apparent_power(T_i + delta_T, omega_i)
            S_minus = self.get_motor_apparent_power(T_i - delta_T, omega_i)
            
            b_S = (S_plus - S_minus) / (2 * delta_T)
            c_S = max(0.001, (S_plus - 2*S_current + S_minus) / (delta_T**2))
            
            H_diag_S.append(c_S)
            f_coeffs_S.append(b_S)
        
        # Combinaison des objectifs avec pondération
        H_diag_new = [self.lambda_P * c_P + self.lambda_S * c_S 
                      for c_P, c_S in zip(H_diag_P, H_diag_S)]
        f_coeffs_new = [self.lambda_P * b_P + self.lambda_S * b_S 
                        for b_P, b_S in zip(f_coeffs_P, f_coeffs_S)]
        
        H_new = np.diag(H_diag_new)  # Matrice Hessienne (diagonale)
        f_new = np.array(f_coeffs_new)  # Vecteur linéaire
        
        print(f"  Coefficients calculés:")
        print(f"    H_diag: {H_diag_new}")
        print(f"    f: {f_coeffs_new}")
        
        # === FORMULATION DU PROBLÈME QP ===
        T = cp.Variable(self.num_moteurs)  # Variables: couples des 4 moteurs
        
        # Fonction objectif: 0.5 * T^T H T + f^T T
        objective = cp.Minimize(0.5 * cp.quad_form(T, H_new) + f_new.T @ T)
        
        # Contraintes
        constraints = [
            cp.sum(T) == T_global_motor,  # Égalité: somme = couple global
            T >= -self.T_max_motor,       # Borne inférieure (freinage)
            T <= self.T_max_motor,        # Borne supérieure
            cp.abs(T) <= self.T_max_adhesion_motor  # Adhérence
        ]
        
        # Contrainte de taux de variation
        if T_previous is not None:
            delta_T_max = 0.3 * self.T_nom  # 30% du couple nominal
            constraints.append(cp.abs(T - T_previous) <= delta_T_max)
            print(f"  Contrainte taux: ΔT ≤ {delta_T_max:.1f} Nm")
        
        # Création et résolution du problème
        prob = cp.Problem(objective, constraints)
        
        try:
            print(f"  Résolution avec OSQP...")
            prob.solve(solver=cp.OSQP, verbose=False, max_iter=2000)
            
            # Analyse du résultat
            if prob.status in ["optimal", "optimal_inaccurate"]:
                T_opt = T.value
                status = prob.status
                print(f"  ✅ Solution {status} trouvée")
            else:
                print(f"  ⚠️ Statut non optimal: {prob.status}")
                # Solution de repli: répartition égale
                T_opt = np.ones(4) * T_global_motor / 4
                status = "fallback"
                print(f"  Utilisation solution de repli (égale)")
                
        except Exception as e:
            print(f"  ❌ Erreur résolution: {e}")
            T_opt = np.ones(4) * T_global_motor / 4
            status = "error"
        
        # Calcul des indicateurs de performance
        total_power = sum(self.calculate_power_consumption(T_opt[i], omega_motor) 
                         for i in range(self.num_moteurs))
        total_apparent_power = sum(self.get_motor_apparent_power(T_opt[i], omega_motor) 
                                  for i in range(self.num_moteurs))
        
        # Cos(Phi) moyen (pondéré par puissance)
        cosphi_mean = total_power / total_apparent_power if total_apparent_power > 0 else 0.0
        
        print(f"  📊 Résultats optimisation:")
        print(f"    Status: {status}")
        print(f"    Couples moteur: {T_opt}")
        print(f"    Puissance active totale: {total_power:.1f} W")
        print(f"    Puissance apparente totale: {total_apparent_power:.1f} W")
        print(f"    Cos(Phi) moyen: {cosphi_mean:.4f}")
        
        # Conversion pour affichage
        T_opt_wheel = T_opt * self.gear_ratio
        
        return {
            'status': status,
            'torque_motor': T_opt,
            'torque_wheel': T_opt_wheel,
            'total_power': total_power,
            'total_apparent_power': total_apparent_power,
            'cosphi_mean': cosphi_mean,
            'omega_motor': omega_motor,
            'coefficients': {
                'H_diag': H_diag_new,
                'f': f_coeffs_new
            }
        }


# ============================================================================
# 2. CLASSES PRINCIPALES EZDOLLY
# ============================================================================

class MoteurAsynchrone:
    """Représente un moteur asynchrone de l'EZDolly"""
    
    def __init__(self, couple_nominal, vitesse_nominale, rendement, pos, 
                 rapport_reduction, rayon_roue, masse_vide, masse_max):
        self.CoupleNominal = couple_nominal
        self.VitesseNominale = vitesse_nominale
        self.rendement = rendement
        self.pos = pos  # Position: "AVG", "AVD", "ARG", "ARD"
        self.RapportdeReduction = rapport_reduction
        self.RayonRoue = rayon_roue
        self.couple_actuel = 0.0
        self.CoupleMax = couple_nominal * 1.2  # Marge de sécurité
        self.puissance_cumulee = 0.0  # Pour suivre la consommation
        
        print(f"[MOTEUR {pos}] Initialisé: {couple_nominal} Nm, réduction {rapport_reduction}:1")
    
    def setCurrentTorque(self, couple):
        """Définit le couple actuel du moteur avec saturation"""
        # Saturation aux limites
        self.couple_actuel = max(-self.CoupleMax, min(couple, self.CoupleMax))
        couple_roue = self.couple_actuel * self.RapportdeReduction
        return self.couple_actuel
    
    def getCurrentTorque(self):
        """Retourne le couple actuel"""
        return self.couple_actuel
    
    def calculate_power(self, omega_motor):
        """Calcule la puissance électrique consommée"""
        if abs(omega_motor) < 0.1:
            return 0.0
            
        # Puissance mécanique
        puissance_mecanique = self.couple_actuel * omega_motor
        
        # Rendement actuel (dépend du couple)
        charge_ratio = min(abs(self.couple_actuel) / self.CoupleNominal, 1.2)
        rendement_actuel = self.rendement * (0.7 + 0.3 * charge_ratio)
        rendement_actuel = max(0.1, min(rendement_actuel, 0.98))
        
        # Puissance électrique
        if rendement_actuel > 0:
            puissance_electrique = puissance_mecanique / rendement_actuel
        else:
            puissance_electrique = 0.0
            
        # Cumul pour statistiques
        self.puissance_cumulee += puissance_electrique
        
        return puissance_electrique
    
    def reset_power(self):
        """Réinitialise le compteur de puissance"""
        self.puissance_cumulee = 0.0


class AllocateurCouple:
    """Gère la répartition du couple entre les 4 moteurs"""
    
    def __init__(self, vehicule):
        self.vehicule = vehicule
        self.use_qp_optimization = False  # Par défaut: répartition égale
        self.qp_optimizer = None
        self.T_previous = None  # Pour contrainte de taux
        
        # Statistiques
        self.puissance_totale = 0.0
        self.cosphi_moyen = 0.0
        
        print("[ALLOCATEUR] Initialisé")
    
    def set_optimization_method(self, use_qp: bool, moteur_params: dict = None, 
                               motor_map_data: Dict = None, lambda_S: float = 0.0):
        """Configure la méthode d'optimisation"""
        self.use_qp_optimization = use_qp
        
        if use_qp and moteur_params is not None:
            self.qp_optimizer = QPConsumptionOptimizerEZDolly(
                moteur_params, motor_map_data, lambda_S
            )
            print(f"[ALLOCATEUR] Optimisation QP activée (lambda_S={lambda_S})")
            return self.qp_optimizer is not None
        else:
            print("[ALLOCATEUR] Méthode égale activée")
            return True
    
    def optiTorque_egale(self, couple_global):
        """Répartition égale du couple (méthode baseline)"""
        print(f"[ALLOCATEUR] Méthode égale: {couple_global:.1f} Nm (roue)")
        
        # Conversion roue → moteur
        gear_ratio = self.vehicule.MotAVG.RapportdeReduction
        couple_global_motor = couple_global / gear_ratio
        couple_par_moteur = np.ones(4) * couple_global_motor / 4
        
        # Calcul vitesse rotation
        v_vehicle = self.vehicule.vitesse_actuelle / 3.6
        if v_vehicle > 0.1:
            omega_motor = (v_vehicle / self.vehicule.MotAVG.RayonRoue) * gear_ratio
        else:
            omega_motor = 0.0
        
        # Application aux moteurs et calcul puissance
        moteurs = [self.vehicule.MotAVG, self.vehicule.MotAVD, 
                  self.vehicule.MotARG, self.vehicule.MotARD]
        
        puissance_totale = 0.0
        puissance_apparente_totale = 0.0
        
        for moteur, couple in zip(moteurs, couple_par_moteur):
            moteur.setCurrentTorque(couple)
            puissance_totale += moteur.calculate_power(omega_motor)
            
            # Pour CosPhi, on utilise l'estimateur de l'optimiseur
            if self.qp_optimizer:
                puissance_apparente_totale += self.qp_optimizer.get_motor_apparent_power(couple, omega_motor)
            else:
                # Estimateur simplifié si pas d'optimiseur
                puissance_apparente_totale += abs(couple * omega_motor) / 0.8  # CosPhi ≈ 0.8
        
        # CosPhi moyen
        self.cosphi_moyen = (puissance_totale / puissance_apparente_totale 
                            if puissance_apparente_totale > 0 else 0.0)
        self.puissance_totale = puissance_totale
        self.T_previous = couple_par_moteur
        
        print(f"  Couples moteur: {couple_par_moteur}")
        print(f"  Puissance: {puissance_totale:.1f} W, CosPhi: {self.cosphi_moyen:.4f}")
        
        return {
            'torque_motor': couple_par_moteur,
            'total_power': puissance_totale,
            'cosphi_mean': self.cosphi_moyen,
            'omega_motor': omega_motor
        }
    
    def optiTorque_qp(self, couple_global):
        """Optimisation QP avancée"""
        if self.qp_optimizer is None:
            print("[ALLOCATEUR] Optimiseur QP non disponible, utilisation méthode égale")
            return self.optiTorque_egale(couple_global)
        
        print(f"[ALLOCATEUR] Optimisation QP: {couple_global:.1f} Nm (roue)")
        
        # Récupération état actuel
        v_vehicle = self.vehicule.vitesse_actuelle / 3.6
        
        # Couples précédents
        if self.T_previous is None:
            moteurs = [self.vehicule.MotAVG, self.vehicule.MotAVD, 
                      self.vehicule.MotARG, self.vehicule.MotARD]
            T_prev = [m.getCurrentTorque() for m in moteurs]
        else:
            T_prev = self.T_previous
        
        # Optimisation
        results = self.qp_optimizer.optimize(couple_global, v_vehicle, T_prev)
        
        # Application des résultats
        omega_motor = results['omega_motor']
        moteurs = [self.vehicule.MotAVG, self.vehicule.MotAVD, 
                  self.vehicule.MotARG, self.vehicule.MotARD]
        
        for moteur, couple in zip(moteurs, results['torque_motor']):
            moteur.setCurrentTorque(couple)
            moteur.calculate_power(omega_motor)
        
        # Mise à jour statistiques
        self.puissance_totale = results['total_power']
        self.cosphi_moyen = results['cosphi_mean']
        self.T_previous = results['torque_motor']
        
        return results
    
    def optiTorque(self, couple_global):
        """Interface principale: choisit la méthode selon configuration"""
        if self.use_qp_optimization:
            return self.optiTorque_qp(couple_global)
        else:
            return self.optiTorque_egale(couple_global)
    
    def getCosPhi(self):
        """Retourne le CosPhi moyen"""
        return self.cosphi_moyen
    
    def reset_power(self):
        """Réinitialise les compteurs de puissance"""
        self.puissance_totale = 0.0
        for m in [self.vehicule.MotAVG, self.vehicule.MotAVD, 
                 self.vehicule.MotARG, self.vehicule.MotARD]:
            m.reset_power()


class ControleurLIN:
    """Contrôleur de haut niveau: calcule le couple global demandé"""
    
    def __init__(self, vehicule):
        self.vehicule = vehicule
        self.couple_global_demande = 0.0
        
        # Caractéristiques moteur
        self.couple_max_moteur = 34.3 * 1.2  # Avec marge
        self.couple_max_roues = self.couple_max_moteur * 4 * 26  # 4 moteurs × réduction
        
        print(f"[CONTROLEUR] Couple max roues: {self.couple_max_roues:.0f} Nm")
    
    def calculateGlobalTorque(self, acceleration_pedal, vitesse_actuelle):
        """Calcule le couple global basé sur pédale et vitesse"""
        
        # Facteur pédale (0-100%)
        pedal_factor = max(0, min(acceleration_pedal, 100)) / 100.0
        
        # Facteur vitesse (réduction à haute vitesse)
        vitesse_m_s = vitesse_actuelle / 3.6
        vitesse_max_m_s = 15 / 3.6  # 15 km/h
        
        if vitesse_m_s <= vitesse_max_m_s:
            speed_factor = 1.0 - (vitesse_m_s / vitesse_max_m_s) * 0.5
        else:
            speed_factor = 0.0
        
        # Couple global
        self.couple_global_demande = self.couple_max_roues * pedal_factor * speed_factor
        
        print(f"[CONTROLEUR] Pédale: {acceleration_pedal}% → {self.couple_global_demande:.1f} Nm")
        return self.couple_global_demande


class Vehicule:
    """Véhicule EZDolly complet"""
    
    def __init__(self):
        # Caractéristiques véhicule
        self.masseaVide = 4900  # kg
        self.ChargeMax = 7000  # kg
        self.VitesseMax = 15  # km/h
        
        # Moteurs (initialisés plus tard)
        self.MotAVG = None
        self.MotAVD = None
        self.MotARG = None
        self.MotARD = None
        
        # Contrôleurs
        self.controleur = ControleurLIN(self)
        self.allocateur = AllocateurCouple(self)
        
        # État
        self.vitesse_actuelle = 0.0
        self.masse_actuelle = self.masseaVide
        
        # Données
        self.moteur_params = {}
        self.motor_map_data = None
        
        print("[VEHICULE] Véhicule EZDolly initialisé")
    
    def initialize_motors(self, moteur_params):
        """Initialise les 4 moteurs avec les paramètres donnés"""
        self.moteur_params = moteur_params
        
        # Création des 4 moteurs identiques
        self.MotAVG = MoteurAsynchrone(pos="AVG", **moteur_params)
        self.MotAVD = MoteurAsynchrone(pos="AVD", **moteur_params)
        self.MotARG = MoteurAsynchrone(pos="ARG", **moteur_params)
        self.MotARD = MoteurAsynchrone(pos="ARD", **moteur_params)
        
        print("[VEHICULE] 4 moteurs initialisés")
    
    def updateVehicleState(self, vitesse, charge=None):
        """Met à jour l'état du véhicule"""
        self.vitesse_actuelle = vitesse
        
        if charge is not None:
            self.masse_actuelle = self.masseaVide + min(charge, self.ChargeMax)
            print(f"[VEHICULE] État: {vitesse} km/h, {self.masse_actuelle:.0f} kg")
    
    def setAllocationMethod(self, use_qp: bool, motor_map_data: Dict = None, 
                           lambda_S: float = 0.0):
        """Configure la méthode d'allocation"""
        self.motor_map_data = motor_map_data
        return self.allocateur.set_optimization_method(
            use_qp, self.moteur_params, motor_map_data, lambda_S
        )
    
    def demanderCouple(self, acceleration_pedal):
        """Chaîne complète de demande de couple"""
        couple_global = self.controleur.calculateGlobalTorque(
            acceleration_pedal, self.vitesse_actuelle
        )
        return self.allocateur.optiTorque(couple_global)
    
    def getTorqueStatus(self):
        """Retourne l'état des couples"""
        return {
            m.pos: m.getCurrentTorque() 
            for m in [self.MotAVG, self.MotAVD, self.MotARG, self.MotARD]
        }
    
    def getTotalPower(self):
        """Retourne la puissance totale consommée"""
        return self.allocateur.puissance_totale
    
    def getCosPhi(self):
        """Retourne le CosPhi moyen"""
        return self.allocateur.getCosPhi()
    
    def reset_power(self):
        """Réinitialise les compteurs de puissance"""
        self.allocateur.reset_power()


# ============================================================================
# 3. FONCTION DE TEST ET COMPARAISON
# ============================================================================

def tester_scenario_cosphi(vehicule, scenario_nom, vitesse, charge, acceleration, 
                          moteur_params, motor_map_data):
    """Teste un scénario avec les trois méthodes d'allocation"""
    
    print(f"\n{'='*80}")
    print(f"SCÉNARIO: {scenario_nom}")
    print(f"Vitesse: {vitesse} km/h, Charge: {charge} kg, Accélération: {acceleration}%")
    print(f"{'='*80}")
    
    # Mise à jour état véhicule
    vehicule.updateVehicleState(vitesse=vitesse, charge=charge)
    
    # Résultats des trois méthodes
    resultats = {'egale': {}, 'qp_P': {}, 'qp_S': {}}
    
    # 1. MÉTHODE RÉPARTITION ÉGALE (Baseline)
    print(f"\n[1] MÉTHODE RÉPARTITION ÉGALE")
    vehicule.setAllocationMethod(use_qp=False, motor_map_data=motor_map_data)
    vehicule.reset_power()
    resultat_egale = vehicule.demanderCouple(acceleration_pedal=acceleration)
    resultats['egale'] = {
        'P': resultat_egale['total_power'],
        'CosPhi': resultat_egale['cosphi_mean'],
        'couples': resultat_egale['torque_motor']
    }
    
    # 2. OPTIMISATION QP PURE EFFICACITÉ (min P, lambda_S = 0.0)
    print(f"\n[2] OPTIMISATION QP PURE EFFICACITÉ (min P)")
    vehicule.setAllocationMethod(use_qp=True, motor_map_data=motor_map_data, lambda_S=0.0)
    vehicule.reset_power()
    resultat_qp_P = vehicule.demanderCouple(acceleration_pedal=acceleration)
    resultats['qp_P'] = {
        'P': resultat_qp_P['total_power'],
        'CosPhi': resultat_qp_P['cosphi_mean'],
        'couples': resultat_qp_P['torque_motor']
    }
    
    # 3. OPTIMISATION QP PRIORITÉ COS(PHI) (min P + S, lambda_S > 0)
    print(f"\n[3] OPTIMISATION QP PRIORITÉ COS(PHI) (min P + S)")
    LAMBDA_COS_PHI_PRIO = 1.0  # Forte pondération sur CosPhi
    vehicule.setAllocationMethod(use_qp=True, motor_map_data=motor_map_data, 
                                lambda_S=LAMBDA_COS_PHI_PRIO)
    vehicule.reset_power()
    resultat_qp_S = vehicule.demanderCouple(acceleration_pedal=acceleration)
    resultats['qp_S'] = {
        'P': resultat_qp_S['total_power'],
        'CosPhi': resultat_qp_S['cosphi_mean'],
        'couples': resultat_qp_S['torque_motor']
    }
    
    # AFFICHAGE SYNTHÈSE
    print(f"\n{'─'*80}")
    print(f"📊 SYNTHÈSE DES RÉSULTATS")
    print(f"{'─'*80}")
    print(f"{'MÉTHODE':<30} {'Puissance P (W)':<20} {'Cos(phi)':<15} {'Gain P':<10} {'Gain CosPhi':<15}")
    print(f"{'─'*30} {'─'*20} {'─'*15} {'─'*10} {'─'*15}")
    
    # Calcul des gains relatifs
    base_power = resultats['egale']['P']
    base_cosphi = resultats['egale']['CosPhi']
    
    for methode, data in resultats.items():
        nom_affichage = {
            'egale': '1. Répartition Égale',
            'qp_P': '2. QP min P (Efficacité)',
            'qp_S': '3. QP min P + S (CosPhi)'
        }[methode]
        
        gain_p = ((base_power - data['P']) / base_power * 100) if base_power > 0 else 0
        gain_cosphi = (data['CosPhi'] - base_cosphi) * 100  # Différence en points %
        
        print(f"{nom_affichage:<30} {data['P']:<20.1f} {data['CosPhi']:<15.4f} "
              f"{gain_p:>+8.1f}% {gain_cosphi:>+13.2f} pts")
    
    # ANALYSE DÉTAILLÉE
    print(f"\n{'─'*80}")
    print(f"📈 ANALYSE DÉTAILLÉE")
    print(f"{'─'*80}")
    
    # Distribution des couples
    print(f"\nDistribution des couples (Nm moteur):")
    print(f"{'Moteur':<10} {'Égale':<15} {'QP min P':<15} {'QP min P+S':<15}")
    print(f"{'─'*10} {'─'*15} {'─'*15} {'─'*15}")
    
    positions = ['AVG', 'AVD', 'ARG', 'ARD']
    for i, pos in enumerate(positions):
        print(f"{pos:<10} "
              f"{resultats['egale']['couples'][i]:<15.2f} "
              f"{resultats['qp_P']['couples'][i]:<15.2f} "
              f"{resultats['qp_S']['couples'][i]:<15.2f}")
    
    # Conclusions
    print(f"\n{'─'*80}")
    print(f"✅ CONCLUSIONS")
    print(f"{'─'*80}")
    
    if resultats['qp_S']['CosPhi'] > resultats['egale']['CosPhi']:
        gain_cosphi = resultats['qp_S']['CosPhi'] - resultats['egale']['CosPhi']
        print(f"✓ La stratégie 'QP min P + S' améliore le Cos(phi) de {gain_cosphi:.4f}")
        
        if resultats['qp_S']['P'] < resultats['egale']['P']:
            gain_p = (resultats['egale']['P'] - resultats['qp_S']['P']) / resultats['egale']['P'] * 100
            print(f"✓ Double bénéfice: Cos(phi) ↑ et Puissance ↓ de {gain_p:.1f}%")
        else:
            perte_p = (resultats['qp_S']['P'] - resultats['egale']['P']) / resultats['egale']['P'] * 100
            print(f"⚠ Trade-off: Cos(phi) ↑ mais Puissance ↑ de {perte_p:.1f}%")
    else:
        print(f"⚠ Dans ce scénario, l'optimisation n'améliore pas significativement le Cos(phi)")
    
    return resultats


# ============================================================================
# 4. VISUALISATION DES RÉSULTATS
# ============================================================================

def visualiser_resultats(resultats, scenario_nom):
    """Crée des graphiques pour visualiser les résultats"""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Analyse des stratégies - {scenario_nom}', fontsize=16)
    
    méthodes = ['Répartition Égale', 'QP min P', 'QP min P+S']
    valeurs_p = [resultats['egale']['P'], resultats['qp_P']['P'], resultats['qp_S']['P']]
    valeurs_cosphi = [resultats['egale']['CosPhi'], resultats['qp_P']['CosPhi'], 
                     resultats['qp_S']['CosPhi']]
    
    # 1. Puissance consommée
    ax1 = axes[0, 0]
    bars1 = ax1.bar(méthodes, valeurs_p, color=['blue', 'green', 'red'])
    ax1.set_ylabel('Puissance (W)')
    ax1.set_title('Puissance électrique consommée')
    ax1.grid(True, alpha=0.3)
    
    # Ajout des valeurs sur les barres
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 5,
                f'{height:.0f}', ha='center', va='bottom')
    
    # 2. Cos(phi)
    ax2 = axes[0, 1]
    bars2 = ax2.bar(méthodes, valeurs_cosphi, color=['blue', 'green', 'red'])
    ax2.set_ylabel('Cos(φ)')
    ax2.set_title('Facteur de puissance moyen')
    ax2.set_ylim([0, 1])
    ax2.grid(True, alpha=0.3)
    
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom')
    
    # 3. Distribution des couples
    ax3 = axes[1, 0]
    positions = ['AVG', 'AVD', 'ARG', 'ARD']
    x = np.arange(len(positions))
    width = 0.25
    
    ax3.bar(x - width, resultats['egale']['couples'], width, label='Égale', color='blue')
    ax3.bar(x, resultats['qp_P']['couples'], width, label='QP min P', color='green')
    ax3.bar(x + width, resultats['qp_S']['couples'], width, label='QP min P+S', color='red')
    
    ax3.set_xlabel('Moteur')
    ax3.set_ylabel('Couple (Nm moteur)')
    ax3.set_title('Distribution des couples')
    ax3.set_xticks(x)
    ax3.set_xticklabels(positions)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Trade-off Puissance vs Cos(phi)
    ax4 = axes[1, 1]
    ax4.scatter(valeurs_p, valeurs_cosphi, s=200, color=['blue', 'green', 'red'])
    
    for i, méthode in enumerate(méthodes):
        ax4.annotate(méthode, (valeurs_p[i], valeurs_cosphi[i]), 
                    xytext=(10, 10), textcoords='offset points')
    
    ax4.set_xlabel('Puissance (W)')
    ax4.set_ylabel('Cos(φ)')
    ax4.set_title('Trade-off: Puissance vs Cos(φ)')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


# ============================================================================
# 5. POINT D'ENTRÉE PRINCIPAL
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*80)
    print("STRATÉGIE DE RÉPARTITION DE COUPLE POUR L'EZDOLLY")
    print("Optimisation multi-objectif: Puissance vs Cos(Phi)")
    print("="*80)
    
    # Paramètres moteur EZDolly
    moteur_params = {
        "couple_nominal": 34.3,          # Nm
        "vitesse_nominale": 2200.0,      # tr/min
        "rendement": 0.95,               # Rendement nominal
        "rapport_reduction": 26.0,       # Réduction moteur → roue
        "rayon_roue": 0.24,              # m
        "masse_vide": 4900,              # kg
        "masse_max": 11900               # kg (charge max)
    }
    
    print(f"\n📋 Paramètres EZDolly:")
    print(f"  • Couple nominal: {moteur_params['couple_nominal']} Nm")
    print(f"  • Rapport réduction: {moteur_params['rapport_reduction']}:1")
    print(f"  • Masse max: {moteur_params['masse_max']} kg")
    
    # --- CHARGEMENT DE LA CARTE MOTEUR ---
    file_name = "Classeur1.xlsx"  # Ton fichier Excel
    print(f"\n📂 Chargement de la carte moteur: '{file_name}'")
    
    motor_map_data = charger_carte_moteur(file_name)
    
    if motor_map_data is None:
        print(f"⚠️ Impossible de charger la carte moteur. Utilisation d'un modèle estimé.")
        print(f"   L'optimisation Cos(phi) sera basée sur un modèle théorique.")
    else:
        print(f"✅ Carte moteur chargée avec succès!")
        print(f"   Points de données: {len(motor_map_data['dataframe'])}")
    
    # --- INITIALISATION DU VÉHICULE ---
    print(f"\n🚗 Initialisation du véhicule EZDolly...")
    vehicule = Vehicule()
    vehicule.initialize_motors(moteur_params)
    
    # --- SCÉNARIO 1: FAIBLE CHARGE (CosPhi critique) ---
    print(f"\n" + "="*80)
    print("SCÉNARIO 1: FAIBLE CHARGE")
    print("Condition critique pour le Cos(phi)")
    print("="*80)
    
    resultats_scenario1 = tester_scenario_cosphi(
        vehicule=vehicule,
        scenario_nom="Faible charge (CosPhi critique)",
        vitesse=10.0,        # km/h
        charge=0.0,          # kg (véhicule à vide)
        acceleration=20,     # % (accélération modérée)
        moteur_params=moteur_params,
        motor_map_data=motor_map_data
    )
    
    # Visualisation
    if input("\n📊 Voulez-vous visualiser les résultats? (o/n): ").lower() == 'o':
        visualiser_resultats(resultats_scenario1, "Scénario 1: Faible charge")
    
    # --- SCÉNARIO 2: FORTE CHARGE (Efficacité critique) ---
    print(f"\n" + "="*80)
    print("SCÉNARIO 2: FORTE CHARGE")
    print("Condition critique pour l'efficacité énergétique")
    print("="*80)
    
    resultats_scenario2 = tester_scenario_cosphi(
        vehicule=vehicule,
        scenario_nom="Forte charge (Efficacité critique)",
        vitesse=5.0,         # km/h (vitesse lente)
        charge=7000.0,       # kg (charge maximale)
        acceleration=80,     # % (forte accélération)
        moteur_params=moteur_params,
        motor_map_data=motor_map_data
    )
    
    # Visualisation
    if input("\n📊 Voulez-vous visualiser les résultats? (o/n): ").lower() == 'o':
        visualiser_resultats(resultats_scenario2, "Scénario 2: Forte charge")
    
    # --- SYNTHÈSE FINALE ---
    print(f"\n" + "="*80)
    print("SYNTHÈSE FINALE")
    print("="*80)
    
    print(f"\n✅ SIMULATION TERMINÉE AVEC SUCCÈS")
    print(f"\n📝 Récapitulatif:")
    print(f"  1. Méthode 'Répartition Égale': Baseline simple")
    print(f"  2. Méthode 'QP min P': Optimise l'efficacité énergétique")
    print(f"  3. Méthode 'QP min P + S': Optimise le facteur de puissance Cos(phi)")
    
    print(f"\n💡 Recommandations:")
    print(f"  • En condition de faible charge: Privilégier 'QP min P + S' pour améliorer Cos(phi)")
    print(f"  • En condition de forte charge: Privilégier 'QP min P' pour l'efficacité")
    print(f"  • Pour un compromis: Utiliser lambda_S = 0.5")
    
    print(f"\n" + "="*80)
    print("FIN DE LA SIMULATION")
    print("="*80)