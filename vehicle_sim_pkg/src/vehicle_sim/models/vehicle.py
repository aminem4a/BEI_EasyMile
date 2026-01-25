import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

class Vehicle:
    """
    Modèle physique calibré pour reproduire fidèlement l'essai EZDolly.
    """
    def __init__(self):
        # --- PARAMÈTRES CALIBRÉS (POUR COLLER AUX DONNÉES) ---
        # Masse ajustée pour correspondre à la dynamique réelle de l'essai
        self.m = 5637.0        
        
        # Données techniques fixes
        self.r = 0.24          
        self.S = 4.0           
        self.Cx = 0.8          
        self.Crr = 0.015       
        self.rho_air = 1.225   
        self.g = 9.81          

        # Frottement mécanique ajusté pour la décélération (Transmission 26:1)
        self.friction_torque = 221.0 

        # État
        self.v = 0.0
        self.x = 0.0
        self.a = 0.0

    def update(self, torques, dt, slope_rad=0.0):
        # 1. Traction
        f_tract = sum(torques) / self.r
        
        # 2. Sens du mouvement
        if abs(self.v) > 1e-3:
            sign_v = np.sign(self.v)
        else:
            sign_v = np.sign(f_tract) if abs(f_tract) > 0 else 0

        # 3. Résistances
        f_aero = 0.5 * self.rho_air * self.S * self.Cx * (self.v**2) * sign_v
        f_roll = self.m * self.g * self.Crr * np.cos(slope_rad) * sign_v
        f_slope = self.m * self.g * np.sin(slope_rad)
        f_mech = (self.friction_torque / self.r) * sign_v  # Pertes mécaniques

        f_resist = f_aero + f_roll + f_slope + f_mech

        # 4. Dynamique
        self.a = (f_tract - f_resist) / self.m
        
        # 5. Intégration
        self.v += self.a * dt
        self.x += self.v * dt

        # Arrêt net
        if abs(self.v) < 0.01 and abs(f_tract) < abs(f_resist):
            self.v = 0.0
            self.a = 0.0

        return self.v, self.a, self.x

