# -*- coding: utf-8 -*-

class SpeedController:
    """
    Régulateur PI (Proportionnel-Intégral) pour le suivi de vitesse.
    Note: Pas de dérivée (Kd) nécessaire ici car le système est très amorti.
    """
    def __init__(self, kp: float, ki: float, kd: float = 0.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd # Optionnel (par défaut 0.0)
        
        self.integral = 0.0
        self.prev_error = 0.0
        self.max_integ = 5000.0 # Anti-windup

    def compute_command(self, target: float, current: float, dt: float) -> float:
        error = target - current
        
        # Terme Intégral
        self.integral += error * dt
        self.integral = max(-self.max_integ, min(self.integral, self.max_integ))
        
        # Terme Dérivé (si kd est utilisé)
        derivative = (error - self.prev_error) / dt if dt > 0 else 0.0
        self.prev_error = error
        
        return (self.kp * error) + (self.ki * self.integral) + (self.kd * derivative)