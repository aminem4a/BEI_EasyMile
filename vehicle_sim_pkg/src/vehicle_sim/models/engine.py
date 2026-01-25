import numpy as np

class Engine:
    """
    Represents a single engine/motor model using Strejc identification.
    Model: K * exp(-Tu*s) / (1 + Ta*s)^n
    """
    def __init__(self, max_torque: float = 200.0, k: float = 1.0, n: int = 2, ta: float = 0.05, tu: float = 0.02, dt: float = 0.01):
        self.max_torque = max_torque
        self.current_torque = 0.0
        
        # --- Strejc Parameters ---
        self.K = k
        self.n = int(n)
        self.Ta = ta
        self.dt = dt
        
        # Buffer pour le retard pur (Tu)
        steps_delay = int(tu / dt)
        self.buffer = [0.0] * max(1, steps_delay)
        
        # États pour la cascade de filtres (Ordre n)
        self.states = np.zeros(self.n)

    def step(self, torque_cmd: float, dt: float) -> float:
        """
        Updates engine state using Strejc dynamics.
        """
        # Saturation de la commande
        cmd_saturated = max(-self.max_torque, min(torque_cmd, self.max_torque))
        
        # 1. Gain + Retard
        self.buffer.append(cmd_saturated * self.K)
        u_delayed = self.buffer.pop(0)
        
        # 2. Cascade de n filtres du 1er ordre
        input_signal = u_delayed
        for i in range(self.n):
            y_prev = self.states[i]
            # dy/dt = (u - y) / Ta
            dy = (input_signal - y_prev) / self.Ta
            self.states[i] += dy * dt # Integration Euler
            input_signal = self.states[i] # La sortie i devient l'entrée i+1
            
        self.current_torque = self.states[-1]
        return self.current_torque