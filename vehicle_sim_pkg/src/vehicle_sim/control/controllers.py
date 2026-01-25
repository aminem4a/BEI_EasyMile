class SpeedController:
    """
    SISO Controller: Speed Error -> Total Torque Command.
    """
    def __init__(self, kp: float, ki: float, kd: float):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.prev_error = 0.0
        self.integral = 0.0
        self.max_integral = 5000.0 # Anti-windup simple

    def compute_command(self, target_speed: float, current_speed: float, dt: float) -> float:
        """
        Compute total torque requirement based on speed error.
        """
        error = target_speed - current_speed
        
        # Proportional
        p_term = self.kp * error
        
        # Integral (with anti-windup clamping)
        self.integral += error * dt
        self.integral = max(-self.max_integral, min(self.integral, self.max_integral))
        i_term = self.ki * self.integral
        
        # Derivative
        d_term = self.kd * (error - self.prev_error) / dt
        self.prev_error = error
        
        total_torque = p_term + i_term + d_term
        return total_torque