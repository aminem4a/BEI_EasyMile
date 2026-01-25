# src/vehicle_sim/simulation_open_loop.py

from __future__ import annotations
import numpy as np

from vehicle_sim.models.vehicle import Vehicle
from vehicle_sim.control.allocation import TorqueAllocatorSmooth
from vehicle_sim.config import SimulationConfig, VehicleConfig


class SimulationOpenLoop:
    """
    Simulation en boucle ouverte.
    Les consignes (couple total ou couples roues) sont imposées par le scénario.
    """

    def __init__(
        self,
        sim_cfg: SimulationConfig,
        veh_cfg: VehicleConfig,
        allocator: TorqueAllocatorSmooth | None = None,
    ):
        self.sim_cfg = sim_cfg
        self.vehicle = Vehicle(veh_cfg)
        self.allocator = allocator

        self.history = {
            "time": [],
            "speed_mps": [],
            "position_m": [],
            "C_total": [],
            "Cav": [],
            "Car": [],
        }

    def run_total_torque(
        self,
        C_total: np.ndarray,
        speed_rpm: np.ndarray | None = None,
    ):
        """
        Boucle ouverte : C_total(t) imposé.
        """
        dt = self.sim_cfg.dt
        n = len(C_total)

        Cav_prev = None
        Cav_prev2 = None

        for k in range(n):
            t = k * dt

            if self.allocator is not None:
                rpm = (
                    float(speed_rpm[k])
                    if speed_rpm is not None
                    else self.vehicle.speed_rpm()
                )

                out = self.allocator.allocate(
                    float(C_total[k]),
                    rpm,
                    Cav_prev=Cav_prev,
                    Cav_prev2=Cav_prev2,
                )

                Cav_prev2 = Cav_prev
                Cav_prev = out["Cav"]

                Cav = out["Cav"]
                Car = out["Car"]
                wheel_cmds = [Cav, Cav, Car, Car]
            else:
                Cav = Car = float(C_total[k]) / 4.0
                wheel_cmds = [Cav, Cav, Car, Car]

            self.vehicle.update_dynamics(wheel_cmds, dt)

            sel
