import matplotlib.pyplot as plt
from vehicle_sim.simulation import Simulation
from vehicle_sim.config import SimulationConfig, VehicleConfig

def main():
    # Setup configs (Masse réelle EZDolly)
    sim_cfg = SimulationConfig(dt=0.01, duration=15.0)
    veh_cfg = VehicleConfig(mass=5637.0)

    # Initialize Simulation
    sim = Simulation(sim_cfg, veh_cfg)

    # Run Simulation with target speed of 15 km/h (4.16 m/s)
    results = sim.run(target_speed=15.0/3.6)

    # Plot Results
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    # Vitesse
    ax1.plot(results["time"], [v*3.6 for v in results["velocity"]], label="Réel")
    ax1.axhline(15.0, color='r', linestyle='--', label="Cible")
    ax1.set_title("Réponse Vitesse EZDolly")
    ax1.set_ylabel("Vitesse (km/h)")
    ax1.grid(True)
    ax1.legend()
    
    # Couples (Vérification Allocation)
    ax2.plot(results["time"], results["torque_fl"], label="Moteur AV")
    ax2.plot(results["time"], results["torque_rl"], label="Moteur AR")
    ax2.set_title("Allocation Couple Moteur (Optimisation)")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Couple (Nm)")
    ax2.grid(True)
    ax2.legend()
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()