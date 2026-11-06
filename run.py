
import math

from sim.settings import AppSettings
from sim.simulation import Simulation


def main():
    settings = AppSettings(
        target_pitch_rad=math.radians(0.0),
        target_yaw_rad=math.radians(0.0),
    )
    sim = Simulation(settings)

    # Run the sim
    sim.run()


if __name__ == '__main__':
    main()
