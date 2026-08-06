"""_summary_
The main overarching file that calls everything else.
"""

import argparse
from pathlib import Path

from sim.world import WORLD


# Top level hook for simulation
def sensor_selection_simulator():
    """
    A simulator created by Anthony Thompson for the research of sensor selection algorithms,
    particularly in the subject of tracking moving agents and objects.

    Parameters:
        --config <path>  Yaml file to load simulation settings from.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        help="yaml file to load settings from",
        default="config\\scene\\mountain_range\\scene.yaml",
    )
    #    parser.add_argument("--output_dir", help="path to output directory", default="logs") # Unused so far
    #    parser.add_argument("--n", help="number of trials", default=1, type=int) # Unused so far
    args = parser.parse_args()

    world_config = Path(args.config)
    app = WORLD(str(world_config))

    run(app=app)


def run(app):
    """
    Actually runs the app.
    """
    app.run()


if __name__ == "__main__":
    sensor_selection_simulator()
