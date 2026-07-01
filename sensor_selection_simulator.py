"""_summary_
The main overarching file that calls everything else.
"""

from pathlib import Path
import argparse
from sim.sim_world import WORLD


def sensor_selection_simulator():
    """The attachment point to run the program. Takes arguements."""

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
