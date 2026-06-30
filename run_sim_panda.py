from pathlib import Path
import argparse
from sim.World.sim_world import WORLD

def main():
    """The main function that runs the simulation."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        help="yaml file to load settings from",
        default="config\\scene\\mountain_range\\scene.yaml",
    )
    parser.add_argument("--output_dir", help="path to output directory", default="logs")
    parser.add_argument("--n", help="number of trials", default=1, type=int)
    args = parser.parse_args()

    world_config = Path(args.config)
    app = WORLD(str(world_config))

    run(app=app)

def run(app):
    """
    Actually runs the app. 
    TODO: Put this function into its own function that make that file what we call to run the simulation.
    """
    app.run()


if __name__ == "__main__":
    main()
