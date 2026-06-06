import argparse
from sim import print_helpers as ph
from sim.World.sim_world import WORLD
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        help="yaml file to load settings from",
        default="config\scene\mountain_range\scene.yaml",
    )
    parser.add_argument("--output_dir", help="path to output directory", default="logs")
    parser.add_argument("--n", help="number of trials", default=1, type=int)
    args = parser.parse_args()

    world_config = Path(args.config)
    app = WORLD(str(world_config))

    run(app=app)

def run(app):
    app.run()

if __name__ == "__main__":
    main()