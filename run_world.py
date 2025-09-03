import argparse
from stable_baselines3 import PPO
from sim import SensorSelection_Env as SSE
from sim import print_helpers as ph

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        help="json file to load settings from",
        default="config/scene/scene_1.json",
    )
    parser.add_argument("--output_dir", help="path to output directory", default="logs")
    parser.add_argument("--n", help="number of trials", default=1, type=int)
    args = parser.parse_args()

    env = SSE.SensorSelection_Env(config_file=args.config)
    model = PPO("MlpPolicy", env, verbose=1)
    

    for ii in range(args.n):
        try:
            print(f"{ph.BLUE}Trial {ii}/{args.n}{ph.RESET}")
            env.run_sim(model)
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"{ph.RED}Error: {e}{ph.RESET}")


if __name__ == "__main__":
    main()