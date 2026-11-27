"""Time real MuJoCo rollouts and PPO optimizer updates on CPU and CUDA."""

import argparse
import json
from functools import partial
from pathlib import Path
from statistics import median

import torch
from stable_baselines3.common.vec_env import SubprocVecEnv

from go1_t3.policy import RESIDUAL_WEIGHTS
from training.ppo_devices import initialize_actor, make_model
from training.train import WalkingEnv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("training/config.json"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--modes", nargs="+", choices=("cpu", "cuda", "hybrid"), default=["cpu", "cuda", "hybrid"])
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    torch.set_num_threads(1)
    result = {"gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
              "workers": args.workers, "rounds": args.rounds, "modes": {}}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    block_size = args.workers * config["rollout_steps"]
    for mode in args.modes:
        timings = []
        env = SubprocVecEnv([partial(WalkingEnv, config) for _ in range(args.workers)], start_method="spawn")
        try:
            model = make_model(env, config, mode)
            initialize_actor(model, RESIDUAL_WEIGHTS)
            model.after_update = lambda m: timings.append({"rollout_s": m.rollout_seconds, "update_s": m.update_seconds})
            model.learn(block_size * args.rounds)
            measured = timings[1:] or timings
            rollout = median(t["rollout_s"] for t in measured)
            update = median(t["update_s"] for t in measured)
            row = {"rollout_s": rollout, "update_s": update,
                   "steps_per_s": block_size / (rollout + update), "iterations": timings}
            print(json.dumps({"mode": mode, **row}), flush=True)
            result["modes"][mode] = row
            args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        finally:
            env.close()


if __name__ == "__main__":
    main()
