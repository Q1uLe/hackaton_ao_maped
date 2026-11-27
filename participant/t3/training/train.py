"""Train a PPO residual for the pretrained policy on the organizers' Go1 model."""

import argparse
import json
from functools import partial
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from torch import nn

from go1_t3.control import POLICY_PERIOD, ROOT, SAFETY_PERIOD
from go1_t3.policy import RESIDUAL_SCALE, WalkingPolicy, yaw_from_quat
from go1_t3.safety import SafetyBlock, SelfCollision
from tools.plant import Plant


class WalkingEnv(gym.Env):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(12,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(84,), dtype=np.float32)
        self.plant = Plant()
        self.policy = WalkingPolicy()
        self.safety = SafetyBlock(SelfCollision())
        self.model = self.plant.model
        self.damping = self.model.dof_damping.copy()
        self.friction = self.model.geom_friction.copy()

    def observe(self):
        q, dq, quat = self.plant.state()
        q += self.np_random.normal(0, 0.003, 12)
        dq += self.np_random.normal(0, 0.06, 12)
        observation, self.base_action = self.policy.observe(q, dq, quat, self.command)
        return observation

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        rng = self.np_random
        self.policy.reset()
        self.safety.reset()
        self.model.dof_damping[:] = self.damping * rng.uniform(0.87, 1.13)
        self.model.geom_friction[:, 0] = self.friction[:, 0] * rng.uniform(0.85, 1.15)
        self.plant.reset(rng)
        self.plant.settle(rng.uniform(0.02, 0.8))
        speed = self.sample_speed(rng)
        self.command = np.array([speed, 0.0, 0.0])
        self.start = self.plant.data.qpos[:2].copy()
        yaw = yaw_from_quat(self.plant.data.qpos[3:7])
        self.forward = np.array([np.cos(yaw), np.sin(yaw)])
        self.lateral = np.array([-np.sin(yaw), np.cos(yaw)])
        self.previous = np.zeros(12)
        self.steps = 0
        self.bad_steps = 0
        return self.observe(), {}

    def sample_speed(self, rng):
        return rng.uniform(self.config["command_min_mps"], self.config["command_max_mps"])

    def apply_command(self, command):
        d = self.plant.data
        self.safety.receive(command, d.time)
        for _ in range(round(POLICY_PERIOD / SAFETY_PERIOD)):
            self.plant.advance(self.safety.step(d.time), SAFETY_PERIOD)

    def step(self, action):
        physical_action = self.base_action + RESIDUAL_SCALE * action
        command = self.policy.command_from_action(physical_action)
        d = self.plant.data
        before = d.qpos[:2].copy()
        self.apply_command(command)
        velocity_xy = (d.qpos[:2] - before) / POLICY_PERIOD
        forward = float(velocity_xy @ self.forward)
        lateral = float(velocity_xy @ self.lateral)
        yaw = yaw_from_quat(d.qpos[3:7])
        yaw_error = yaw - self.policy.heading
        offset = float((d.qpos[:2] - self.start) @ self.lateral)
        self.bad_steps = self.bad_steps + 1 if self.plant.pose_limits().any() else 0
        terminated = self.bad_steps >= 8
        self.steps += 1
        truncated = self.steps * POLICY_PERIOD >= self.config["episode_seconds"]
        reward = (1.8 * np.exp(-((forward - self.command[0]) / 0.35)**2)
                  + 0.6 * forward - 1.1 * lateral**2 - 1.8 * (1 - np.cos(yaw_error))
                  - 0.6 * offset**2 - 1.4 * (d.qpos[2] - 0.27)**2
                  - 0.025 * np.mean(action**2) - 0.045 * np.mean((action - self.previous)**2)
                  - 0.00003 * np.mean(self.plant.last_torque**2) - 6.0 * terminated)
        self.previous = action.copy()
        info = {"distance_m": float((d.qpos[:2] - self.start) @ self.forward),
                "lateral_m": offset, "fallen": terminated}
        return self.observe(), float(reward), terminated, truncated, info


class TrainingLog(BaseCallback):
    def __init__(self, out):
        super().__init__()
        self.out = out
        self.episodes = []
        self.rollouts = 0

    def _on_step(self):
        self.episodes.extend(info["episode"] for info in self.locals["infos"] if "episode" in info)
        return True

    def _on_rollout_start(self):
        # A new rollout starts after PPO has applied the previous optimizer updates.
        if self.rollouts and self.rollouts % 4 == 0:
            self.save()

    def _on_rollout_end(self):
        self.rollouts += 1
        if self.episodes:
            row = {"steps": self.num_timesteps, "episodes": len(self.episodes),
                   "mean_reward": float(np.mean([x["r"] for x in self.episodes])),
                   "mean_distance_m": float(np.mean([x["distance_m"] for x in self.episodes])),
                   "mean_abs_lateral_m": float(np.mean([abs(x["lateral_m"]) for x in self.episodes])),
                   "falls": sum(bool(x["fallen"]) for x in self.episodes)}
            line = json.dumps(row)
            with (self.out / "metrics.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(line + "\n")
            print(line, flush=True)
            self.episodes.clear()

    def save(self):
        actor = [layer for layer in self.model.policy.mlp_extractor.policy_net if isinstance(layer, nn.Linear)]
        actor.append(self.model.policy.action_net)
        weights = {}
        for i, layer in enumerate(actor):
            weights[f"residual_{i}_w"] = layer.weight.detach().cpu().numpy()
            weights[f"residual_{i}_b"] = layer.bias.detach().cpu().numpy()
        np.savez_compressed(self.out / f"residual_{self.num_timesteps:06d}.npz", **weights)

    def _on_training_end(self):
        self.save()
        self.model.save(self.out / "ppo_final")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "training" / "config.json")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    torch.set_num_threads(1)
    env = VecMonitor(SubprocVecEnv([partial(WalkingEnv, config) for _ in range(config["workers"])],
                                 start_method="spawn"),
                     info_keywords=("distance_m", "lateral_m", "fallen"))
    model = PPO("MlpPolicy", env, device="cpu", seed=config["seed"],
                n_steps=config["rollout_steps"], batch_size=config["batch_size"], n_epochs=config["epochs"],
                learning_rate=config["learning_rate"], gamma=config["gamma"], gae_lambda=config["gae_lambda"],
                policy_kwargs={"activation_fn": nn.ELU, "net_arch": {"pi": [64, 32], "vf": [64, 32]},
                               "log_std_init": config["initial_log_std"]})
    with torch.no_grad():
        model.policy.action_net.weight.zero_()
        model.policy.action_net.bias.zero_()
    try:
        model.learn(config["total_steps"], callback=TrainingLog(args.out))
    finally:
        env.close()


if __name__ == "__main__":
    main()
