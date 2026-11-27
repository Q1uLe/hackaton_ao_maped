"""PPO with a CPU rollout actor and optional CUDA optimizer updates."""

from copy import deepcopy
from time import perf_counter

import numpy as np
import torch
from stable_baselines3 import PPO
from torch import nn


class TimedPPO(PPO):
    def __init__(self, *args, rollout_on_cpu=False, **kwargs):
        self.rollout_on_cpu = rollout_on_cpu
        self.cpu_policy = None
        self.rollout_seconds = 0.0
        self.update_seconds = 0.0
        self.after_update = None
        super().__init__(*args, **kwargs)

    def _excluded_save_params(self):
        return [*super()._excluded_save_params(), "cpu_policy", "after_update"]

    def collect_rollouts(self, *args, **kwargs):
        start = perf_counter()
        policy, device = self.policy, self.device
        if self.rollout_on_cpu and device.type == "cuda":
            if self.cpu_policy is None:
                self.cpu_policy = deepcopy(policy).cpu()
            self.cpu_policy.load_state_dict(policy.state_dict())
            self.policy, self.device = self.cpu_policy, torch.device("cpu")
        try:
            return super().collect_rollouts(*args, **kwargs)
        finally:
            self.policy, self.device = policy, device
            self.rollout_seconds = perf_counter() - start

    def train(self):
        start = perf_counter()
        super().train()
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        self.update_seconds = perf_counter() - start
        if self.after_update is not None:
            self.after_update(self)


def actor_layers(model):
    actor = model.policy.mlp_extractor.policy_net
    return [*(layer for layer in actor if isinstance(layer, nn.Linear)), model.policy.action_net]


def initialize_actor(model, initial):
    with np.load(initial, allow_pickle=False) as weights, torch.no_grad():
        for i, layer in enumerate(actor_layers(model)):
            layer.weight.copy_(torch.from_numpy(weights[f"residual_{i}_w"]))
            layer.bias.copy_(torch.from_numpy(weights[f"residual_{i}_b"]))


def make_model(env, config, mode):
    return TimedPPO(
        "MlpPolicy", env, device="cpu" if mode == "cpu" else "cuda",
        rollout_on_cpu=mode == "hybrid", seed=config["seed"],
        n_steps=config["rollout_steps"], batch_size=config["batch_size"], n_epochs=config["epochs"],
        learning_rate=config["learning_rate"], gamma=config["gamma"], gae_lambda=config["gae_lambda"],
        clip_range=config.get("clip_range", 0.2), target_kl=config.get("target_kl"),
        policy_kwargs={"activation_fn": nn.ELU, "net_arch": {"pi": [64, 32], "vf": [64, 32]},
                       "log_std_init": config["initial_log_std"]},
    )
