"""Training perturbations and fixed validation scenarios for the flat Go1 course."""

from dataclasses import dataclass
from statistics import mean, median

import numpy as np

from go1_t3.control import POLICY_PERIOD
from go1_t3.policy import WalkingPolicy
from go1_t3.safety import SafetyBlock, SelfCollision
from tools.evaluate import episode
from tools.plant import Plant
from training.train import WalkingEnv


@dataclass(frozen=True)
class Push:
    at: float
    duration: float
    force: tuple[float, float]


class Disturbance:
    def __init__(self, friction=1.0, damping=1.0, pushes=(), delayed=False):
        self.friction = friction
        self.damping = damping
        self.pushes = pushes
        self.delayed = delayed
        self.previous = None

    def prepare(self, plant):
        self.previous = None
        self.trunk = plant.model.body("trunk").id
        plant.model.geom_friction[:, 0] *= self.friction
        plant.model.dof_damping[:] *= self.damping

    def apply(self, plant, elapsed):
        xy = plant.data.xfrc_applied[self.trunk, :2]
        xy[:] = 0
        for pulse in self.pushes:
            if pulse.at <= elapsed < pulse.at + pulse.duration:
                xy[:] += pulse.force

    def delay(self, command):
        if not self.delayed:
            return command
        previous, self.previous = self.previous, command
        return command if previous is None else previous


class FineTuneEnv(WalkingEnv):
    def __init__(self, config):
        super().__init__(config)
        self.difficulty = config["starting_difficulty"]

    def set_difficulty(self, difficulty):
        self.difficulty = difficulty

    def sample_speed(self, rng):
        if rng.random() < 0.65:
            return self.config["command_max_mps"]
        return super().sample_speed(rng)

    def reset(self, **kwargs):
        observation, info = super().reset(**kwargs)
        rng = self.np_random
        d = self.difficulty
        self.motion_start = self.plant.data.time
        if rng.random() < 0.35:
            self.model.geom_friction[:] = self.friction
            self.model.dof_damping[:] = self.damping
            self.disturbance = Disturbance()
        else:
            pushes = []
            if rng.random() < 0.35 + 0.55 * d:
                for at in [rng.uniform(1.8, 5.0), rng.uniform(6.5, 9.0)][:1 + int(d > 0.55)]:
                    phi = rng.uniform(-np.pi, np.pi)
                    force = rng.uniform(10, 16 + 27 * d)
                    pushes.append(Push(at, rng.uniform(0.09, 0.18), (force * np.cos(phi), force * np.sin(phi))))
            self.disturbance = Disturbance(
                rng.uniform(1 - 0.27 * d, 1 + 0.23 * d), rng.uniform(1 - 0.23 * d, 1 + 0.23 * d),
                pushes, delayed=rng.random() < 0.35 * d,
            )
        self.disturbance.prepare(self.plant)
        self.smoothed_speed = float(self.command[0])
        self.last_distance = 0.0
        return observation, info

    def apply_command(self, command):
        self.disturbance.apply(self.plant, self.plant.data.time - self.motion_start)
        super().apply_command(self.disturbance.delay(command))

    def step(self, action):
        observation, reward, terminated, truncated, info = super().step(action)
        velocity = (info["distance_m"] - self.last_distance) / POLICY_PERIOD
        self.last_distance = info["distance_m"]
        self.smoothed_speed += 0.08 * (velocity - self.smoothed_speed)
        # Average over the gait cycle and increasingly penalize sustained lateral drift.
        speed_error = (self.smoothed_speed - self.command[0]) / 0.23
        reward += 0.85 * np.exp(-speed_error**2) - 1.6 * min(2.0, info["lateral_m"]**2)
        return observation, float(reward), terminated, truncated, info


class Validation:
    def __init__(self, *, holdout=False):
        self.holdout = holdout
        self.plant = Plant()
        self.friction = self.plant.model.geom_friction.copy()
        self.damping = self.plant.model.dof_damping.copy()
        self.collision = SelfCollision()

    def evaluate(self, weights):
        policy = WalkingPolicy(residual_weights=weights)
        safety = SafetyBlock(self.collision)
        rows = []
        nominal_count = 10 if self.holdout else 3
        perturb_count = 12 if self.holdout else 4
        first_seed = 792000 if self.holdout else 625000
        for warm in (False, True):
            for i in range(nominal_count):
                seed = first_seed + 33 * warm + i
                rows.append(self.run_episode(policy, safety, "warmup_walk" if warm else "nominal", seed, 0.6, warm))
        for i in range(perturb_count):
            seed = first_seed + 70 + i
            rng = np.random.default_rng(seed + 275)
            direction = 1 if i % 2 else -1
            pulse = Push(3.3 + 0.15 * (i % 4), 0.14, (rng.uniform(-12, 12), direction * rng.uniform(21, 38)))
            disturbance = Disturbance(rng.uniform(0.78, 1.22), rng.uniform(0.82, 1.20), (pulse,), delayed=i % 3 == 0)
            rows.append(self.run_episode(policy, safety, "perturbed", seed, 0.6 if i % 3 else 0.45,
                                         i % 2 == 0, disturbance))
        nominal = [r for r in rows if r["scenario"] != "perturbed"]
        perturbed = [r for r in rows if r["scenario"] == "perturbed"]
        nominal_time = median(r["scored_seconds"] for r in nominal)
        nominal_lateral = mean(abs(r["lateral_m"]) for r in nominal)
        disturbed_lateral = mean(abs(r["lateral_m"]) for r in perturbed)
        cost = nominal_time + 4.0 * nominal_lateral + 1.5 * disturbed_lateral
        return {"nominal_successes": sum(r["success"] for r in nominal), "nominal_total": len(nominal),
                "perturbed_successes": sum(r["success"] for r in perturbed), "perturbed_total": len(perturbed),
                "median_nominal_seconds": nominal_time, "mean_abs_nominal_lateral_m": nominal_lateral,
                "mean_abs_perturbed_lateral_m": disturbed_lateral, "cost": cost, "episodes": rows}

    def run_episode(self, policy, safety, scenario, seed, speed, warm, disturbance=None):
        self.plant.model.geom_friction[:] = self.friction
        self.plant.model.dof_damping[:] = self.damping
        metrics = episode(self.plant, policy, safety, np.random.default_rng(seed), speed, 90 if self.holdout else 16,
                          warmup_walk=warm, disturbance=disturbance)
        return {"scenario": scenario, "seed": seed, "speed_mps": speed, "warmup_walk": warm, **metrics}


def score(metrics):
    return metrics["nominal_successes"], metrics["perturbed_successes"], -metrics["cost"]
