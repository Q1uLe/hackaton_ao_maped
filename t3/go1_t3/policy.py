"""Walk These Ways policy and its proprioceptive observation history."""

import numpy as np

from .control import POLICY_PERIOD, ROOT, STAND, position_command, rotation_gravity

BASE_WEIGHTS = ROOT / "weights" / "walk_these_ways.npz"
RESIDUAL_WEIGHTS = ROOT / "weights" / "residual.npz"
MOTOR_ORDER = np.array([3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8])
DEFAULT_Q = np.array([0.1, 0.8, -1.5, -0.1, 0.8, -1.5, 0.1, 1.0, -1.5, -0.1, 1.0, -1.5])
ACTION_SCALE = np.tile([0.125, 0.25, 0.25], 4)
COMMAND_SCALE = np.array([2.0, 2.0, 0.25, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.15, 0.3, 0.3, 1.0, 1.0, 1.0])
TROT = np.array([0, 0, 0, 0, 3.0, 0.5, 0, 0, 0.5, 0.08, 0, 0, 0.25, 0.4, 0])
KP = 45.0
KD = 1.0
RESIDUAL_SCALE = 0.35


def yaw_from_quat(quat):
    w, x, y, z = quat / np.linalg.norm(quat)
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y*y + z*z))


def load_layers(archive, prefix, count):
    return tuple((archive[f"{prefix}_{i}_w"], archive[f"{prefix}_{i}_b"]) for i in range(count))


def elu_network(layers, vector):
    for weights, biases in layers[:-1]:
        vector = weights @ vector + biases
        negative = vector < 0
        vector[negative] = np.expm1(vector[negative])
    weights, biases = layers[-1]
    return weights @ vector + biases


class WalkingPolicy:
    def __init__(self, base_weights=BASE_WEIGHTS, residual_weights=None):
        with np.load(base_weights, allow_pickle=False) as archive:
            self.adaptation = load_layers(archive, "adaptation", 3)
            self.actor = load_layers(archive, "actor", 4)
        self.residual = None
        if residual_weights is not None:
            with np.load(residual_weights, allow_pickle=False) as archive:
                self.residual = load_layers(archive, "residual", 3)
        self.gait = TROT.copy()
        self.kp = KP
        self.kd = KD
        self.reset()

    def reset(self):
        self.history = np.zeros((30, 70), dtype=np.float32)
        self.previous = np.zeros(12)
        self.previous2 = np.zeros(12)
        self.clock = np.zeros(4)
        self.phase = 0.0
        self.heading = None

    def observe(self, q, dq, quat, velocity):
        gait = self.gait.copy()
        gait[:3] = velocity
        obs = np.concatenate((rotation_gravity(quat), gait * COMMAND_SCALE,
                              q[MOTOR_ORDER] - DEFAULT_Q, dq[MOTOR_ORDER] * 0.05,
                              self.previous, self.previous2, self.clock)).astype(np.float32)
        self.history[:-1] = self.history[1:]
        self.history[-1] = obs
        history = self.history.ravel()
        latent = elu_network(self.adaptation, history)
        action = elu_network(self.actor, np.concatenate((history, latent)))
        yaw = yaw_from_quat(quat)
        if self.heading is None:
            self.heading = yaw
        self.heading += velocity[2] * POLICY_PERIOD
        error = yaw - self.heading
        features = np.concatenate((obs, action, [np.sin(error), np.cos(error) - 1])).astype(np.float32)
        return features, action

    def command_from_action(self, action):
        self.previous2 = self.previous
        self.previous = np.clip(action, -10, 10)
        self.phase = (self.phase + POLICY_PERIOD * self.gait[4]) % 1.0
        phase, offset, bound = self.gait[5:8]
        foot_phase = self.phase + np.array([phase + offset + bound, offset, bound, phase])
        self.clock = np.sin(2 * np.pi * foot_phase)
        target = DEFAULT_Q + ACTION_SCALE * self.previous
        return position_command(target[MOTOR_ORDER], self.kp, self.kd)

    def step(self, q, dq, quat, velocity):
        if np.max(np.abs(velocity)) < 0.01:
            if self.heading is not None:
                self.reset()
            return position_command(STAND, 40.0, 2.0)
        observation, action = self.observe(q, dq, quat, velocity)
        if self.residual is not None:
            action = action + RESIDUAL_SCALE * np.clip(elu_network(self.residual, observation), -1.0, 1.0)
        return self.command_from_action(action)
