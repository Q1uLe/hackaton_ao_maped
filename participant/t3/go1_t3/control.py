"""Conventions shared by the policy, Safety Block, and MuJoCo evaluation."""

from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT.parent / "models"
JOINT_NAMES = tuple(
    f"{leg}_{joint}_joint"
    for leg in ("FR", "FL", "RR", "RL")
    for joint in ("hip", "thigh", "calf")
)
STAND = np.tile([0.0, 0.8536, -1.7072], 4)
JOINT_MIN = np.tile([-0.863, -0.686, -2.818], 4)
JOINT_MAX = np.tile([0.863, 4.501, -0.888], 4)
TORQUE_LIMIT = np.tile([23.7, 23.7, 35.55], 4)
SERVO_MODE = 0x0A
POLICY_PERIOD = 0.02
SAFETY_PERIOD = 0.01


def position_command(target, kp, kd):
    """Columns in order: q, dq, tau, kp, kd."""
    command = np.zeros((12, 5))
    command[:, 0] = target
    command[:, 3] = kp
    command[:, 4] = kd
    return command


def rotation_gravity(quat):
    """World unit gravity in the body frame, with quaternion in wxyz order."""
    w, x, y, z = quat / np.linalg.norm(quat)
    return np.array([2 * (w * y - x * z), -2 * (y * z + w * x), 2 * (x*x + y*y) - 1])
