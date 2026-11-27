"""Check every accepted target and its joint-space transition against the T2 model."""

import csv
import math

import mujoco
import numpy as np

from .control import JOINT_MAX, JOINT_MIN, JOINT_NAMES, MODEL_DIR, STAND, TORQUE_LIMIT, position_command

TRANSITION_STEP = np.deg2rad(0.1)
# Keep room for conversion to float32 in LowCmd.
MAX_CHANGE = np.deg2rad(5.0) - 1e-6
HOLD_TIMEOUT = 0.4
POSITION_MARGIN = 1e-6


class SelfCollision:
    def __init__(self, model_dir=MODEL_DIR):
        self.model = mujoco.MjModel.from_xml_path(str(model_dir / "scene.xml"))
        self.data = mujoco.MjData(self.model)
        self.q_addresses = [self.model.joint(name).qposadr.item() for name in JOINT_NAMES]
        with (model_dir / "t2_collision_pairs.csv").open(newline="", encoding="utf-8") as stream:
            checked = [(int(row["geom1_id"]), int(row["geom2_id"]))
                       for row in csv.DictReader(stream) if row["diagnostic_policy"] == "checked"]
        if len(checked) != 655 or self.model.ngeom != 56:
            raise ValueError("Unexpected Go1 model or collision-pair table")
        self.checked = np.zeros((self.model.ngeom, self.model.ngeom), dtype=bool)
        for first, second in checked:
            self.checked[first, second] = self.checked[second, first] = True
        floor = self.model.geom("floor").id
        self.model.geom_contype[floor] = 0
        self.model.geom_conaffinity[floor] = 0
        # A margin lets the narrow phase see approaching pairs even before penetration.
        self.model.geom_margin[:] = np.maximum(self.model.geom_margin, 0.003)
        mujoco.mj_forward(self.model, self.data)

    def collides(self, joint_angles):
        self.data.qpos[self.q_addresses] = joint_angles
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_collision(self.model, self.data)
        if self.data.ncon == 0:
            return False
        contacts = self.data.contact
        flags = contacts.dist < -0.00099
        return bool(np.any(self.checked[contacts.geom1[flags], contacts.geom2[flags]]))

    def transition_is_safe(self, start, target):
        delta = target - start
        samples = math.ceil(np.max(np.abs(delta)) / TRANSITION_STEP)
        # The start is the last accepted safe configuration.
        return all(not self.collides(start + delta * (i / samples)) for i in range(1, samples + 1))


class SafetyBlock:
    def __init__(self, collision):
        self.collision = collision
        self.reset()

    def reset(self):
        self.accepted = position_command(STAND, 40.0, 2.0)
        self.request = None
        self.received_at = None

    def receive(self, command, now):
        if command.shape != (12, 5) or not np.isfinite(command).all():
            self.request = None
            return False
        self.request = command.copy()
        self.received_at = now
        return True

    def step(self, now):
        if self.received_at is None or now - self.received_at >= HOLD_TIMEOUT:
            self.request = None
            self.accepted[:, 1:3] = 0
            self.accepted[:, 3] = np.maximum(self.accepted[:, 3], 20.0)
            self.accepted[:, 4] = np.maximum(self.accepted[:, 4], 0.8)
            return self.accepted
        if self.request is None:
            return self.accepted

        command = self.request.copy()
        target = np.clip(command[:, 0], JOINT_MIN + POSITION_MARGIN, JOINT_MAX - POSITION_MARGIN)
        old = self.accepted[:, 0]
        delta = target - old
        target = old + delta * min(1.0, MAX_CHANGE / max(1e-12, np.max(np.abs(delta))))
        if not self.collision.transition_is_safe(old, target):
            self.request = None
            return self.accepted
        command[:, 0] = target
        command[:, 1] = np.clip(command[:, 1], -30.0, 30.0)
        command[:, 2] = np.clip(command[:, 2], -TORQUE_LIMIT, TORQUE_LIMIT)
        command[:, 3] = np.clip(command[:, 3], 0.0, 100.0)
        command[:, 4] = np.clip(command[:, 4], 0.0, 10.0)
        self.accepted = command
        return self.accepted
