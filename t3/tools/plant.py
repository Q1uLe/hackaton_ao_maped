"""Headless, accelerated model of the public simulator's LowCmd/LowState loop."""

import mujoco
import numpy as np

from go1_t3.control import JOINT_NAMES, MODEL_DIR, STAND, TORQUE_LIMIT, position_command


class Plant:
    def __init__(self, model_dir=MODEL_DIR):
        self.model = mujoco.MjModel.from_xml_path(str(model_dir / "scene.xml"))
        self.data = mujoco.MjData(self.model)
        self.home = self.model.key("home").id
        self.q_addresses = [self.model.joint(name).qposadr.item() for name in JOINT_NAMES]
        self.dq_addresses = [self.model.joint(name).dofadr.item() for name in JOINT_NAMES]
        self.model.actuator_gainprm[:, 0] = 0.0
        self.model.actuator_biasprm[:, :] = 0.0
        self.dt = self.model.opt.timestep
        self.reset()

    def reset(self, rng=None):
        d = self.data
        mujoco.mj_resetDataKeyframe(self.model, d, self.home)
        if rng is not None:
            d.qpos[self.q_addresses] += rng.normal(0, 0.02, 12)
            d.qvel[:] = rng.normal(0, 0.07, self.model.nv)
            roll, pitch, yaw = rng.uniform(-0.04, 0.04, 3)
            mujoco.mju_euler2Quat(d.qpos[3:7], np.array([roll, pitch, yaw]), "xyz")
        d.ctrl[:] = 0
        mujoco.mj_forward(self.model, d)
        self.last_torque = np.zeros(12)

    def state(self):
        d = self.data
        return d.qpos[self.q_addresses], d.qvel[self.dq_addresses], d.qpos[3:7].copy()

    def advance(self, command, period):
        d = self.data
        target, speed, torque, kp, kd = command.T
        for _ in range(round(period / self.dt)):
            q = d.qpos[self.q_addresses]
            dq = d.qvel[self.dq_addresses]
            effort = torque + kp * (target - q) + kd * (speed - dq)
            self.last_torque = np.clip(effort, -TORQUE_LIMIT, TORQUE_LIMIT)
            d.qfrc_applied[self.dq_addresses] = self.last_torque
            mujoco.mj_step(self.model, d)

    def settle(self, seconds):
        self.advance(position_command(STAND, 40, 2), seconds)

    def pose_limits(self):
        _, _, z, w, x, y, qz = self.data.qpos[:7]
        roll = np.arctan2(2 * (w * x + y * qz), 1 - 2 * (x*x + y*y))
        pitch = np.arcsin(np.clip(2 * (w * y - qz * x), -1.0, 1.0))
        return np.array([z < 0.16, abs(roll) > np.deg2rad(50), abs(pitch) > np.deg2rad(50)])
