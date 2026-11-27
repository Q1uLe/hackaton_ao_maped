"""ROS 2 adapters for the walking policy and Safety Block."""

import json

import numpy as np
import rclpy
from geometry_msgs.msg import Twist
from rclpy.executors import ExternalShutdownException, SingleThreadedExecutor
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import QoSProfile, ReliabilityPolicy
from unitree_go.msg import LowCmd, LowState

from .control import POLICY_PERIOD, ROOT, SAFETY_PERIOD, SERVO_MODE
from .policy import RESIDUAL_WEIGHTS, WalkingPolicy
from .safety import SafetyBlock, SelfCollision

LATEST = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
MOTOR_FIELDS = ("q", "dq", "tau", "kp", "kd")
STATE_TIMEOUT = 0.12


def read_command(message):
    return np.array([[getattr(motor, field) for field in MOTOR_FIELDS] for motor in message.motor_cmd[:12]])


def low_command(command):
    message = LowCmd()
    message.head = [0xFE, 0xEF]
    message.level_flag = 0xFF
    for motor, row in zip(message.motor_cmd, command):
        motor.mode = SERVO_MODE
        for field, value in zip(MOTOR_FIELDS, row):
            setattr(motor, field, float(value))
    return message


class PolicyNode(Node):
    def __init__(self, policy):
        super().__init__("t3_walking_policy", parameter_overrides=[Parameter("use_sim_time", value=True)])
        self.policy = policy
        self.velocity = np.zeros(3)
        self.state = None
        self.last_state_time = None
        self.publisher = self.create_publisher(LowCmd, "/lowcmd_bare", 1)
        self.create_subscription(Twist, "/cmd_vel", self.read_velocity, LATEST)
        self.create_subscription(LowState, "/lowstate", self.read_state, LATEST)
        self.create_timer(POLICY_PERIOD, self.publish_command)

    def read_velocity(self, message):
        velocity = np.array([message.linear.x, message.linear.y, message.angular.z])
        if np.isfinite(velocity).all():
            self.velocity = np.clip(velocity, [-0.6, -0.4, -1.5], [0.6, 0.4, 1.5])

    def read_state(self, message):
        q, dq = np.array([(m.q, m.dq) for m in message.motor_state[:12]]).T
        quat = np.asarray(message.imu_state.quaternion, dtype=float)
        if np.isfinite(q).all() and np.isfinite(dq).all() and np.isfinite(quat).all() and np.linalg.norm(quat) > 1e-6:
            self.state = (q, dq, quat)
            self.last_state_time = self.get_clock().now().nanoseconds * 1e-9

    def publish_command(self):
        now = self.get_clock().now().nanoseconds * 1e-9
        if self.state is None or not 0 <= now - self.last_state_time <= STATE_TIMEOUT:
            if self.state is not None:
                self.state = None
                self.policy.reset()
            return
        self.publisher.publish(low_command(self.policy.step(*self.state, self.velocity)))


class SafetyNode(Node):
    def __init__(self, block):
        super().__init__("t3_safety_block", parameter_overrides=[Parameter("use_sim_time", value=True)])
        self.block = block
        self.publisher = self.create_publisher(LowCmd, "/lowcmd", 1)
        self.create_subscription(LowCmd, "/lowcmd_bare", self.read_request, LATEST)
        self.create_timer(SAFETY_PERIOD, self.publish_command)

    def read_request(self, message):
        self.block.receive(read_command(message), self.get_clock().now().nanoseconds * 1e-9)

    def publish_command(self):
        command = self.block.step(self.get_clock().now().nanoseconds * 1e-9)
        self.publisher.publish(low_command(command))


def main():
    config = json.loads((ROOT / "submission.json").read_text(encoding="utf-8"))
    residual = RESIDUAL_WEIGHTS if config["self_trained_rl"] else None
    policy = WalkingPolicy(residual_weights=residual)
    block = SafetyBlock(SelfCollision())
    rclpy.init()
    executor = SingleThreadedExecutor()
    try:
        for node in (PolicyNode(policy), SafetyNode(block)):
            executor.add_node(node)
        executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        nodes = executor.get_nodes()
        executor.shutdown()
        for node in nodes:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
