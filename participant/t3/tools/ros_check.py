"""Score a local episode over ROS 2 with the public simulator's debug odometry."""

import argparse
import json
import time

import numpy as np
import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import qos_profile_sensor_data
from unitree_go.msg import LowCmd

from go1_t3.control import JOINT_MAX, JOINT_MIN
from go1_t3.policy import yaw_from_quat


class EpisodeCheck(Node):
    def __init__(self, speed):
        super().__init__("t3_local_check", parameter_overrides=[Parameter("use_sim_time", value=True)])
        self.speed = speed
        self.command = self.create_publisher(Twist, "/cmd_vel", 1)
        self.create_subscription(Odometry, "/odom/ground_truth", self.observe, qos_profile_sensor_data)
        self.create_subscription(LowCmd, "/lowcmd_bare", self.check_policy, qos_profile_sensor_data)
        self.create_subscription(LowCmd, "/lowcmd", self.check_output, qos_profile_sensor_data)
        self.first = None
        self.heading = None
        self.origin = None
        self.bad_since = [None] * 3
        self.last_command = None
        self.last_output_time = None
        self.max_command_interval = 0.0
        self.output_count = 0
        self.policy_count = 0
        self.scored_output_count = 0
        self.scored = 0.0
        self.distance = 0.0
        self.lateral = 0.0
        self.fallen = False
        self.min_height = float("inf")
        self.errors = set()

    def check_policy(self, _message):
        self.policy_count += 1

    def check_output(self, msg):
        command = np.array([(m.q, m.dq, m.tau, m.kp, m.kd) for m in msg.motor_cmd[:12]])
        q = command[:, 0]
        if not np.isfinite(command).all():
            self.errors.add("nonfinite command")
        if np.any(q < JOINT_MIN) or np.any(q > JOINT_MAX):
            self.errors.add("joint limit exceeded")
        # The simulation clock advances in ~10 ms increments, so two
        # received messages can share the same reported time.
        now = self.get_clock().now().nanoseconds * 1e-9
        if self.origin is not None:
            if self.last_output_time is not None:
                gap = now - self.last_output_time
                self.max_command_interval = max(self.max_command_interval, gap)
                if (gap <= 0.015 and self.last_command is not None
                        and np.max(np.abs(q - self.last_command)) > np.deg2rad(5) + 1e-6):
                    self.errors.add("step limit exceeded")
            self.scored_output_count += 1
        self.output_count += 1
        self.last_output_time = now
        self.last_command = q

    def observe(self, msg):
        stamp = msg.header.stamp
        now = stamp.sec + stamp.nanosec * 1e-9
        pos = msg.pose.pose.position
        o = msg.pose.pose.orientation
        xy = np.array([pos.x, pos.y])
        if self.first is None:
            if self.command.get_subscription_count() == 0 or min(self.output_count, self.policy_count) == 0:
                return
            self.first = now
            yaw = yaw_from_quat(np.array([o.w, o.x, o.y, o.z]))
            self.heading = np.array([np.cos(yaw), np.sin(yaw)])
        roll = np.arctan2(2 * (o.w * o.x + o.y * o.z), 1 - 2 * (o.x**2 + o.y**2))
        pitch = np.arcsin(np.clip(2 * (o.w * o.y - o.z * o.x), -1, 1))
        limits = (pos.z < 0.16, abs(roll) > np.deg2rad(50), abs(pitch) > np.deg2rad(50))
        self.min_height = min(self.min_height, pos.z)
        for i, bad in enumerate(limits):
            if not bad:
                self.bad_since[i] = None
            elif self.bad_since[i] is None:
                self.bad_since[i] = now
            elif now - self.bad_since[i] > 0.5:
                self.fallen = True
        if now - self.first < 1.5:
            return
        if self.origin is None:
            self.origin = xy
        self.scored = now - self.first - 1.5
        self.distance = float((xy - self.origin) @ self.heading)
        self.lateral = float((xy - self.origin) @ np.array([-self.heading[1], self.heading[0]]))

    def run(self, wall_limit):
        deadline = time.monotonic() + wall_limit
        next_velocity = 0.0
        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.05)
            wall = time.monotonic()
            if wall >= next_velocity:
                velocity = Twist()
                velocity.linear.x = self.speed if self.origin is not None else 0.0
                self.command.publish(velocity)
                next_velocity = wall + 0.05
            if self.distance >= 3 or self.scored >= 90 or self.fallen:
                break
        return {"success": self.distance >= 3 and not self.fallen and not self.errors,
                "fallen": self.fallen, "distance_m": round(self.distance, 4),
                "lateral_m": round(self.lateral, 4), "scored_seconds": round(self.scored, 3),
                "min_height_m": round(self.min_height, 4) if self.first is not None else None,
                "bare_commands": self.policy_count,
                "safe_commands": self.scored_output_count,
                "mean_command_hz": round(self.scored_output_count / self.scored, 2) if self.scored else 0,
                "max_command_interval_s": round(self.max_command_interval, 4),
                "errors": sorted(self.errors)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--speed", type=float, default=0.6)
    parser.add_argument("--wall-limit", type=float, default=150)
    args = parser.parse_args()
    rclpy.init()
    node = EpisodeCheck(args.speed)
    try:
        result = node.run(args.wall_limit)
        print(json.dumps(result), flush=True)
        return 0 if result["success"] else 1
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
