from math import isfinite, radians
import rclpy
from rclpy.node import Node
from unitree_go.msg import LowState, LowCmd
from . import limits
import numpy as np
from .collision_checker import CollisionChecker
from copy import deepcopy


class StateListener(Node):
    def __init__(self):
        super().__init__('state_listener')
        self.q_current = None # Telemetry
        self.q_target = None # Actual command
        self.q_last_safe = None # Last safe command
        self.last_safe_cmd = None
        self.last_cmd_time = None
        
        # Listen to robot's telemetry
        self.lowstate_subscription = self.create_subscription(
            LowState,
            '/lowstate',
            self.lowstate_callback,
            10
        )
        
        # Listen input commands (might be safe and unsafe)
        self.lowcmd_bare_subscription = self.create_subscription(
            LowCmd,
            '/lowcmd_bare', self.lowcmd_bare_callback,
            10
        )
      
        # Publish safe checked commands
        self.lowcmd_publisher = self.create_publisher(
            LowCmd,
            '/lowcmd',
            10
        )

        self.max_delta = radians(5)
        self.collision_checker = CollisionChecker(
            '/opt/participant/models/scene.xml',
            '/opt/participant/models/t2_collision_pairs.csv'
        )

        self.timer = self.create_timer(1/60, self.publish_lowcmd)

    def lowstate_callback(self, msg):
        self.q_current = np.array([
            msg.motor_state[i].q for i in range(12)
        ])

    def lowcmd_bare_callback(self, msg):
        self.last_cmd_time = self.get_clock().now()
        if not self.check_finite(msg.motor_cmd):
            self.get_logger().warning('Incorrect angles')
            return
        self.q_target = np.array([
            msg.motor_cmd[i].q for i in range(12)
        ])
        if not self.check_bound(self.q_target):
            self.get_logger().warning('Joint limit exceeded')
            return
        if not self.check_angle_increment(self.q_target):
            self.get_logger().warning('Too big angle increment')
            return
        if not self.check_collision():
            self.get_logger().warning('Collision detected!')
            return

        self.q_last_safe = self.q_target
        self.last_safe_cmd = msg
        self.publish_lowcmd()

    @staticmethod
    def check_bound(q_: np.ndarray) -> bool:
        for (q_i, q_i_bounds) in zip(q_, limits.limits.values()):
            if not q_i_bounds[0] <= q_i <= q_i_bounds[1]:
               return False
        return True

    @staticmethod
    def check_finite(commands: list) -> bool:
        return all(
            map(
                lambda x: all(
                    map(
                        isfinite,
                        [
                            x.q, x.dq, x.tau, x.kp, x.kd
                        ]
                    )
                ), 
                commands
            )
        )
    
    def check_angle_increment(self, q_) -> bool:
        if self.q_last_safe is None: return True
        return all(
            map(
                lambda x: abs(x[0] - x[1]) <= self.max_delta,
                zip(q_, self.q_last_safe)
            )
        )

    def check_collision(self):
        if self.q_target is None:
            return False
        if self.q_last_safe is None:
            if self.q_current is None:
                return False
            if not all(map(
                    isfinite,
                    self.q_current
                )):
                return False
            return self.collision_checker.check_process(
                self.q_current,
                self.q_target
            )
        return self.collision_checker.check_process(
            self.q_last_safe,
            self.q_target
        )

    def publish_lowcmd(self):
        if self.last_cmd_time is None or self.last_safe_cmd is None:
            return
        elapsed = self.get_clock().now() - self.last_cmd_time
        elapsed_sec = elapsed.nanoseconds / 10 ** 9
        if elapsed_sec < 0.5:
            self.lowcmd_publisher.publish(self.last_safe_cmd)
        else:
            cmd_cpy = deepcopy(self.last_safe_cmd)
            for i in range(12):
                cmd_cpy.motor_cmd[i].dq = 0.0
                cmd_cpy.motor_cmd[i].tau = 0.0
                cmd_cpy.motor_cmd[i].kd = 2.0
                cmd_cpy.motor_cmd[i].kp = 40.0
                cmd_cpy.motor_cmd[i].mode = 10
            self.lowcmd_publisher.publish(cmd_cpy)


def main():
    rclpy.init()
    node = StateListener()
    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__=='__main__':
    main()
