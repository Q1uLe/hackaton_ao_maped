import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu, LaserScan
from nav_msgs.msg import Odometry
from tf2_msgs.msg import TFMessage
from .lidar_odometry import estimate_delta, convert_ranges_to_coords


class OdometryNode(Node):
    def __init__(self):
        super().__init__('odometry_node')
        self.lidar_prev_points = None
        self.lidar_curr_points = None

        self.lidar_listener = self.create_subscription(
            LaserScan,
            '/scan',
            self.lidar_callback,
            10
        )

        self.imu_listener = self.create_subscription(
            Imu,
            '/imu/data',
            self.imu_callback,
            10
        )

        self.odom_publisher = self.create_publisher(
            Odometry,
            '/odom',
            10
        )

    def lidar_callback(self, msg: LaserScan):
        """
        Handle LiDAR's data
        """
        if self.lidar_prev_points is None:
            self.lidar_prev_points = convert_ranges_to_coords(msg.ranges, msg.angle_min, msg.angle_max, msg.angle_increment)
            return
        if self.lidar_curr_points is None:
            self.lidar_curr_points = msg.ranges
            return
        pass

    def imu_callback(self, msg):
        """
        Handle IMU's data
        """
        pass


def main():
    pass


if __name__=="__main__":
    rclpy.init()
    node = OdometryNode()
    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()

