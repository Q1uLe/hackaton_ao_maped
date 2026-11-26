import rclpy
import math
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import Imu, LaserScan
from nav_msgs.msg import Odometry
from tf2_msgs.msg import TFMessage
from .lidar_odometry import estimate_delta, convert_ranges_to_coords
from tf2_ros import Buffer, TransformBroadcaster, TransformListener
from tf2_py import TransformException
from geometry_msgs.msg import TransformStamped

class OdometryNode(Node):
    def __init__(self):
        super().__init__('odometry_node')
        self.last_scan = None
        self.curr_pose = (0, 0, 0, 0) # dx, dy, dyaw, t_ns
        self.last_movement = None
        self.scan_frame = None

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
        self.timer = self.create_timer(1/60, self.publish_pose)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.tf_publisher = TransformBroadcaster(self)

    def lidar_callback(self, msg: LaserScan):
        """
        Handle LiDAR's data
        """
        points = convert_ranges_to_coords(
            msg.ranges.tolist(),
            msg.angle_min,
            msg.angle_max,
            msg.angle_increment,
            range_min=msg.range_min,
            range_max=msg.range_max
        )
        if len(points) < 12:
            return
        
        stamp = msg.header.stamp
        t_ns = stamp.sec * 1000000000 + stamp.nanosec
        prev = self.last_scan

        if prev is None:
            self.scan_frame = msg.header.frame_id
            self.last_scan = points, t_ns
            self.curr_pose = (0, 0, 0, t_ns)
            return

        prev_pts, prev_t_ns = prev
        dt_ns = t_ns - prev_t_ns
        if dt_ns <= 0: return

        delta = estimate_delta(prev_pts, points)

        if delta is None:
            return

        dx, dy, dyaw = delta
        x, y, yaw, t = self.curr_pose
        c, s = math.cos(yaw), math.sin(yaw)

        self.curr_pose = (
            x + c * dx - s * dy,
            y + s * dx + c * dy,
            yaw + dyaw,
            t_ns
        )
        self.last_scan = points, t_ns

        self.last_movement = (
            c * dx - s * dy,
            s * dx + c * dy,
            dyaw,
            dt_ns
        )

    def imu_callback(self, msg: Imu):
        """
        Handle IMU's data
        """
        pass

    def estimate_pose_at(self, t_ns):
        """
        Функция передаёт положение последнего скана, либо предсказывает по последним двум
        """
        x, y, yaw, scan_t_ns = self.curr_pose
        if self.last_movement is None:
            return x, y, yaw
        dx, dy, dyaw, dt_ns = self.last_movement
        alpha = max(0, min(1, (t_ns - scan_t_ns) / dt_ns))
        return x + alpha * dx, y + alpha * dy, yaw + alpha * dyaw
        

    def publish_pose(self):
        if self.scan_frame is None:
            return

        try:
            mount = self.tf_buffer.lookup_transform(
                "trunk", self.scan_frame, Time()
            ).transform
        except TransformException:
            return

        now = self.get_clock().now()
        t_ns = now.nanoseconds
        lidar_x, lidar_y, lidar_yaw = self.estimate_pose_at(t_ns)

        q = mount.rotation
        offset_yaw = math.atan2(
            2 * (q.w * q.z + q.x * q.y),
            1 - 2 * (q.y ** 2 + q.z ** 2)
        )
        yaw = lidar_yaw - offset_yaw
        c, s = math.cos(yaw), math.sin(yaw)
        offset_x = mount.translation.x
        offset_y = mount.translation.y
        x = lidar_x - (c * offset_x - s * offset_y)
        y = lidar_y - (s * offset_x + c * offset_y)

        odom = Odometry()
        odom.header.stamp = now.to_msg()
        odom.header.frame_id = "odom"
        odom.child_frame_id = "trunk"

        pose = odom.pose.pose
        pose.position.x = x
        pose.position.y = y
        pose.position.z = 0.0
        pose.orientation.z = math.sin(yaw/2)
        pose.orientation.w = math.cos(yaw / 2)

        transform = TransformStamped()
        transform.header = odom.header
        transform.child_frame_id = odom.child_frame_id
        transform.transform.translation.x = pose.position.x
        transform.transform.translation.y = pose.position.y
        transform.transform.translation.z = pose.position.z
        transform.transform.rotation = pose.orientation

        self.odom_publisher.publish(odom)
        self.tf_publisher.sendTransform(transform)


def main():
    rclpy.init()
    node = OdometryNode()
    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__=="__main__":
    main()
