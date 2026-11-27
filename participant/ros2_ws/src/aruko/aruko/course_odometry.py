"""Adapt the shared scan odometry to the Course start and Nav2 velocity feedback."""

import math

from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry
from odometry_block.lidar_odometry import convert_ranges_to_coords
from odometry_block.odometry import OdometryNode
import rclpy
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import LaserScan
from tf2_py import TransformException

from .scan_motion import course_delta


class CourseOdometry(OdometryNode):
    """Express the lidar trajectory in the initial trunk frame."""

    def __init__(self):
        super().__init__()
        self.destroy_subscription(self.lidar_listener)
        self.lidar_listener = self.create_subscription(
            LaserScan, '/scan', self.lidar_callback, qos_profile_sensor_data)

    def lidar_callback(self, msg):
        """Rebase on each new scan, including after a rejected match."""
        points = convert_ranges_to_coords(
            msg.ranges, msg.angle_min, msg.angle_max, msg.angle_increment,
            range_min=msg.range_min, range_max=msg.range_max)
        if len(points) < 12:
            return
        stamp = msg.header.stamp
        t_ns = stamp.sec * 1_000_000_000 + stamp.nanosec
        previous = self.last_scan
        if previous is not None and t_ns <= previous[1]:
            return
        x, y, yaw, _ = self.curr_pose
        self.last_scan = points, t_ns
        self.curr_pose = x, y, yaw, t_ns
        self.last_movement = None
        if previous is None or self.scan_frame != msg.header.frame_id:
            self.scan_frame = msg.header.frame_id
            return

        dt_ns = t_ns - previous[1]
        delta = course_delta(previous[0], points, dt_ns * 1e-9)
        if delta is None:
            return
        dx, dy, dyaw = delta
        c, s = math.cos(yaw), math.sin(yaw)
        move_x = c * dx - s * dy
        move_y = s * dx + c * dy
        self.curr_pose = x + move_x, y + move_y, yaw + dyaw, t_ns
        self.last_movement = move_x, move_y, dyaw, dt_ns

    def publish_pose(self):
        """Publish the trunk pose and body-frame velocity with matching TF."""
        if self.scan_frame is None:
            return
        try:
            mount = self.tf_buffer.lookup_transform(
                'trunk', self.scan_frame, Time()).transform
        except TransformException:
            return

        q = mount.rotation
        mount_yaw = math.atan2(
            2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        now = self.get_clock().now()
        lidar_x, lidar_y, yaw = self.estimate_pose_at(now.nanoseconds)
        cm, sm = math.cos(mount_yaw), math.sin(mount_yaw)
        c, s = math.cos(yaw), math.sin(yaw)
        mx, my = mount.translation.x, mount.translation.y

        odom = Odometry()
        odom.header.stamp = now.to_msg()
        odom.header.frame_id = 'odom'
        odom.child_frame_id = 'trunk'
        pose = odom.pose.pose
        pose.position.x = mx + cm * lidar_x - sm * lidar_y - (c * mx - s * my)
        pose.position.y = my + sm * lidar_x + cm * lidar_y - (s * mx + c * my)
        pose.orientation.z = math.sin(yaw / 2.0)
        pose.orientation.w = math.cos(yaw / 2.0)

        if self.last_movement and now.nanoseconds - self.curr_pose[3] < 350_000_000:
            dx, dy, dyaw, elapsed_ns = self.last_movement
            dt = elapsed_ns * 1e-9
            relative_yaw = yaw - mount_yaw
            cr, sr = math.cos(relative_yaw), math.sin(relative_yaw)
            wz = dyaw / dt
            twist = odom.twist.twist
            twist.linear.x = (cr * dx + sr * dy) / dt + wz * my
            twist.linear.y = (-sr * dx + cr * dy) / dt - wz * mx
            twist.angular.z = wz

        transform = TransformStamped(header=odom.header, child_frame_id='trunk')
        transform.transform.translation.x = pose.position.x
        transform.transform.translation.y = pose.position.y
        transform.transform.rotation = pose.orientation
        self.odom_publisher.publish(odom)
        self.tf_publisher.sendTransform(transform)


def main(args=None):
    """Run the T5 odometry adapter."""
    rclpy.init(args=args)
    node = CourseOdometry()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
