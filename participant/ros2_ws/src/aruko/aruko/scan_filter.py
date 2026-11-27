"""Remove returns inside the robot's own envelope before mapping and planning."""

import math

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import LaserScan
from tf2_ros import Buffer, TransformException, TransformListener


class ScanFilter(Node):
    """Apply a planar self mask using the laser mount relative to trunk."""

    def __init__(self):
        super().__init__('course_scan_filter')
        self.tf = Buffer()
        self.listener = TransformListener(self.tf, self)
        self.publisher = self.create_publisher(LaserScan, '/course/scan', qos_profile_sensor_data)
        self.subscription = self.create_subscription(
            LaserScan, '/scan', self.filter_scan, qos_profile_sensor_data)

    def filter_scan(self, scan):
        """Discard self returns without tracing free space through the robot."""
        try:
            mount = self.tf.lookup_transform('trunk', scan.header.frame_id, Time()).transform
        except TransformException:
            return
        p, q = mount.translation, mount.rotation
        yaw = math.atan2(2*(q.w*q.z + q.x*q.y), 1-2*(q.y*q.y + q.z*q.z))
        for i, distance in enumerate(scan.ranges):
            if not scan.range_min <= distance <= scan.range_max:
                continue
            angle = yaw + scan.angle_min + i * scan.angle_increment
            x = p.x + distance * math.cos(angle)
            y = p.y + distance * math.sin(angle)
            if abs(x) <= 0.40 and abs(y) <= 0.29:
                scan.ranges[i] = math.nan
        self.publisher.publish(scan)


def main(args=None):
    """Run the Course self-filter."""
    rclpy.init(args=args)
    node = ScanFilter()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
