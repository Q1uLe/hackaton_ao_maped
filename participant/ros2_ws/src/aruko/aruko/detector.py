"""Publish IDs detected in the current RGB frame from any T5 camera."""

import json

import cv2
from cv_bridge import CvBridge, CvBridgeError
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from sensor_msgs.msg import Image
from std_msgs.msg import String


CAMERAS = ('camera', 'camera_chin', 'camera_left', 'camera_right', 'camera_rearDown')


class Detector(Node):
    """Decode DICT_4X4_50 with both Humble and newer OpenCV APIs."""

    def __init__(self):
        super().__init__('aruco_detector')
        aruco = cv2.aruco
        dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
        if hasattr(aruco, 'ArucoDetector'):
            detector = aruco.ArucoDetector(dictionary, aruco.DetectorParameters())
            self.detect = detector.detectMarkers
        else:
            parameters = aruco.DetectorParameters_create()
            self.detect = lambda gray: aruco.detectMarkers(
                gray, dictionary, parameters=parameters)
        self.bridge = CvBridge()
        self.detections = self.create_publisher(String, '/aruco/detections', 10)
        self.cameras = [
            self.create_subscription(Image, f'/{camera}/image_raw', self.on_image,
                                     QoSProfile(depth=1))
            for camera in CAMERAS
        ]

    def on_image(self, image):
        """Publish observed IDs without retaining detections across frames."""
        try:
            gray = self.bridge.imgmsg_to_cv2(image, desired_encoding='mono8')
            _, ids, _ = self.detect(gray)
        except (CvBridgeError, cv2.error) as error:
            self.get_logger().warning(str(error), throttle_duration_sec=5.0)
            return
        if ids is not None and ids.size:
            current = sorted({int(marker) for marker in ids.ravel()})
            self.detections.publish(String(data=json.dumps({'ids': current})))


def main(args=None):
    """Run the detector."""
    rclpy.init(args=args)
    node = Detector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
