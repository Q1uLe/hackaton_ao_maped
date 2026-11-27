"""Keep unknown cells beyond the scanned map available to Nav2."""

from array import array
from copy import deepcopy
import math

import numpy as np
from nav_msgs.msg import OccupancyGrid
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile


class PlanningMap(Node):
    """Pad the SLAM map without changing measured cells or their coordinates."""

    def __init__(self):
        super().__init__('planning_map')
        self.margin = self.declare_parameter('unknown_margin', 25.0).value
        qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.output = self.create_publisher(OccupancyGrid, '/planning_map', qos)
        self.input = self.create_subscription(OccupancyGrid, '/map', self.on_map, qos)

    def on_map(self, grid):
        """Publish a map large enough to plan beyond the first lidar scan."""
        info = grid.info
        if info.resolution <= 0.0 or not info.width or not info.height:
            return
        pad = math.ceil(max(0.0, self.margin) / info.resolution)
        cells = np.asarray(grid.data, dtype=np.int8).reshape(info.height, info.width)
        expanded = np.pad(cells, pad, constant_values=-1)
        result = OccupancyGrid(header=grid.header, info=deepcopy(info))
        result.info.height, result.info.width = expanded.shape
        # SLAM Toolbox publishes axis-aligned occupancy grids.
        result.info.origin.position.x -= pad * info.resolution
        result.info.origin.position.y -= pad * info.resolution
        result.data = array('b', expanded.tobytes())
        self.output.publish(result)


def main(args=None):
    """Run the planning map publisher."""
    rclpy.init(args=args)
    node = PlanningMap()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
