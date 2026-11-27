"""Reject scan matches outside the motion range used for Course."""

import math

from odometry_block.lidar_odometry import estimate_delta


def course_delta(previous, current, dt):
    """Estimate motion between consecutive, sufficiently close lidar scans."""
    if not 0.0 < dt <= 0.25:
        return None
    delta = estimate_delta(previous, current)
    if delta is None or not all(math.isfinite(value) for value in delta):
        return None
    dx, dy, dyaw = delta
    if math.hypot(dx, dy) > 0.025 + 0.8 * dt or abs(dyaw) > 0.04 + 1.6 * dt:
        return None
    return delta
