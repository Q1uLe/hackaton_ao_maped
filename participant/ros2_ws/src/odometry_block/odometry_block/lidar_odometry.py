"""Estimate planar LiDAR motion from consecutive scans."""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


_MIN_POINTS = 12
_MIN_NORMAL_NEIGHBORS = 7
_MAX_NORMAL_NEIGHBORS = 17
_NORMAL_RADIUS = 0.45  # m, limit the support of each local line
_SEARCH_RADIUS = 0.45  # m, initial distance between corresponding points
_INLIER_DISTANCE = 0.15  # m, final distance between corresponding points
_LINE_TOLERANCE = 0.06  # m, final distance perpendicular to the surface
_MIN_OVERLAP = 0.50
_MIN_SINGULAR_RATIO = 0.10


def convert_ranges_to_coords(
    ranges: list[float],
    angle_min: float,
    angle_max: float,
    angle_increment: float,
    *,
    range_min: float = 0.0,
    range_max: float = np.inf,
) -> np.ndarray:
    """Return the finite, in-range LaserScan returns as an (N, 2) array.

    Angles follow the original beam indices. The number of measurements and
    the increment define the sampled angles; angle_max is kept for callers
    that pass the full angular metadata.
    """
    distances = np.asarray(ranges, dtype=float)
    metadata = (angle_min, angle_max, angle_increment)
    if distances.ndim != 1 or not np.isfinite(metadata).all():
        return np.empty((0, 2))
    valid = (
        np.isfinite(distances) & (distances > 0.0)
        & (distances >= range_min) & (distances <= range_max)
    )
    angles = angle_min + np.flatnonzero(valid) * angle_increment
    distances = distances[valid]
    direction = np.column_stack((np.cos(angles), np.sin(angles)))
    return distances[:, None] * direction


def _clean_points(points):
    """Remove invalid and duplicate rows, and sort the points."""
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2:
        return np.empty((0, 2))
    return np.unique(points[np.isfinite(points).all(axis=1)], axis=0)


def _local_surfaces(points, tree):
    """Fit a local line where nearby points agree on a direction."""
    count = min(_MAX_NORMAL_NEIGHBORS, len(points))
    distance, indices = tree.query(points, k=count)
    within_radius = distance <= _NORMAL_RADIUS
    support = within_radius.sum(axis=1)
    neighbors = points[indices]
    # Dense returns from one wall need a wider fit so noise does not create
    # artificial differences in the normals used to check observability.
    mask = within_radius[:, :, None]
    center = (neighbors * mask).sum(axis=1) / support[:, None]
    centered = (neighbors - center[:, None, :]) * mask
    values, axes = np.linalg.eigh(centered.transpose(0, 2, 1) @ centered)
    usable = (
        (support >= _MIN_NORMAL_NEIGHBORS)
        & (values[:, 1] > 1e-8)
        & (values[:, 0] <= 0.08 * values[:, 1])
    )
    return usable, axes[:, :, 0]


def _rotation(yaw):
    """Return the active 2D rotation for yaw measured in radians."""
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s], [s, c]])


def _linearized_fit(rotated, normal, residual):
    """Solve for a weighted translation and yaw update, if observable."""
    yaw_derivative = (
        -normal[:, 0] * rotated[:, 1] + normal[:, 1] * rotated[:, 0]
    )
    # Express the yaw column in units comparable to the translation columns.
    lever = max(0.1, np.sqrt(np.mean(np.sum(rotated**2, axis=1))))
    jacobian = np.column_stack((normal, yaw_derivative / lever))
    weight = np.sqrt(0.03 / np.maximum(0.03, np.abs(residual)))
    step, _, _, singular = np.linalg.lstsq(
        jacobian * weight[:, None], -residual * weight, rcond=None,
    )
    if singular[-1] < _MIN_SINGULAR_RATIO * singular[0]:
        return None
    step[2] /= lever
    return step


def estimate_delta(
    prev_points: np.ndarray, curr_points: np.ndarray,
) -> tuple[float, float, float] | None:
    """Return (dx, dy, dyaw) mapping current points into the previous frame.

    This gives the motion of the LiDAR in its previous axes. Angles are in
    radians and positive counterclockwise. Return None when there is too
    little overlap or the surfaces do not constrain all three components.
    The identity initialization assumes small motion between scans.
    """
    previous, current = _clean_points(prev_points), _clean_points(curr_points)
    if min(len(previous), len(current)) < _MIN_POINTS:
        return None

    tree = cKDTree(previous)
    usable, normals = _local_surfaces(previous, tree)
    if np.count_nonzero(usable) < _MIN_POINTS:
        return None
    translation = np.zeros(2)
    yaw = 0.0

    for _ in range(30):
        rotated = current @ _rotation(yaw).T
        moved = rotated + translation
        distance, index = tree.query(moved)
        matched = (distance <= _SEARCH_RADIUS) & usable[index]
        if np.count_nonzero(matched) < _MIN_POINTS:
            return None
        normal = normals[index[matched]]
        displacement = moved[matched] - previous[index[matched]]
        residual = np.sum(normal * displacement, axis=1)
        step = _linearized_fit(rotated[matched], normal, residual)
        if step is None:
            return None
        translation += step[:2]
        yaw += step[2]
        if np.linalg.norm(step) < 2e-4:
            break
    else:
        return None

    # Check final correspondences and overlap in both directions.
    rotated = current @ _rotation(yaw).T
    moved = rotated + translation
    distance, index = tree.query(moved)
    delta = moved - previous[index]
    residual = np.sum(normals[index] * delta, axis=1)
    near = distance <= _INLIER_DISTANCE
    inliers = near & usable[index] & (np.abs(residual) <= _LINE_TOLERANCE)
    min_inliers = max(_MIN_POINTS, _MIN_OVERLAP * len(current))
    if np.count_nonzero(inliers) < min_inliers:
        return None
    reverse_distance, _ = cKDTree(moved).query(previous)
    if np.mean(reverse_distance <= _INLIER_DISTANCE) < _MIN_OVERLAP:
        return None
    final_step = _linearized_fit(
        rotated[inliers], normals[index[inliers]], residual[inliers],
    )
    if final_step is None:
        return None

    yaw = np.arctan2(np.sin(yaw), np.cos(yaw))
    return float(translation[0]), float(translation[1]), float(yaw)
