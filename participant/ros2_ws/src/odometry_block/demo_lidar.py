"""Check synthetic scenarios and optionally measure accuracy on a ROS 2 bag."""

import argparse
import math
from pathlib import Path
from time import perf_counter

import numpy as np

from odometry_block.lidar_odometry import (
    convert_ranges_to_coords,
    estimate_delta,
)


def rotate(vector, yaw):
    """Rotate one planar vector by yaw."""
    x, y = vector
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array((c * x - s * y, s * x + c * y))


def wrap(angle):
    """Wrap an angle or an array of angles into [-pi, pi]."""
    return np.arctan2(np.sin(angle), np.cos(angle))


def scan_scene(segments, pose):
    """Cast rays and return the nearest visible hit on each ray."""
    beam_angles = np.linspace(-np.pi, np.pi, 720, endpoint=False)
    return_angles = beam_angles + pose[2]
    rays = np.column_stack((np.cos(return_angles), np.sin(return_angles)))
    ranges = np.full(len(rays), np.inf)
    for start, end in np.asarray(segments):
        offset = start - pose[:2]
        edge = end - start
        # Solve pose_xy + distance * ray = start + fraction * edge.
        denominator = rays[:, 0] * edge[1] - rays[:, 1] * edge[0]
        eligible = np.abs(denominator) > 1e-9
        distance = (
            (offset[0] * edge[1] - offset[1] * edge[0]) / denominator[eligible]
        )
        fraction = (
            offset[0] * rays[eligible, 1] - offset[1] * rays[eligible, 0]
        ) / denominator[eligible]
        hit = (distance > 0) & (fraction >= 0) & (fraction <= 1)
        beam = np.flatnonzero(eligible)[hit]
        ranges[beam] = np.minimum(ranges[beam], distance[hit])
    return convert_ranges_to_coords(
        ranges, beam_angles[0], beam_angles[-1],
        beam_angles[1] - beam_angles[0],
        range_min=0.12, range_max=20.0,
    )


def run_synthetic():
    """Print expected and observed results and report whether they pass."""
    room = [
        ((-2.0, -1.8), (3.1, -1.8)),
        ((3.1, -1.8), (3.1, 2.6)),
        ((3.1, 2.6), (-2.0, 2.6)),
        ((-2.0, 2.6), (-2.0, -1.8)),
        ((0.8, 0.8), (1.6, 0.8)),
        ((1.6, 0.8), (1.6, 1.3)),
        ((1.6, 1.3), (0.8, 1.3)),
        ((0.8, 1.3), (0.8, 0.8)),
        ((-1.1, -1.0), (-0.6, -0.6)),
        ((-0.6, -0.6), (-1.2, -0.5)),
        ((-1.2, -0.5), (-1.1, -1.0)),
    ]
    wall = [((2.0, -2.0), (2.0, 2.0))]
    near_wall = [((0.2, -2.0), (0.2, 2.0))]
    origin = (0.0, 0.0, 0.0)
    forward = (0.20, 0.0, 0.0)
    turn = (0.0, 0.0, math.radians(10))
    reference = scan_scene(room, origin)
    rng = np.random.default_rng(4)
    sideways = (0.0, 0.20, 0.0)
    near_previous = scan_scene(near_wall, origin)
    near_current = scan_scene(near_wall, sideways)
    wall_noise = np.random.default_rng(11)
    near_previous += wall_noise.normal(0.0, 0.003, near_previous.shape)
    near_current += wall_noise.normal(0.0, 0.003, near_current.shape)
    cases = [
        ('Standing still', origin, reference, scan_scene(room, origin)),
        ('Forward 20 cm', forward, reference, scan_scene(room, forward)),
        ('CCW turn 10 deg', turn, reference, scan_scene(room, turn)),
        ('Empty scan', None, reference, np.empty((0, 2))),
        ('Single wall', None, scan_scene(wall, origin),
         scan_scene(wall, sideways)),
        ('Noisy near wall', None, near_previous, near_current),
    ]
    print('Synthetic cases (dx [m], dy [m], dyaw [rad])')
    print('Tolerances: 0.02 m for XY error, 2 deg for yaw error.')
    all_passed = True
    for name, expected, previous, current in cases:
        actual = estimate_delta(
            rng.permutation(previous), rng.permutation(current),
        )
        if expected is None:
            passed = actual is None
        else:
            passed = (
                actual is not None
                and np.linalg.norm(np.subtract(actual[:2], expected[:2]))
                <= 0.02
                and abs(wrap(actual[2] - expected[2])) <= math.radians(2)
            )
        all_passed &= passed
        actual_text = None if actual is None else np.round(actual, 4).tolist()
        expected_text = (
            None if expected is None else np.round(expected, 4).tolist()
        )
        result = 'PASS' if passed else 'FAIL'
        print(f'  {result}  {name:16} expected={expected_text}'
              f'  actual={actual_text}')
    return all_passed


def load_bag(path):
    """Deserialize the three offline validation topics from a SQLite rosbag."""
    import sqlite3

    from nav_msgs.msg import Odometry
    from rclpy.serialization import deserialize_message
    from sensor_msgs.msg import LaserScan
    from tf2_msgs.msg import TFMessage

    types = {
        '/scan': LaserScan, '/odom/ground_truth': Odometry,
        '/tf_static': TFMessage,
    }
    if not path.is_file():
        raise ValueError(f'Bag file does not exist: {path}')
    series = {topic: [] for topic in types}
    with sqlite3.connect(path.resolve().as_uri() + '?mode=ro', uri=True) as db:
        rows = db.execute("""
            SELECT topics.name, messages.data FROM messages
            JOIN topics ON topics.id = messages.topic_id
            WHERE topics.name IN (?, ?, ?)
            ORDER BY messages.timestamp, messages.id
        """, tuple(types))
        for topic, data in rows:
            series[topic].append(deserialize_message(data, types[topic]))
    return series


def stamp(message):
    """Read acquisition time from a ROS message header."""
    time = message.header.stamp
    return time.sec + time.nanosec * 1e-9


def xyz(vector):
    """Read a ROS three-dimensional vector."""
    return np.array([vector.x, vector.y, vector.z])


def xyzw(quaternion):
    """Read a ROS quaternion in SciPy order."""
    return [quaternion.x, quaternion.y, quaternion.z, quaternion.w]


def lidar_in_trunk(messages, frame):
    """Find this bag's direct static transform trunk -> scan frame."""
    from scipy.spatial.transform import Rotation

    for message in messages:
        for entry in message.transforms:
            frames = entry.header.frame_id, entry.child_frame_id
            if frames == ('trunk', frame):
                transform = entry.transform
                rotation = Rotation.from_quat(xyzw(transform.rotation))
                return xyz(transform.translation), rotation
    raise ValueError(f'Missing static transform trunk -> {frame}')


def sample_truth(messages, times, lidar_offset, lidar_rotation):
    """Interpolate trunk truth and obtain LiDAR truth at scan timestamps."""
    from scipy.spatial.transform import Rotation, Slerp

    truth_times = np.array([stamp(message) for message in messages])
    truth_times, indices = np.unique(truth_times, return_index=True)
    poses = [messages[i].pose.pose for i in indices]
    truth_xyz = np.array([xyz(pose.position) for pose in poses])
    truth_rotation = Rotation.from_quat(
        [xyzw(pose.orientation) for pose in poses],
    )
    position = np.column_stack([
        np.interp(times, truth_times, axis) for axis in truth_xyz.T
    ])
    rotation = Slerp(truth_times, truth_rotation)(times)
    trunk = np.column_stack((position[:, :2], rotation.as_euler('xyz')[:, 2]))
    lidar_position = position + rotation.apply(lidar_offset)
    lidar_yaw = (rotation * lidar_rotation).as_euler('xyz')[:, 2]
    lidar = np.column_stack((lidar_position[:, :2], lidar_yaw))
    return trunk, lidar


def relative(start, end):
    """Express a planar displacement in the starting pose's axes."""
    return np.r_[
        rotate(end[:2] - start[:2], -start[2]), wrap(end[2] - start[2]),
    ]


def meter_rpe(truth, estimated):
    """Compute a local T4-style RPE over consecutive one-meter segments."""
    intervals = np.linalg.norm(np.diff(truth[:, :2], axis=0), axis=1)
    distance = np.r_[0, np.cumsum(intervals)]
    ends = np.arange(0, distance[-1] + 1e-9, 1.0)
    if len(ends) < 2:
        return np.empty(0)
    # Select two endpoints for each full meter of accumulated true path.
    indices = np.searchsorted(distance, ends).clip(max=len(distance) - 1)
    errors = []
    for start, end in zip(indices[:-1], indices[1:]):
        actual = relative(estimated[start], estimated[end])
        expected = relative(truth[start], truth[end])
        length = distance[end] - distance[start]
        errors.append(np.linalg.norm(actual[:2] - expected[:2]) / length)
    return np.asarray(errors)


def summarize_rest(times, truth, estimated):
    """Report drift if the end of the bag contains 10 seconds of rest."""
    start_time = times[-1] - 10.0
    if start_time < times[0]:
        return '10 s drift: unavailable (recording too short)'
    start = np.searchsorted(times, start_time, side='left')
    truth_yaw = np.unwrap(truth[:, 2])
    truth_axes = [truth[:, 0], truth[:, 1], truth_yaw]
    truth_start = [np.interp(start_time, times, axis) for axis in truth_axes]
    end_truth = truth[start:] - truth_start
    if (np.max(np.linalg.norm(end_truth[:, :2], axis=1)) > 0.02
            or np.max(np.abs(wrap(end_truth[:, 2]))) > math.radians(1)):
        return '10 s drift: unavailable (last 10 s are not stationary)'
    axes = [estimated[:, 0], estimated[:, 1], np.unwrap(estimated[:, 2])]
    estimated_start = np.array([np.interp(start_time, times, a) for a in axes])
    drift = relative(estimated_start, estimated[-1])
    distance, angle = np.linalg.norm(drift[:2]), abs(np.degrees(drift[2]))
    return f'10 s drift: {distance:.3f} m, {angle:.2f} deg'


def match_scans(scans):
    """Estimate consecutive scan deltas without using reference poses."""
    results = []
    elapsed = []
    previous = None
    for scan in scans:
        started = perf_counter()
        points = convert_ranges_to_coords(
            scan.ranges, scan.angle_min, scan.angle_max, scan.angle_increment,
            range_min=scan.range_min, range_max=scan.range_max,
        )
        if previous is not None:
            results.append(estimate_delta(previous, points))
            elapsed.append(perf_counter() - started)
        previous = points
    return results, np.asarray(elapsed)


def run_bag(path):
    """Report relative errors, coverage and integrated behavior on one bag."""
    series = load_bag(path)
    truth = sorted(series['/odom/ground_truth'], key=stamp)
    scans = sorted(series['/scan'], key=stamp)
    if len(truth) < 2 or len(scans) < 2:
        raise ValueError('Need at least two scans and two reference poses')
    if any((msg.header.frame_id, msg.child_frame_id) != ('odom', 'trunk')
           for msg in truth):
        raise ValueError('Ground truth must describe odom -> trunk')
    frame = scans[0].header.frame_id
    if any(scan.header.frame_id != frame for scan in scans):
        raise ValueError('The scan frame changes within the recording')
    results, elapsed = match_scans(scans)

    # Reference poses determine only which estimates can be scored.
    scan_times = np.array([stamp(scan) for scan in scans])
    first = np.searchsorted(scan_times, stamp(truth[0]), side='left')
    last = np.searchsorted(scan_times, stamp(truth[-1]), side='right')
    times = scan_times[first:last]
    if len(times) < 2:
        raise ValueError('Scan and ground truth timestamps do not overlap')
    results = results[first:last - 1]
    offset, rotation = lidar_in_trunk(series['/tf_static'], frame)
    trunk_truth, lidar_truth = sample_truth(truth, times, offset, rotation)
    # Apply the planar part of trunk -> LiDAR when accumulating trunk poses.
    offset_xy = offset[:2]
    mount_yaw = rotation.as_euler('xyz')[2]

    position_errors = []
    yaw_errors = []
    trajectory = np.zeros((len(times), 3))
    for index, delta in enumerate(results, start=1):
        trajectory[index] = trajectory[index - 1]
        if delta is None:
            continue
        expected = relative(lidar_truth[index - 1], lidar_truth[index])
        error = np.linalg.norm(np.subtract(delta[:2], expected[:2]))
        position_errors.append(error)
        yaw_errors.append(abs(wrap(delta[2] - expected[2])))
        lever_shift = offset_xy - rotate(offset_xy, delta[2])
        dxdy = rotate(delta[:2], mount_yaw) + lever_shift
        trajectory[index, :2] += rotate(dxdy, trajectory[index - 1, 2])
        trajectory[index, 2] += delta[2]

    print(f'\nBag validation: {path.name} ({times[-1] - times[0]:.2f} s)')
    print(f'  Accepted pairs: {len(position_errors)}/{len(results)}'
          f' ({len(position_errors) / len(results):.1%})')
    p95_time = np.percentile(elapsed, 95) * 1000
    print(f'  Conversion + matching p95: {p95_time:.2f} ms')
    if not position_errors:
        return False
    position_mm = np.asarray(position_errors) * 1000
    position_med, position_p95 = np.percentile(position_mm, [50, 95])
    yaw_med, yaw_p95 = np.degrees(np.percentile(yaw_errors, [50, 95]))
    print('  LiDAR pair error, median / p95: '
          f'{position_med:.1f} / {position_p95:.1f} mm, '
          f'{yaw_med:.3f} / {yaw_p95:.3f} deg')
    errors = meter_rpe(trunk_truth, trajectory)
    if len(errors):
        print(f'  Local 1 m trunk RPE: {errors.mean():.2%} mean '
              f'({len(errors)} segments)')
    else:
        print('  Local 1 m trunk RPE: unavailable (path shorter than 1 m)')
    print(f'  {summarize_rest(times, trunk_truth, trajectory)}')
    if len(position_errors) < len(results):
        print('  For integrated metrics, pose is held on rejected pairs.')
    return True


def main():
    """Run the checks, with optional ROS 2 bag validation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--bag', type=Path, help='SQLite .db3 file; source ROS 2 Humble first',
    )
    args = parser.parse_args()
    passed = run_synthetic()
    if args.bag is not None:
        try:
            passed = run_bag(args.bag) and passed
        except (ImportError, OSError, ValueError) as error:
            parser.exit(2, f'Bag validation failed: {error}\n')
    return 0 if passed else 1


if __name__ == '__main__':
    raise SystemExit(main())
