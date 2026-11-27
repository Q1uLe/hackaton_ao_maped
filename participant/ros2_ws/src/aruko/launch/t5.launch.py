"""Start the T5 solution against an already running Course simulator."""

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node
from pathlib import Path


def generate_launch_description():
    """Launch odometry, online mapping, navigation and marker recognition."""
    config = Path(get_package_share_directory('aruko')) / 'config'
    clock = {'use_sim_time': True}
    nav = [str(config / 'nav2.yaml')]
    managed = ['planner_server', 'controller_server', 'behavior_server', 'bt_navigator']
    return LaunchDescription([
        Node(package='aruko', executable='scan_filter', parameters=[clock], output='screen'),
        Node(package='aruko', executable='course_odometry', parameters=[clock],
             remappings=[('/scan', '/course/scan')],
             output='screen'),
        Node(package='slam_toolbox', executable='async_slam_toolbox_node',
             name='slam_toolbox', parameters=[str(config / 'slam.yaml')], output='screen'),
        Node(package='aruko', executable='planning_map', parameters=[clock], output='screen'),
        Node(package='aruko', executable='detector', parameters=[clock], output='screen'),
        Node(package='nav2_planner', executable='planner_server',
             name='planner_server', parameters=nav, output='screen'),
        Node(package='nav2_controller', executable='controller_server',
             name='controller_server', parameters=nav, output='screen'),
        Node(package='nav2_behaviors', executable='behavior_server',
             name='behavior_server', parameters=nav, output='screen'),
        Node(package='nav2_bt_navigator', executable='bt_navigator', name='bt_navigator',
             remappings=[('/goal_pose', '/aruco/rviz_goal')],
             parameters=nav + [{
                 'default_nav_to_pose_bt_xml': str(config / 'navigate.xml'),
             }], output='screen'),
        Node(package='nav2_lifecycle_manager', executable='lifecycle_manager',
             name='lifecycle_manager_navigation', parameters=[
                 clock, {'autostart': True, 'node_names': managed},
             ], output='screen'),
    ])
