#!/bin/bash
if [[ "$1" == "ping" ]]; then
	echo '{"protocol_version": "1.0", "supported_tasks": ["T2", "T4"], "self_trained_rl": false, "preferred_speed_mps": null}'
elif [[ "$1" == "T2" ]]; then
    source /opt/ros/humble/setup.bash
    source /opt/participant/ros2_ws/install/setup.bash
    exec ros2 run safety_block state_listener --ros-args -p use_sim_time:=true
elif [[ "$1" == "T4" ]]; then
    source /opt/ros/humble/setup.bash
    source /opt/participant/ros2_ws/install/setup.bash
    exec ros2 run odometry_block odometry --ros-args -p use_sim_time:=true
fi

