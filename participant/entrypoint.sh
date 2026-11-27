#!/bin/bash
if [[ "$1" == "ping" ]]; then
 echo '{"protocol_version": "1.0", "supported_tasks": ["T2", "T3", "T4"], "self_trained_rl": true, "preferred_speed_mps": 0.6}'
elif [[ "$1" == "T2" ]]; then
    source /opt/ros/humble/setup.bash
    source /opt/participant/ros2_ws/install/setup.bash
    exec ros2 run safety_block state_listener --ros-args -p use_sim_time:=true
elif [[ "$1" == "T3" ]]; then
    source /opt/ros/humble/setup.bash
    source /opt/participant/ros2_ws/install/setup.bash
    cd /opt/participant/t3 || exit 1
    export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
    exec python3 -m go1_t3.ros_nodes "${@:2}"
elif [[ "$1" == "T4" ]]; then
    source /opt/ros/humble/setup.bash
    source /opt/participant/ros2_ws/install/setup.bash
    exec ros2 run odometry_block odometry --ros-args -p use_sim_time:=true
elif [[ "$1" == "T5" ]]; then
    source /opt/ros/humble/setup.bash
    source /opt/participant/ros2_ws/install/setup.bash
    exec ros2 launch aruko t5.launch.py
fi

