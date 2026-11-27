#!/usr/bin/env bash
set -euo pipefail

root=$(cd -- "$(dirname -- "$0")/.." && pwd)
participant=${1:-go1-participant:local}
simulator=${SIM_IMAGE:-registry.cookcloud.ru/go1-hackathon-remote-stage/go1-sim:v1.0}
episodes=${EPISODES:-1}
factor=${SIM_REALTIME_FACTOR:-1}
wall_limit=${WALL_LIMIT:-150}
speed=$(docker run --rm --network none "$participant" ping | python3 -c \
    'import json,sys; print(json.load(sys.stdin)["preferred_speed_mps"])')
tag=t3-check-$$
dds='<CycloneDDS><Domain id="any"><General><Interfaces><NetworkInterface name="lo" multicast="true"/></Interfaces><AllowMulticast>spdp</AllowMulticast><EnableMulticastLoopback>true</EnableMulticastLoopback></General><Discovery><ParticipantIndex>none</ParticipantIndex></Discovery></Domain></CycloneDDS>'
network=(--network host -e "ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-83}" -e RMW_IMPLEMENTATION=rmw_cyclonedds_cpp -e "CYCLONEDDS_URI=$dds")
sim_args=()
sim_command=(ros2 launch go1_simulation scenario.launch.py scenario:=lowlevel gui:=false cameras:=false)
if [[ "$factor" != 1 ]]; then
    sim_args=(-v "$root/tools/slow_simulator.py:/check/slow_simulator.py:ro")
    sim_command=(python3 /check/slow_simulator.py "$factor" --ros-args -p headless:=true)
fi

cleanup() {
    docker rm -f "${tag}-driver" "${tag}-participant" "${tag}-simulator" >/dev/null 2>&1 || true
}
trap cleanup EXIT

failures=0
for ((episode=1; episode<=episodes; episode++)); do
    docker run -d --name "${tag}-simulator" "${network[@]}" -e GO1_DDS_LOCAL=1 \
        "${sim_args[@]}" "$simulator" "${sim_command[@]}" >/dev/null
    docker run -d --name "${tag}-participant" "${network[@]}" "$participant" T3 >/dev/null
    result=0
    docker run --name "${tag}-driver" "${network[@]}" --entrypoint /bin/bash \
        -v "$root/tools/ros_check.py:/check/ros_check.py:ro" "$participant" -c \
        'source /opt/ros/humble/setup.bash; source /opt/participant/ros2_ws/install/setup.bash; export PYTHONPATH="/opt/participant/t3:${PYTHONPATH:-}"; exec python3 /check/ros_check.py --speed "$1" --wall-limit "$2"' \
        _ "$speed" "$wall_limit" || result=$?
    if ((result != 0)); then
        docker logs --tail 30 "${tag}-participant" >&2
        docker logs --tail 30 "${tag}-simulator" >&2
        failures=$((failures + 1))
    fi
    cleanup
done
printf '%s/%s successful ROS 2 episodes\n' "$((episodes - failures))" "$episodes" >&2
((failures == 0))
