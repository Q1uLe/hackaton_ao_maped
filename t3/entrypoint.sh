#!/usr/bin/env bash
set -e
cd /opt/go1-t3
case "${1:-}" in
  ping)
    cat submission.json
    ;;
  T3)
    source /opt/ros/humble/setup.bash
    source /opt/unitree/install/local_setup.bash
    export RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_cyclonedds_cpp}"
    exec python3 -m go1_t3.ros_nodes "${@:2}"
    ;;
  *)
    printf 'Usage: /entrypoint.sh ping|T3\n' >&2
    exit 64
    ;;
esac
