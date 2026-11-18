#!/bin/bash
if [[ "$1" == "ping" ]]; then
	echo '{"protocol_version": "1.0", "supported_tasks": [], "self_trained_rl": false, "preferred_speed_mps": null}'
fi
