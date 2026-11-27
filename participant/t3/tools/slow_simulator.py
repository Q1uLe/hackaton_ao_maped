"""Slow the public Go1 node's wall-time pacing for tests on emulated CPUs."""

import importlib.util
import sys
import time
from types import SimpleNamespace


def main():
    factor = float(sys.argv.pop(1))
    if not 0 < factor <= 1:
        raise ValueError("The real-time factor must be in (0, 1]")
    path = "/home/ros/go1_ws/install/go1_simulation/lib/go1_simulation/go1_mujoco.py"
    spec = importlib.util.spec_from_file_location("public_go1_mujoco", path)
    simulator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(simulator)
    start = time.perf_counter()
    simulator.time = SimpleNamespace(
        perf_counter=lambda: start + factor * (time.perf_counter() - start),
        sleep=time.sleep,
    )
    simulator.main()


if __name__ == "__main__":
    main()
