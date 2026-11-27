"""Replay the public MuJoCo Lowlevel motor loop and score local T3 episodes."""

import argparse
import json
from pathlib import Path

import numpy as np

from go1_t3.control import POLICY_PERIOD, ROOT, SAFETY_PERIOD
from go1_t3.policy import KD, KP, RESIDUAL_WEIGHTS, WalkingPolicy
from go1_t3.safety import SafetyBlock, SelfCollision
from tools.plant import Plant


def episode(plant, policy, safety, rng, speed, limit, *, warmup_walk=False, trace=False, disturbance=None):
    plant.reset(rng)
    if disturbance is not None:
        disturbance.prepare(plant)
    policy.reset()
    safety.reset()
    d = plant.data
    warmup_steps = round(1.5 / SAFETY_PERIOD)
    velocity = np.array([speed, 0, 0])
    zero = np.zeros(3)
    violation = np.zeros(3)
    min_height = float(d.qpos[2])
    initial_position = d.qpos[:2].copy()
    heading = np.array([1 - 2 * (d.qpos[5]**2 + d.qpos[6]**2),
                        2 * (d.qpos[4] * d.qpos[5] + d.qpos[3] * d.qpos[6])])
    heading /= np.linalg.norm(heading)
    lateral = np.array([-heading[1], heading[0]])
    distance = 0.0
    fallen = False
    for step in range(warmup_steps + round(limit / SAFETY_PERIOD)):
        if step == warmup_steps:
            initial_position = d.qpos[:2].copy()
        if step % round(POLICY_PERIOD / SAFETY_PERIOD) == 0:
            cmd = velocity if warmup_walk or step >= warmup_steps else zero
            command = policy.step(*plant.state(), cmd)
            if disturbance is not None:
                command = disturbance.delay(command)
            safety.receive(command, d.time)
        if disturbance is not None:
            disturbance.apply(plant, step * SAFETY_PERIOD)
        plant.advance(safety.step(d.time), SAFETY_PERIOD)
        min_height = min(min_height, d.qpos[2])
        violation = np.where(plant.pose_limits(), violation + 1, 0)
        if trace and step % round(0.5 / SAFETY_PERIOD) == 0:
            print(f"t={d.time:.2f} xyz={d.qpos[:3].round(3)} q={d.qpos[7:10].round(3)} "
                  f"target={safety.accepted[:3, 0].round(3)}")
        distance = float((d.qpos[:2] - initial_position) @ heading)
        fallen = bool(np.any(violation * SAFETY_PERIOD > 0.5))
        if fallen or (step >= warmup_steps and distance >= 3.0):
            break
    return {"success": not fallen and distance >= 3.0, "fallen": fallen,
            "distance_m": round(distance, 4), "scored_seconds": round(max(0, d.time - 1.5), 3),
            "lateral_m": round(float((d.qpos[:2] - initial_position) @ lateral), 4),
            "min_height_m": round(min_height, 4)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    config = json.loads((ROOT / "submission.json").read_text(encoding="utf-8"))
    parser.add_argument("--speed", type=float, default=config["preferred_speed_mps"])
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=32026)
    parser.add_argument("--limit", type=float, default=90)
    parser.add_argument("--kp", type=float, default=KP)
    parser.add_argument("--kd", type=float, default=KD)
    parser.add_argument("--frequency", type=float, default=3.0)
    parser.add_argument("--residual", type=Path)
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--nominal", action="store_true")
    parser.add_argument("--warmup-walk", action="store_true")
    parser.add_argument("--trace", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    plant = Plant()
    residual = args.residual
    if residual is None and config["self_trained_rl"] and not args.pretrained:
        residual = RESIDUAL_WEIGHTS
    policy = WalkingPolicy(residual_weights=residual)
    policy.kp = args.kp
    policy.kd = args.kd
    policy.gait[4] = args.frequency
    safety = SafetyBlock(SelfCollision())
    metrics = []
    for index in range(args.episodes):
        seed = None if args.nominal else args.seed + index
        rng = None if seed is None else np.random.default_rng(seed)
        result = episode(plant, policy, safety, rng, args.speed, args.limit,
                         warmup_walk=args.warmup_walk, trace=args.trace)
        row = {"episode": index + 1, "seed": seed, **result}
        metrics.append(row)
        print(json.dumps(row), flush=True)
    summary = {"preferred_speed_mps": args.speed, "warmup_walk": args.warmup_walk,
               "residual": residual.name if residual else None,
               "kp": args.kp, "kd": args.kd, "frequency_hz": args.frequency,
               "successes": sum(x["success"] for x in metrics), "episodes": metrics}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"{summary['successes']}/{args.episodes} successful episodes")


if __name__ == "__main__":
    main()
