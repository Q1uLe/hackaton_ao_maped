"""Run wall-time-bounded PPO fine-tuning with validation, recovery, and portable exports."""

import argparse
import ctypes
import hashlib
import json
import math
import os
import shutil
import time
from datetime import datetime, timezone
from functools import partial
from pathlib import Path

import numpy as np
import torch
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

from go1_t3.control import ROOT
from go1_t3.policy import BASE_WEIGHTS, RESIDUAL_WEIGHTS
from training.ppo_devices import TimedPPO, actor_layers, initialize_actor, make_model
from training.scenarios import FineTuneEnv, Validation, score


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def sha256(path):
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def write_json(path, obj):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def copy_atomic(source, path):
    tmp = path.with_suffix(".tmp")
    shutil.copyfile(source, tmp)
    os.replace(tmp, path)


def export_actor(model, path):
    arrays = {}
    for i, layer in enumerate(actor_layers(model)):
        arrays[f"residual_{i}_w"] = layer.weight.detach().cpu().numpy()
        arrays[f"residual_{i}_b"] = layer.bias.detach().cpu().numpy()
    if not all(np.isfinite(values).all() for values in arrays.values()):
        raise FloatingPointError("The PPO actor contains nonfinite weights")
    tmp = path.with_suffix(".tmp")
    with tmp.open("wb") as file:
        np.savez_compressed(file, **arrays)
    os.replace(tmp, path)


def save_ppo(model, path):
    tmp = path.with_name(path.stem + ".tmp.zip")
    model.save(tmp)
    os.replace(tmp, path)


def set_sleep_prevention(on):
    if os.name == "nt":
        flags = 0x80000001 if on else 0x80000000  # ES_CONTINUOUS, ES_SYSTEM_REQUIRED
        return bool(ctypes.windll.kernel32.SetThreadExecutionState(flags))
    return False


class RunMonitor(BaseCallback):
    def __init__(self, out, config, settings, resumed=None):
        super().__init__()
        self.out = out
        self.config = config
        self.settings = settings
        self.deadline = settings["deadline_unix"]
        self.validation = Validation()
        self.episodes = []
        self.last_status = 0.0
        self.last_checkpoint = time.time()
        self.next_validation = time.time() + config["early_validation_s"]
        self.baseline = None
        self.best = None
        self.best_steps = 0
        self.trained_steps = 0
        self.last_validation_steps = 0
        self.difficulty = config["starting_difficulty"]
        self.lr = config["learning_rate"]
        self.regressions = 0
        self.stale_evaluations = 0
        if resumed:
            self.baseline = json.loads((out / "baseline-validation.json").read_text(encoding="utf-8"))
            self.best = json.loads((out / "best-validation.json").read_text(encoding="utf-8"))
            for key in ("best_steps", "trained_steps", "last_validation_steps", "difficulty", "lr", "regressions",
                        "stale_evaluations"):
                setattr(self, key, resumed[key])

    def log(self, name, row, show=False):
        item = {"utc": utc_now(), **row}
        line = json.dumps(item, allow_nan=False)
        with (self.out / name).open("a", encoding="utf-8") as file:
            file.write(line + "\n")
        if show:
            print(line, flush=True)

    def status(self, model, state, **extra):
        now = time.time()
        row = {"state": state, "pid": os.getpid(), "updated_utc": utc_now(), "steps": model.num_timesteps,
               "start_utc": self.settings["start_utc"], "deadline_utc": self.settings["deadline_utc"],
               "seconds_remaining": round(max(0, self.deadline - now)), "mode": self.settings["mode"],
               "best_steps": self.best_steps, "trained_steps": self.trained_steps,
               "last_validation_steps": self.last_validation_steps, "difficulty": self.difficulty, "lr": self.lr,
               "regressions": self.regressions, "stale_evaluations": self.stale_evaluations, **extra}
        if model.device.type == "cuda":
            row["cuda_peak_allocated_mib"] = round(torch.cuda.max_memory_allocated(model.device) / (1024 * 1024), 2)
        if self.best is not None:
            row["best_score"] = list(score(self.best))
        write_json(self.out / "status.json", row)
        self.last_status = now

    def configure(self, model):
        model.learning_rate = self.lr
        model._setup_lr_schedule()
        model.env.env_method("set_difficulty", self.difficulty)

    def bootstrap(self, model):
        self.status(model, "baseline_validation")
        self.baseline = self.validation.evaluate(self.out / "baseline.npz")
        self.best = self.baseline
        write_json(self.out / "baseline-validation.json", self.baseline)
        write_json(self.out / "best-validation.json", self.best)
        copy_atomic(self.out / "baseline.npz", self.out / "best.npz")
        save_ppo(model, self.out / "best.zip")
        save_ppo(model, self.out / "latest.zip")
        self.log("events.jsonl", {"event": "baseline", "score": list(score(self.best))}, show=True)
        self.configure(model)
        self.status(model, "training")

    def _on_step(self):
        self.episodes.extend(info["episode"] for info in self.locals["infos"] if "episode" in info)
        return time.time() < self.deadline and not (self.out / "STOP").exists()

    def validate(self, model):
        self.status(model, "validation")
        candidate = self.out / f"residual_{self.trained_steps:09d}.npz"
        export_actor(model, candidate)
        metrics = self.validation.evaluate(candidate)
        self.last_validation_steps = self.trained_steps
        row = {"steps": self.trained_steps, "difficulty": self.difficulty, "lr": self.lr,
               "weights": candidate.name, "metrics": metrics}
        self.log("validation.jsonl", row)
        current = score(metrics)
        best = score(self.best)
        improved = current[:2] > best[:2] or (current[:2] == best[:2] and current[2] > best[2] + 0.015)
        if improved:
            self.best, self.best_steps = metrics, self.trained_steps
            save_ppo(model, self.out / "best.zip")
            copy_atomic(candidate, self.out / "best.npz")
            write_json(self.out / "best-validation.json", metrics)
            self.regressions = self.stale_evaluations = 0
        else:
            self.stale_evaluations += 1
            regressed = current[:2] < best[:2] or (current[:2] == best[:2] and current[2] < best[2] - 0.5)
            self.regressions = self.regressions + 1 if regressed else 0

        complete = current[:2] == (metrics["nominal_total"], metrics["perturbed_total"])
        if complete and self.regressions == 0:
            self.difficulty = min(self.config["max_difficulty"], self.difficulty + (0.08 if improved else 0.03))
        if self.regressions >= 2:
            self.rollback(model, "validation_regression")
        elif self.stale_evaluations >= 6:
            self.lr = max(self.config["min_learning_rate"], self.lr * 0.82)
            self.stale_evaluations = 0
        self.configure(model)
        self.log("events.jsonl", {"event": "validation", "steps": self.trained_steps, "score": list(current),
                                 "best_steps": self.best_steps, "improved": improved}, show=True)
        self.next_validation = time.time() + self.config["validation_interval_s"]

    def rollback(self, model, reason):
        model.set_parameters(self.out / "best.zip", device=model.device)
        self.lr = max(self.config["min_learning_rate"], self.lr * 0.7)
        self.difficulty = max(self.config["starting_difficulty"], self.difficulty - 0.1)
        self.regressions = 0
        self.stale_evaluations = 0
        self.configure(model)
        self.log("events.jsonl", {"event": "rollback", "reason": reason, "steps": model.num_timesteps,
                                 "best_steps": self.best_steps, "lr": self.lr}, show=True)

    def record_update(self, model):
        self.trained_steps = model.num_timesteps
        total = model.rollout_seconds + model.update_seconds
        logger = model.logger.name_to_value
        row = {"steps": model.num_timesteps, "rollout_seconds": round(model.rollout_seconds, 4),
               "update_seconds": round(model.update_seconds, 4),
               "steps_per_s": round(model.n_steps * model.n_envs / max(total, 1e-9), 2),
               "episodes": len(self.episodes), "falls": sum(bool(x["fallen"]) for x in self.episodes),
               "lr": self.lr, "difficulty": self.difficulty}
        if self.episodes:
            row["mean_reward"] = float(np.mean([x["r"] for x in self.episodes]))
            row["mean_distance_m"] = float(np.mean([x["distance_m"] for x in self.episodes]))
            row["mean_abs_lateral_m"] = float(np.mean([abs(x["lateral_m"]) for x in self.episodes]))
        self.episodes.clear()
        for name in ("approx_kl", "clip_fraction", "value_loss", "entropy_loss", "explained_variance"):
            value = float(logger.get(f"train/{name}", math.nan))
            row[name] = value if math.isfinite(value) else None
        self.log("train.jsonl", row)
        now = time.time()
        if row["approx_kl"] is None:
            self.rollback(model, "nonfinite_kl")
        evaluated = now >= self.next_validation and now < self.deadline
        if evaluated:
            try:
                self.validate(model)
            except FloatingPointError:
                self.rollback(model, "nonfinite_actor")
                self.next_validation = time.time() + self.config["validation_interval_s"]
            self.last_checkpoint = 0.0
        if now - self.last_checkpoint >= self.config["checkpoint_interval_s"]:
            save_ppo(model, self.out / "latest.zip")
            self.last_checkpoint = now
        if evaluated or time.time() - self.last_status >= 25:
            self.status(model, "training", last_update=row)

    def finish(self, model, *, audit=True):
        if self.trained_steps > self.last_validation_steps:
            try:
                self.validate(model)
            except FloatingPointError:
                self.rollback(model, "nonfinite_actor_at_finish")
        save_ppo(model, self.out / "latest.zip")
        if audit:
            self.status(model, "holdout")
            holdout = Validation(holdout=True)
            baseline = holdout.evaluate(self.out / "baseline.npz")
            candidate = holdout.evaluate(self.out / "best.npz")
        else:
            baseline, candidate = self.baseline, self.best
        improved = (score(candidate) > score(baseline)
                    and candidate["nominal_successes"] == candidate["nominal_total"]
                    and candidate["perturbed_successes"] >= baseline["perturbed_successes"]
                    and candidate["median_nominal_seconds"] <= baseline["median_nominal_seconds"] + 0.25)
        final = {"total_steps": model.num_timesteps, "trained_steps": self.trained_steps,
                 "best_steps": self.best_steps, "holdout": audit,
                 "improved": improved, "elapsed_seconds": round(time.time() - self.settings["start_unix"]),
                 "baseline_sha256": sha256(self.out / "baseline.npz"), "candidate_sha256": sha256(self.out / "best.npz"),
                 "baseline": baseline, "candidate": candidate}
        write_json(self.out / "final.json", final)
        self.status(model, "complete", improved=improved, final_report="final.json")
        self.log("events.jsonl", {"event": "complete", "steps": model.num_timesteps, "best_steps": self.best_steps,
                                 "improved": improved}, show=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "training" / "long_config.json")
    parser.add_argument("--hours", type=float, default=8.0)
    parser.add_argument("--mode", choices=("cpu", "cuda", "hybrid"), default="hybrid")
    parser.add_argument("--initial", type=Path, default=RESIDUAL_WEIGHTS)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="Use validation episodes for the final quick check")
    args = parser.parse_args()
    out = args.out.resolve()
    resumed = None
    torch.set_num_threads(1)
    if args.resume:
        settings = json.loads((out / "run.json").read_text(encoding="utf-8"))
        config = json.loads((out / "config.json").read_text(encoding="utf-8"))
        resumed = json.loads((out / "status.json").read_text(encoding="utf-8"))
        if resumed["state"] == "complete":
            raise SystemExit("This run has already completed")
    else:
        if args.hours <= 0:
            parser.error("--hours must be positive")
        out.mkdir(parents=True, exist_ok=False)
        config = json.loads(args.config.read_text(encoding="utf-8"))
        start = time.time()
        deadline = start + args.hours * 3600
        settings = {"start_unix": start, "start_utc": utc_now(), "deadline_unix": deadline,
                    "deadline_utc": datetime.fromtimestamp(deadline, timezone.utc).isoformat(),
                    "mode": args.mode, "base_weights_sha256": sha256(BASE_WEIGHTS), "initial_sha256": sha256(args.initial),
                    "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
                    "torch_version": torch.__version__}
        write_json(out / "config.json", config)
        write_json(out / "run.json", settings)
        copy_atomic(args.initial, out / "baseline.npz")

    if settings["mode"] != "cpu" and not torch.cuda.is_available():
        raise RuntimeError("CUDA training was requested but torch.cuda.is_available() is false")
    sleep_prevented = set_sleep_prevention(True)
    print(json.dumps({"event": "start", "sleep_prevented": sleep_prevented, **settings}), flush=True)
    env = None
    try:
        env = VecMonitor(SubprocVecEnv([partial(FineTuneEnv, config) for _ in range(config["workers"])],
                                       start_method="spawn"),
                         info_keywords=("distance_m", "lateral_m", "fallen"))
        if resumed and (out / "latest.zip").exists():
            model = TimedPPO.load(out / "latest.zip", env=env, device="cpu" if settings["mode"] == "cpu" else "cuda")
            monitor = RunMonitor(out, config, settings, resumed)
            monitor.trained_steps = model.num_timesteps
            monitor.configure(model)
        else:
            model = make_model(env, config, settings["mode"])
            initialize_actor(model, out / "baseline.npz")
            monitor = RunMonitor(out, config, settings)
            monitor.bootstrap(model)
        model.after_update = monitor.record_update
        monitor.status(model, "training")
        if time.time() < settings["deadline_unix"] and not (out / "STOP").exists():
            model.learn(1_000_000_000, callback=monitor, reset_num_timesteps=False)
        monitor.finish(model, audit=not args.smoke)
    except BaseException as error:
        if "monitor" in locals() and "model" in locals():
            monitor.status(model, "failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        if env is not None:
            env.close()
        if sleep_prevented:
            set_sleep_prevention(False)


if __name__ == "__main__":
    main()
