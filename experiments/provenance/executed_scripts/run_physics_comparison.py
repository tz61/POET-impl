#!/usr/bin/env python3
"""Actual CPU physics comparison on four fixed tasks, not open-ended POET.

Uses production POET.optimize/transfer and Evaluator; never calls reproduce.
--pilot runs one seed/full mode/two iterations to measure feasibility first.
All outputs are new. This script never changes existing code or artifacts.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback

WORKSPACE = Path(__file__).resolve().parents[2]
PROJECT = WORKSPACE / "poet-reproduction"
sys.path.insert(0, str(PROJECT))
sys.dont_write_bytecode = True
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("XDG_CACHE_HOME", str(PROJECT / ".cache"))
os.environ.setdefault("WARP_CACHE_PATH", str(PROJECT / ".cache" / "warp"))

import numpy as np
from isaaclab_port.core import PARAMETER_COUNT, POET, Settings, Terrain

SCOPE = "Actual IsaacLab Beta CPU physics; controlled fixed-task population, not full open-ended POET."
METRIC_KEYS = ("requested_episodes", "first_episode_steps", "simulated_world_steps")


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def hash_array(value):
    return hashlib.sha256(np.asarray(value, dtype="<f8").tobytes()).hexdigest()


def fresh_population(seed, mode, pairs, training_tasks, evaluator, event):
    settings = Settings(seed=seed, pairs=pairs, transfers=mode != "no_transfer",
                        periodic_transfer_mode="topk" if mode == "topk" else "full",
                        transfer_top_k=1, transfer_interval=5, max_active=4)
    poet = POET(settings, evaluator, event)
    # The same seed generates identical independent policies for every mode.
    policies = np.random.default_rng(seed).normal(0., .1, (len(training_tasks), PARAMETER_COUNT))
    poet.active = [poet.new_pair(terrain, theta, 0) for terrain, theta in zip(training_tasks, policies)]
    poet.archive = list(training_tasks)
    return poet, hash_array(policies)


class MeteredEvaluator:
    """Partition actual production evaluator work into training and evaluation."""
    def __init__(self, delegate, stream):
        self.delegate = delegate
        self.stream = stream
        self.phase = "train"
        self.totals = {phase: {**dict.fromkeys(METRIC_KEYS, 0), "evaluation_wall_seconds": 0.0,
                               "evaluator_calls": 0} for phase in ("train", "eval")}

    def __call__(self, terrain, matrix):
        matrix = np.atleast_2d(matrix)
        before = self.delegate.metrics()
        started = time.monotonic()
        values = self.delegate(terrain, matrix)
        elapsed = time.monotonic() - started
        after = self.delegate.metrics()
        delta = {key: after[key] - before[key] for key in METRIC_KEYS}
        for key, amount in delta.items():
            self.totals[self.phase][key] += amount
        self.totals[self.phase]["evaluation_wall_seconds"] += elapsed
        self.totals[self.phase]["evaluator_calls"] += 1
        row = {"phase": self.phase, "terrain": asdict(terrain), "policies": len(matrix),
               "policy_matrix_sha256": hash_array(matrix), "returns": values.tolist(),
               "episode_lengths": list(self.delegate.last_episode_lengths),
               "goal_successes": list(self.delegate.last_episode_successes),
               "wall_seconds": elapsed, **delta}
        for name in ("last_terminal_x", "last_max_x"):
            value = getattr(self.delegate, name, None)
            if value is not None:
                row[name.removeprefix("last_")] = value
        self.last = row
        self.stream.write(json.dumps(row, allow_nan=False) + "\n")
        self.stream.flush()
        return values


def evaluate_library(poet, evaluator, heldout):
    """No adaptation: evaluate each current policy on every frozen test task."""
    previous_phase = evaluator.phase
    evaluator.phase = "eval"
    policies = np.stack([pair.theta.copy() for pair in poet.active])
    rows = []
    try:
        for label, terrain in heldout:
            returns = evaluator(terrain, policies)
            row = {"task_id": label, "terrain": asdict(terrain),
                   "returns_by_policy": returns.tolist(),
                   "goal_successes_by_policy": list(evaluator.delegate.last_episode_successes),
                   "episode_lengths_by_policy": list(evaluator.delegate.last_episode_lengths),
                   "terminal_x_by_policy": getattr(evaluator.delegate, "last_terminal_x", None),
                   "max_x_by_policy": getattr(evaluator.delegate, "last_max_x", None)}
            row["best_library_return"] = float(np.max(returns))
            row["any_policy_reached_goal"] = bool(any(row["goal_successes_by_policy"]))
            rows.append(row)
    finally:
        evaluator.phase = previous_phase
    return {"policy_rule": "all four current theta; test-time oracle best-of-library, no adaptation",
            "policy_matrix_sha256": hash_array(policies), "tasks": rows,
            "mean_best_library_return": float(np.mean([row["best_library_return"] for row in rows])),
            "tasks_reached_goal": sum(row["any_policy_reached_goal"] for row in rows),
            "task_count": len(rows)}


def run_condition(seed, mode, args, training_tasks, heldout, delegate, output, fingerprint):
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    events = []
    with (output / "events.jsonl").open("w", buffering=1) as event_stream, \
         (output / "evaluations.jsonl").open("w", buffering=1) as evaluation_stream, \
         (output / "progress.jsonl").open("w", buffering=1) as progress_stream:
        def event(kind, **data):
            record = {"event": kind, **data}
            events.append(record)
            event_stream.write(json.dumps(record) + "\n")

        evaluator = MeteredEvaluator(delegate, evaluation_stream)
        poet, initial_hash = fresh_population(seed, mode, args.pairs, training_tasks, evaluator, event)
        write_json(output / "settings.json", {"scope": SCOPE, "settings": asdict(poet.settings),
                   "episode_steps": args.episode_steps, "num_envs": args.num_envs,
                   "target_training_world_steps": args.world_step_budget,
                   "soft_wall_seconds": args.max_seconds, "initial_policy_sha256": initial_hash})
        write_json(output / "task_fingerprint.json", fingerprint)
        for pair in poet.active:
            pair.score = float(poet.evaluate(pair.terrain, pair.theta)[0])
            poet.update_best(pair)
        checkpoints = []

        def snapshot():
            number = poet.iteration
            path = output / f"checkpoint_{number:03d}.json"
            poet.save(path)
            result = evaluate_library(poet, evaluator, heldout)
            item = {"iteration": number, "checkpoint": path.name,
                    "train_metrics": dict(evaluator.totals["train"]),
                    "eval_metrics": dict(evaluator.totals["eval"]),
                    "wall_seconds": time.monotonic() - started, "evaluation": result}
            write_json(output / f"evaluation_{number:03d}.json", item)
            checkpoints.append(item)
            return item

        snapshot()
        stop_reason = "iteration_limit"
        while poet.iteration < args.iterations:
            if args.world_step_budget and evaluator.totals["train"]["simulated_world_steps"] >= args.world_step_budget:
                stop_reason = "training_world_step_budget_at_iteration_boundary"
                break
            if args.max_seconds and time.monotonic() - started >= args.max_seconds:
                stop_reason = "soft_wall_budget_at_iteration_boundary"
                break
            evaluator.phase = "train"
            for pair in poet.active:
                poet.optimize(pair)
            poet.iteration += 1
            if poet.iteration % 5 == 0 and poet.settings.transfers:
                poet.transfer()
            record = {"iteration": poet.iteration, "mode": mode, "seed": seed,
                      "train_scores": [pair.score for pair in poet.active],
                      "train_metrics": dict(evaluator.totals["train"]),
                      "eval_metrics": dict(evaluator.totals["eval"]),
                      "wall_seconds": time.monotonic() - started}
            progress_stream.write(json.dumps(record, allow_nan=False) + "\n")
            print(json.dumps({"progress": str(output.name), **record}), flush=True)
            if poet.iteration % 5 == 0:
                snapshot()
        if checkpoints[-1]["iteration"] != poet.iteration:
            snapshot()
        attempts = [event for event in events if event["event"] == "transfer_attempt"]
        by_event = {}
        for attempt in attempts:
            key = (attempt["iteration"], attempt["target"])
            by_event.setdefault(key, []).append(attempt["direct_score"])
        training_steps = evaluator.totals["train"]["simulated_world_steps"]
        result = {"status": "completed", "scope": SCOPE, "seed": seed, "mode": mode,
                  "initial_policy_sha256": initial_hash, "iterations_completed": poet.iteration,
                  "algorithm_policy_evaluations": poet.evaluations,
                  "active_tasks": len(poet.active), "archive_size": len(poet.archive),
                  "reproduction_enabled": False,
                  "stop_reason": stop_reason, "target_training_world_steps": args.world_step_budget,
                  "training_world_step_overshoot": max(0, training_steps - args.world_step_budget) if args.world_step_budget else 0,
                  "train_metrics": evaluator.totals["train"], "eval_metrics": evaluator.totals["eval"],
                  "wall_seconds": time.monotonic() - started,
                  "transfer_summary": {"direct_attempts": len(attempts),
                       "adaptation_proposals": sum(event["proposal_selected"] for event in attempts),
                       "accepted_targets": sum(event["event"] == "transfer_accepted" for event in events),
                       "target_events_with_direct_score_ties": sum(len(values) != len(set(values)) for values in by_event.values())},
                  "initial_evaluation": checkpoints[0]["evaluation"],
                  "final_evaluation": checkpoints[-1]["evaluation"], "checkpoints": checkpoints}
        write_json(output / "summary.json", result)
        return result


def make_physical_evaluator(args, cfg_factory):
    import torch
    from isaaclab_port.env import PoetBipedEnv
    from isaaclab_port.runtime import Evaluator

    class PositionObservedEnv(PoetBipedEnv):
        def _get_dones(self):
            result = super()._get_dones()
            # Clone before DirectRLEnv's same-step reset; never change dynamics.
            self.extras["diagnostic_root_x"] = self.robot.data.joint_pos.torch[:, self._root_x].clone()
            return result

        def reset(self, *reset_args, **kwargs):
            result = super().reset(*reset_args, **kwargs)
            self._diagnostic_active = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
            self._diagnostic_terminal_x = torch.full((self.num_envs,), self.cfg.start_x, device=self.device)
            self._diagnostic_max_x = self._diagnostic_terminal_x.clone()
            return result

        def step(self, actions):
            result = super().step(actions)
            _, _, terminated, truncated, info = result
            x = info["diagnostic_root_x"]
            active = self._diagnostic_active
            self._diagnostic_terminal_x = torch.where(active, x, self._diagnostic_terminal_x)
            self._diagnostic_max_x = torch.where(active, torch.maximum(self._diagnostic_max_x, x), self._diagnostic_max_x)
            self._diagnostic_active &= ~(terminated | truncated)
            return result

    class PositionEvaluator(Evaluator):
        def __call__(self, terrain, matrix):
            if self.env is None or self.terrain != terrain:
                self.close()
                self.env = PositionObservedEnv(self.cfg_factory(terrain))
                self.terrain = terrain
            values = super().__call__(terrain, matrix)
            # Every heldout call evaluates all four policies in one batch.
            # Multi-batch training calls do not pretend the last batch is all.
            n = len(matrix)
            self.last_terminal_x = self.env._diagnostic_terminal_x[:n].cpu().tolist() if n <= self.num_envs else None
            self.last_max_x = self.env._diagnostic_max_x[:n].cpu().tolist() if n <= self.num_envs else None
            return values
    return PositionEvaluator(args.num_envs, cfg_factory)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--seeds", type=int, nargs="+", default=[24582922, 24582923, 24582924])
    parser.add_argument("--modes", nargs="+", choices=("full", "topk", "no_transfer"), default=["full", "topk", "no_transfer"])
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--pairs", type=int, default=4)
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--episode-steps", type=int, default=600)
    parser.add_argument("--substeps", type=int, default=16)
    parser.add_argument("--world-step-budget", type=int, default=30000)
    parser.add_argument("--max-seconds", type=float, default=180.)
    args = parser.parse_args()
    if args.pilot:
        args.seeds, args.modes, args.iterations = args.seeds[:1], ["full"], 2
    if min(args.iterations, args.pairs, args.episode_steps, args.substeps) < 1 or args.num_envs < 4:
        parser.error("positive iterations/pairs/episode-steps/substeps and num-envs>=4 required")
    if args.world_step_budget < 0 or args.max_seconds < 0:
        parser.error("budgets must be nonnegative; 0 disables that limit")
    if len(set(args.seeds)) != len(args.seeds) or len(set(args.modes)) != len(args.modes):
        parser.error("duplicate seeds/modes would overwrite output")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("output must be new or empty; old artifacts are never overwritten")
    args.output.mkdir(parents=True, exist_ok=True)
    report = {"status": "started", "scope": SCOPE, "pilot": args.pilot, "runs": []}
    delegate = None
    try:
        from isaaclab_port.doctor import inspect
        from isaaclab_port.env import make_env_cfg, build_course_profile
        from isaaclab_port.task_fingerprint import build_task_fingerprint
        from isaaclab_newton.physics import NewtonCfg, FeatherstoneSolverCfg
        try:
            from isaaclab.app import launch_simulation
        except ImportError:
            from isaaclab_tasks.utils import launch_simulation
        import warp as wp
        wp.config.kernel_cache_dir = str(PROJECT / ".cache" / "warp")
        environment = inspect()
        write_json(args.output / "environment.json", environment)
        training_tasks = [Terrain(seed=910001), Terrain(roughness=.06, seed=910002),
                          Terrain(stump_height=.1, seed=910003), Terrain(gap_width=.4, seed=910004)]
        bank_path = WORKSPACE / "presentation" / "experiments" / "heldout_terrains.json"
        bank = json.loads(bank_path.read_text())
        chosen = ("rough_low", "stump_low", "gap_mid", "mixed_low")
        heldout = [(item["id"], Terrain(**item["terrain"])) for item in bank["terrains"] if item["id"] in chosen]
        assert len(heldout) == 4

        def geometry(terrain):
            profile = build_course_profile(asdict(terrain), terrain.seed)
            values = [None if np.isnan(x) else float(x) for x in profile]
            return {"terrain": asdict(terrain), "tile_top_heights_m": values,
                    "profile_sha256": hashlib.sha256(json.dumps(values, separators=(",", ":")).encode()).hexdigest()}

        train_geometry = [geometry(terrain) for terrain in training_tasks]
        eval_geometry = [{"task_id": label, **geometry(terrain)} for label, terrain in heldout]
        train_hashes = {item["profile_sha256"] for item in train_geometry}
        for item in eval_geometry:
            item["geometry_overlaps_training"] = item["profile_sha256"] in train_hashes
        assert not any(item["geometry_overlaps_training"] for item in eval_geometry)
        protocol = {"scope": SCOPE, "arguments": vars(args) | {"output": str(args.output)},
                    "training_tasks": train_geometry, "heldout_tasks": eval_geometry,
                    "heldout_source_sha256": hashlib.sha256(bank_path.read_bytes()).hexdigest(),
                    "warning": "Evaluation geometries are disjoint from the four fixed training geometries, not a representative sample of possible tasks.",
                    "transfer_iterations": "5, 10, 15, 20 after optimization; no iteration-zero transfer",
                    "success_definition": "reaches x>=30m without falling; no paper 230-point criterion",
                    "policy_evaluation_rule": "all four current theta, best-of-library oracle; no heldout adaptation"}
        write_json(args.output / "protocol.json", protocol)

        def cfg_factory(terrain):
            return make_env_cfg(asdict(terrain), seed=terrain.seed, num_envs=args.num_envs,
                       device="cpu", episode_steps=args.episode_steps,
                       physics=NewtonCfg(solver_cfg=FeatherstoneSolverCfg(),
                                         use_cuda_graph=False, num_substeps=args.substeps))

        fingerprint = build_task_fingerprint(episode_steps=args.episode_steps, solver="featherstone",
            substeps=args.substeps,
            simulation_versions={"source_commit": environment.get("source_commit"), **environment["packages"]},
            task_config={"device_family": "cpu", "num_envs": args.num_envs,
                         "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                         "fixed_task_protocol": {
                             "training_tasks": train_geometry, "heldout_tasks": eval_geometry,
                             "transfer_iterations": protocol["transfer_iterations"],
                             "success_definition": protocol["success_definition"],
                             "policy_evaluation_rule": protocol["policy_evaluation_rule"]}})
        report["environment"] = environment
        report["protocol"] = protocol
        total_start = time.monotonic()
        with launch_simulation(cfg_factory(training_tasks[0])):
            for seed in args.seeds:
                for mode in args.modes:
                    delegate = make_physical_evaluator(args, cfg_factory)
                    try:
                        run = run_condition(seed, mode, args, training_tasks, heldout, delegate,
                                            args.output / f"{mode}_seed_{seed}", fingerprint)
                    finally:
                        delegate.close()
                        delegate = None
                    report["runs"].append(run)
                    report["total_wall_seconds"] = time.monotonic() - total_start
                    write_json(args.output / "comparison.json", report)
        report["status"] = "completed"
        write_json(args.output / "comparison.json", report)
        print(json.dumps({"completed": str(args.output), "runs": len(report["runs"]),
                          "wall_seconds": report["total_wall_seconds"]}), flush=True)
        return 0
    except Exception as exc:
        report.update(status="failed", error=repr(exc), traceback=traceback.format_exc())
        write_json(args.output / "comparison.json", report)
        traceback.print_exc()
        return 1
    finally:
        if delegate is not None:
            delegate.close()


if __name__ == "__main__":
    raise SystemExit(main())
