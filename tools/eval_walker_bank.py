#!/usr/bin/env python3
"""Evaluate saved policies on a frozen bank of courses, without any adaptation.

Run from the repository root with the Isaac Lab Python:
    ~/env_isaaclab/bin/python tools/eval_walker_bank.py --bank bank.json --output out.json \
        --num_envs 8192 runs/x/full_seed_1/checkpoint.json ...
The library score of a course is the best return among a run's policies, an oracle
metric rather than one general policy. A library covers a course if any policy solves it.
GPU worlds are not repeatable, so each pair averages rollouts and needs a majority of finishes.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
from compare_walker import run_name
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / ".cache"))
os.environ.setdefault("WARP_CACHE_PATH", str(ROOT / ".cache" / "warp"))


def make_bank(seed=4000000000):
    """Frozen grid across the paper's three obstacle families, two course seeds each."""
    from isaaclab_port.walker.terrain import EnvConfig
    bank = []
    index = 0
    for roughness in (0.0, 1.5, 3.0, 4.5):
        for pit in ((), (0.8, 2.0), (2.0, 4.0), (4.0, 6.0)):
            for stump in ((), (0.4, 1.0), (1.0, 1.8), (1.8, 2.6)):
                stump_fields = dict(stump_width=(1, 2), stump_height=stump, stump_float=(0, 1)) if stump else {}
                config = EnvConfig(ground_roughness=roughness, pit_gap=pit, **stump_fields)
                for _ in range(2):
                    bank.append(dict(config=config, seed=seed + index))
                    index += 1
    return bank


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoints", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num_envs", type=int, default=8192)
    parser.add_argument("--repeats", type=int, default=5)
    from isaaclab_tasks.utils import add_launcher_args, launch_simulation
    add_launcher_args(parser)
    args = parser.parse_args()
    args.device = args.device or "cuda:0"
    import warp as wp
    wp.config.kernel_cache_dir = os.environ["WARP_CACHE_PATH"]
    import numpy as np
    from dataclasses import asdict
    from isaaclab_port.walker.engine import RolloutEngine
    from isaaclab_port.walker.env import WalkerEnv, make_cfg
    from isaaclab_port.walker.poet import PARAMETER_COUNT

    bank = make_bank()
    cfg = make_cfg(num_envs=args.num_envs, device=args.device, terrain_capacity=len(bank) + 16)
    with launch_simulation(cfg, args):
        env = WalkerEnv(cfg)
        engine = RolloutEngine(env, PARAMETER_COUNT)
        results = []
        start = time.monotonic()
        for path in args.checkpoints:
            data = json.loads(path.read_text())
            policies = [np.array(p["theta"]) for p in data["pairs"] + data.get("retired", [])]
            tasks = [(item["config"], item["seed"], theta) for theta in policies for item in bank
                     for _ in range(args.repeats)]
            batch = engine.run(tasks)
            returns = batch.returns.reshape(len(policies), len(bank), args.repeats).mean(axis=2)
            finished = batch.finished.reshape(len(policies), len(bank), args.repeats).mean(axis=2) > 0.5
            solved = finished & (returns >= 230.0)
            results.append(dict(
                run=run_name(path), iteration=data["iteration"], steps=data["steps"], rollouts=data["rollouts"],
                policies=len(policies),
                mean_library_return=float(returns.max(axis=0).mean()),
                mean_best_single_policy_return=float(returns.mean(axis=1).max()),
                courses_solved_by_library=int(solved.any(axis=0).sum()), courses=len(bank),
                courses_solved_by_best_policy=int(solved.sum(axis=1).max()),
                per_course_library_return=returns.max(axis=0).tolist(),
                per_course_solved=solved.any(axis=0).tolist()))
            print(json.dumps({k: v for k, v in results[-1].items() if not k.startswith("per_")}), flush=True)
        args.output.write_text(json.dumps(dict(
            bank=[dict(config=asdict(item["config"]), seed=item["seed"]) for item in bank],
            repeats=args.repeats, results=results, wall_seconds=time.monotonic() - start), indent=1))
        sys.stdout.flush()
        os._exit(0)


if __name__ == "__main__":
    main()
