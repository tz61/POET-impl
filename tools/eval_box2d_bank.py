#!/usr/bin/env python3
"""Evaluate saved policies on the frozen course bank in the official Box2D environment.

The CPU twin of eval_walker_bank.py, for the fidelity question: the GPU port is a sibling
simulator, so a ranking of modes on GPU courses need not hold in the paper's simulator.
Box2D on a fixed course seed repeats exactly, so one episode per policy and course is enough.
Run with poet-reproduction/.venv:
    .venv/bin/python tools/eval_box2d_bank.py --bank bank.json --output out.json --workers 6 ckpt.json ...
The output has the same fields as eval_walker_bank.py, so compare_walker.py can report it.
"""
import argparse
import json
from multiprocessing import Pool
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare_walker import run_name
from eval_in_box2d import evaluate

BANK = []


def init(bank):
    BANK[:] = bank


def score_policy(theta):
    """Return and finish of one policy on every bank course."""
    rows = [evaluate(item["config"], item["seed"], theta) for item in BANK]
    return [row[0] for row in rows], [row[2] for row in rows]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoints", type=Path, nargs="+")
    parser.add_argument("--bank", type=Path, required=True, help="Output of eval_walker_bank.py, for its courses")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    bank = json.loads(args.bank.read_text())["bank"]
    start = time.monotonic()
    results = []
    with Pool(args.workers, initializer=init, initargs=(bank,)) as pool:
        for path in args.checkpoints:
            data = json.loads(path.read_text())
            policies = [p["theta"] for p in data["pairs"] + data.get("retired", [])]
            scored = pool.map(score_policy, policies)
            returns = np.array([row[0] for row in scored])
            finished = np.array([row[1] for row in scored])
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
    args.output.write_text(json.dumps(dict(bank=bank, repeats=1, simulator="box2d", results=results,
                                           wall_seconds=time.monotonic() - start), indent=1))


if __name__ == "__main__":
    main()
