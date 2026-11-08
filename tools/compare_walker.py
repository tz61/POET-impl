#!/usr/bin/env python3
"""Compare POET modes on the frozen course bank at equal iterations and at equal compute.

Top-1 exists to save compute, so a comparison at equal iterations alone misses its main claim.
`list` prints the checkpoints to give eval_walker_bank.py, and `report` reads what it wrote:
    python3 tools/compare_walker.py list runs/walker_main runs/walker_main_snapshots > checkpoints.txt
    python3 tools/compare_walker.py report bank.json --iteration 2000 --output comparison.json
Three seeds per mode cannot support a significance claim, so every seed stays visible.
"""
import argparse
import json
from pathlib import Path
import re

MODES = ("full", "topk", "no_transfer", "full_confirm", "topk_confirm")
METRICS = ("mean_library_return", "courses_solved_by_library", "courses_solved_by_best_policy",
           "steps", "rollouts", "iteration", "policies")


def run_name(path):
    """A checkpoint's run: its folder, or for a snapshot the file name less the iteration."""
    path = Path(path)
    if path.name == "checkpoint.json":
        return path.parent.name
    return re.sub(r"_\d{5}$", "", path.stem)


def list_checkpoints(root, snapshots, modes=MODES):
    """Every saved checkpoint of the requested modes, keeping one per run and iteration."""
    found = {}
    paths = sorted(Path(snapshots).glob("*_seed_*.json")) + sorted(Path(root).glob("*_seed_*/checkpoint.json"))
    for path in paths:
        run = run_name(path)
        if run.rsplit("_seed_", 1)[0] in modes:
            found.setdefault((run, json.loads(path.read_text())["iteration"]), path)
    return [found[key] for key in sorted(found)]


def by_run(results):
    runs = {}
    for row in results:
        runs.setdefault(row["run"], []).append(row)
    return {run: sorted(rows, key=lambda row: row["iteration"]) for run, rows in runs.items()}


def common_budget(runs):
    """The most compute every run has spent."""
    return min(rows[-1]["steps"] for rows in runs.values())


def at_iteration(runs, iteration):
    return {run: row for run, rows in runs.items() for row in rows if row["iteration"] == iteration}


def at_budget(runs, budget):
    """What each run held after spending at most the budget: its latest checkpoint within it."""
    view = {}
    for run, rows in runs.items():
        within = [row for row in rows if row["steps"] <= budget]
        if within:
            view[run] = within[-1]
    return view


def summarize(view, expected):
    """Per-mode values for each seed and their mean. Missing runs are named, never ignored."""
    table = {}
    for run in sorted(view):
        mode, seed = run.rsplit("_seed_", 1)
        for metric in METRICS:
            entry = table.setdefault(mode, {}).setdefault(metric, dict(seeds={}))
            entry["seeds"][seed] = view[run][metric]
    for metrics in table.values():
        for entry in metrics.values():
            entry["mean"] = sum(entry["seeds"].values()) / len(entry["seeds"])
    return dict(modes=table, missing=sorted(set(expected) - set(view)))


def print_view(title, summary):
    print(title)
    if summary["missing"]:
        print(f"  MISSING, so this view is not a fair comparison: {', '.join(summary['missing'])}")
    for mode, metrics in summary["modes"].items():
        for metric in METRICS:
            values = " ".join(f"{value:12.1f}" for value in metrics[metric]["seeds"].values())
            print(f"  {mode:12s} {metric:30s} {values}   mean {metrics[metric]['mean']:12.1f}")


def report(results, iteration=None, budget=None):
    runs = by_run(results)
    budget = common_budget(runs) if budget is None else budget
    iteration = min(rows[-1]["iteration"] for rows in runs.values()) if iteration is None else iteration
    return dict(iteration=iteration, budget=budget,
                equal_iteration=summarize(at_iteration(runs, iteration), runs),
                equal_compute=summarize(at_budget(runs, budget), runs))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    listing = commands.add_parser("list")
    listing.add_argument("root", type=Path)
    listing.add_argument("snapshots", type=Path)
    reporting = commands.add_parser("report")
    reporting.add_argument("bank", type=Path)
    reporting.add_argument("--iteration", type=int)
    reporting.add_argument("--budget", type=int)
    reporting.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.command == "list":
        print("\n".join(str(path) for path in list_checkpoints(args.root, args.snapshots)))
        return
    result = report(json.loads(args.bank.read_text())["results"], args.iteration, args.budget)
    print_view(f"Equal iterations: {result['iteration']}", result["equal_iteration"])
    print_view(f"Equal compute: at most {result['budget']} steps", result["equal_compute"])
    if args.output:
        args.output.write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
