#!/usr/bin/env python3
"""Pick POET-solved niches as targets for the direct-ES and curriculum controls.

Usage: python3 tools/make_targets.py runs/x --output targets.json
Needs only the standard library. At each difficulty level it keeps the niches furthest
from flat ground. A target's budget is the iteration at which POET solved it, so a
control gets as many ES steps as POET's whole run had spent by then.
"""
import argparse
import json
import math
from pathlib import Path

from summarize_walker import read_jsonl, summarize_run

# Same scale as CurriculumRun.NORMALIZE.
NORMALIZE = (8.0, 8.0, 8.0, 3.0, 3.0)


def distance_from_flat(config):
    vector = (config.get("ground_roughness", 0), *(config.get("pit_gap") or (0, 0)),
              *(config.get("stump_height") or (0, 0)))
    return math.hypot(*(value / scale for value, scale in zip(vector, NORMALIZE)))


def pick_targets(root, modes=("full",), per_level=2):
    candidates = {}
    for folder in sorted(p for p in root.iterdir() if p.is_dir()):
        if not any(folder.name.startswith(f"{mode}_seed_") for mode in modes):
            continue
        run = summarize_run(folder)
        if not run:
            continue
        seeds = {e["niche"]: e["seed"] for e in read_jsonl(folder / "events.jsonl") if e["event"] == "admitted"}
        for name, niche in run["solved_niches"].items():
            if name not in seeds:  # The flat starting niche is no test of anything.
                continue
            key = (json.dumps(niche["config"], sort_keys=True), seeds[name])
            target = dict(name=f"{folder.name}_{name}", config=niche["config"], seed=seeds[name], budget=niche["at"],
                          level=niche["level"], distance=distance_from_flat(niche["config"]))
            # The same course solved in several runs keeps its earliest, hardest budget.
            if key not in candidates or target["budget"] < candidates[key]["budget"]:
                candidates[key] = target
    targets = []
    for level in sorted({t["level"] for t in candidates.values()}):
        chosen = sorted((t for t in candidates.values() if t["level"] == level),
                        key=lambda t: (-t["distance"], t["name"]))
        targets += chosen[:per_level]
    return targets


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--modes", nargs="+", default=["full"])
    parser.add_argument("--per_level", type=int, default=2)
    args = parser.parse_args()
    targets = pick_targets(args.root, args.modes, args.per_level)
    for t in targets:
        print(f"{t['name']:44s} {t['level']:22s} distance={t['distance']:.3f} budget={t['budget']}")
    if args.output:
        args.output.write_text(json.dumps(targets, indent=2))


if __name__ == "__main__":
    main()
