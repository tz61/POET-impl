#!/usr/bin/env python3
"""Summarize train_walker.py runs: learning, solved niches by difficulty, and cost.

Usage: python3 tools/summarize_walker.py runs/x [--output summary.json]
Needs only the standard library, and reads each run's progress and events.
"""
import argparse
import json
import math
from pathlib import Path
import random

CHALLENGE = (("stump", 2.4), ("gap", 6.0), ("roughness", 4.5))
LEVELS = {0: "below_challenging", 1: "challenging", 2: "very_challenging", 3: "extremely_challenging"}


def difficulty(config):
    """The paper's Table 2: how many of three conditions the encoding meets."""
    stump = config.get("stump_height") or [0, 0]
    gap = config.get("pit_gap") or [0, 0]
    met = (stump[1] >= 2.4) + (gap[1] >= 6.0) + (config.get("ground_roughness", 0) >= 4.5)
    return LEVELS[int(met)]


def sample_probes(rng, per_level=1000):
    """The paper's coverage probes: uniform encodings, 1000 at each difficulty level."""
    probes = {level: [] for level in LEVELS.values() if level != "below_challenging"}
    while any(len(v) < per_level for v in probes.values()):
        roughness = rng.uniform(0.0, 10.0)
        gap = sorted(rng.uniform(0.0, 8.0) for _ in range(2))
        stump = sorted(rng.uniform(0.0, 5.0) for _ in range(2))
        level = difficulty(dict(ground_roughness=roughness, pit_gap=gap, stump_height=stump))
        if level in probes and len(probes[level]) < per_level:
            probes[level].append((roughness, *gap, *stump))
    return probes


def coverage(solved_configs, probes):
    """Mean distance from each probe to the nearest solved encoding; smaller is broader."""
    vectors = [(c.get("ground_roughness", 0), *(c.get("pit_gap") or (0, 0)), *(c.get("stump_height") or (0, 0)))
               for c in solved_configs]
    if not vectors:
        return None
    result = {}
    for level, points in probes.items():
        result[level] = sum(min(math.dist(p, v) for v in vectors) for p in points) / len(points)
    return result


def read_jsonl(path):
    if not path.exists():
        return []
    rows = []
    for line in path.read_text().splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:  # A run still writing may leave a partial last line.
            pass
    return rows


def summarize_run(folder):
    progress = read_jsonl(folder / "progress.jsonl")
    events = read_jsonl(folder / "events.jsonl")
    if not progress:
        return None
    last = progress[-1]
    admitted = {e["niche"]: e for e in events if e["event"] == "admitted"}
    solved, first_score = {}, {}
    for row in progress:
        for name, niche in row["niches"].items():
            first_score.setdefault(name, row["iteration"])
            if niche["solved_at"] >= 0 and name not in solved:
                solved[name] = niche["solved_at"]
    # A niche's config is known from its admission; the starting niche is flat.
    configs = {name: e["config"] for name, e in admitted.items()}
    start = next(iter(progress[0]["niches"]))
    configs.setdefault(start, {})
    levels = {}
    for name in solved:
        level = difficulty(configs.get(name, {}))
        levels[level] = levels.get(level, 0) + 1
    attempts = [e for e in events if e["event"] == "transfer_attempt"]
    accepted = [e for e in events if e["event"] == "transfer_accepted"]
    best = max((n["score"] for row in progress for n in row["niches"].values()), default=None)
    return dict(
        run=folder.name, iteration=last["iteration"] + 1, active=last["active"], archive=last["archive"],
        rollouts=last["rollouts"], control_steps=last["steps"], wall_seconds=last.get("wall_seconds"),
        best_score=best, admitted=len(admitted), solved=len(solved), solved_by_difficulty=levels,
        solved_niches={name: dict(at=at, born=first_score.get(name), level=difficulty(configs.get(name, {})),
                                  config=configs.get(name, {})) for name, at in solved.items()},
        transfer_attempts=len(attempts), transfer_accepted=len(accepted),
        proposal_wins=sum(1 for e in accepted if str(e.get("donor", "")).endswith(":proposal")),
        current={name: n for name, n in last["niches"].items()},
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    runs = [r for r in (summarize_run(p) for p in sorted(args.root.iterdir()) if p.is_dir()) if r]
    probes = sample_probes(random.Random(1901017530))
    for r in runs:
        r["coverage"] = coverage([n["config"] for n in r["solved_niches"].values()], probes)
    for r in runs:
        print(f"{r['run']:28s} it={r['iteration']:5d} active={r['active']:2d} admitted={r['admitted']:3d} "
              f"solved={r['solved']:3d} {r['solved_by_difficulty']} best={r['best_score']:8.1f} "
              f"steps={r['control_steps']/1e6:8.1f}M accepted={r['transfer_accepted']}/{r['transfer_attempts']}")
    if args.output:
        args.output.write_text(json.dumps(runs, indent=2))


if __name__ == "__main__":
    main()
