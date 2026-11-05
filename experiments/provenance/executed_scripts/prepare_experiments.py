#!/usr/bin/env python3
"""Freeze analytic geometry and print a physics experiment matrix; never run it.

The exact build_course_profile AST is loaded from the local task source. This
avoids importing/launching IsaacLab and avoids a separately maintained copy of
the geometry algorithm. Only its required math/numpy/type names are supplied.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
from collections.abc import Mapping
from pathlib import Path
import shlex
import sys

import numpy as np

WORKSPACE = Path(__file__).resolve().parents[2]
PROJECT = WORKSPACE / "poet-reproduction"
TASK_SOURCE = PROJECT / "isaaclab_port" / "env.py"
COURSE = {"course_length": 30.0, "tile_length": 0.4, "flat_margin": 4.0}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def load_profile_builder():
    source = TASK_SOURCE.read_text()
    node = next(item for item in ast.parse(source).body
                if isinstance(item, ast.FunctionDef) and item.name == "build_course_profile")
    namespace = {"np": np, "math": math, "Mapping": Mapping}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(TASK_SOURCE), "exec"), namespace)
    return namespace["build_course_profile"], hashlib.sha256(ast.get_source_segment(source, node).encode()).hexdigest()


def geometry_record(profile):
    values = [float(value) if np.isfinite(value) else None for value in profile]
    geometry = {"course": COURSE, "lane_width": 2.0, "floor_thickness": 2.0,
                "tile_top_heights_m": values}
    return geometry, hashlib.sha256(canonical(geometry)).hexdigest()


def build_bank():
    builder, function_sha = load_profile_builder()
    specs = [
        ("flat_control", 0.0, 0.0, 0.0),
        ("rough_low", 0.12, 0.0, 0.0), ("rough_mid", 0.30, 0.0, 0.0), ("rough_high", 0.48, 0.0, 0.0),
        ("stump_low", 0.0, 0.2, 0.0), ("stump_mid", 0.0, 0.4, 0.0), ("stump_high", 0.0, 0.6, 0.0),
        ("gap_low", 0.0, 0.0, 0.4), ("gap_mid", 0.0, 0.0, 0.8), ("gap_high", 0.0, 0.0, 1.2),
        ("mixed_low", 0.12, 0.2, 0.4), ("mixed_mid", 0.30, 0.4, 0.8), ("mixed_high", 0.48, 0.6, 1.2),
    ]
    candidates = []
    for index, (name, roughness, stump_height, gap_width) in enumerate(specs):
        candidates.append((name, {"roughness": roughness, "stump_height": stump_height,
                                  "gap_width": gap_width, "seed": 4_000_000_000 + index}))
    # Different seeds do not imply different geometry: flat and pure-gap
    # profiles ignore RNG in their final geometry. Exercise real deduplication.
    for name, terrain in list(candidates):
        if name == "flat_control" or name.startswith("gap_"):
            candidates.append((name + "_duplicate_seed", {**terrain, "seed": terrain["seed"] + 100}))
    seen, terrains, removed = {}, [], []
    for name, terrain in candidates:
        geometry, digest = geometry_record(builder(terrain, terrain["seed"], **COURSE))
        if digest in seen:
            removed.append({"candidate": name, "same_geometry_as": seen[digest], "geometry_sha256": digest})
            continue
        seen[digest] = name
        terrains.append({"id": name, "role": "control" if name == "flat_control" else "evaluation",
                         "terrain": terrain, "geometry_sha256": digest, **geometry})
    assert len(terrains) == 13 and len(removed) == 4
    assert len({item["geometry_sha256"] for item in terrains}) == len(terrains)
    bank = {
        "schema_version": 1, "status": "frozen_configuration_not_evaluated",
        "team": 33, "paper_id": 44,
        "source": {"file": "poet-reproduction/isaaclab_port/env.py",
                   "env_sha256": hashlib.sha256(TASK_SOURCE.read_bytes()).hexdigest(),
                   "build_course_profile_sha256": function_sha, "numpy_version": np.__version__},
        "evaluation_count": 12, "control_count": 1,
        "definition": "Predeclared geometric task bank; no claim of solvability or measured performance.",
        "heldout_limit": "Seeds differ from planned training seeds and mutation RNG range, but geometric overlap with training must still be audited. Flat is an explicit training-overlap control.",
        "deduplication": "Exact SHA256 of task dimensions and actual tile-top geometry; null denotes a missing floor tile.",
        "terrains": terrains, "removed_duplicates": removed,
    }
    bank["bank_sha256"] = hashlib.sha256(canonical(bank)).hexdigest()
    return bank


def build_matrix(args, bank):
    rows = []
    for seed in (24582922, 24582923, 24582924):
        for mode in ("full", "topk", "no_transfer"):
            output = args.output.resolve() / "planned_runs" / f"{mode}_seed_{seed}"
            command = [str(args.python.expanduser().resolve()), str(PROJECT / "train_isaaclab.py"),
                       "--mode", "train", "--device", args.device, "--solver", args.solver,
                       "--substeps", str(args.substeps), "--num_envs", "32", "--pairs", "32",
                       "--episode_steps", "2000", "--iterations", str(args.iterations),
                       "--max_seconds", str(args.max_seconds), "--seed", str(seed), "--output", str(output)]
            command += (["--no_transfer"] if mode == "no_transfer" else ["--transfer_mode", mode])
            if mode == "topk":
                command += ["--transfer_top_k", "1"]
            if args.allow_local_beta:
                command += ["--allow_local_beta"]
            rows.append({"mode": mode, "seed": seed, "status": "not_run", "output": str(output),
                         "argv": command, "shell_command": shlex.join(command)})
    return {
        "schema_version": 1, "status": "proposed_protocol_not_executed",
        "team": 33, "paper_id": 44, "evaluation_bank_sha256": bank["bank_sha256"],
        "device": args.device, "solver": args.solver, "substeps": args.substeps,
        "allow_local_beta": args.allow_local_beta,
        "budget": {"max_iterations": args.iterations, "soft_max_seconds": args.max_seconds,
                   "strict_simulated_world_step_budget": None,
                   "limitation": "Current CLI checks wall time after a full iteration; initialization counts toward elapsed time but the first iteration still runs. Equal caps do not guarantee equal actual time or world steps."},
        "prerequisites": ["Validate requested simulator/device with current source and fingerprint.",
                          "Use a separate pilot to select a feasible budget, then lock it for all nine runs.",
                          "Demonstrate reproduction and at least three active environments before claiming screening was exercised.",
                          "Add checkpoint snapshots/heldout evaluation runner before claiming an evaluation learning curve."],
        "runs": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--python", type=Path, default=Path.home() / "env_isaaclab" / "bin" / "python")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--solver", choices=("featherstone", "mjwarp"), default="featherstone")
    parser.add_argument("--substeps", type=int, default=16)
    parser.add_argument("--iterations", type=int, default=2000)
    parser.add_argument("--max-seconds", type=float, default=3600)
    parser.add_argument("--allow-local-beta", action="store_true")
    args = parser.parse_args()
    if min(args.substeps, args.iterations, args.max_seconds) <= 0:
        parser.error("substeps, iterations and max-seconds must be positive")
    if args.device == "cpu" and args.solver == "mjwarp":
        parser.error("The current runner supports CPU with featherstone only")
    bank = build_bank()
    matrix = build_matrix(args, bank)
    args.output.mkdir(parents=True, exist_ok=True)
    for name, value in (("heldout_terrains.json", bank), ("launch_matrix.json", matrix)):
        (args.output / name).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    commands = [row["shell_command"] for row in matrix["runs"]]
    shell = ["#!/usr/bin/env bash", "set -euo pipefail",
             "# This file describes pending runs; generation does not execute physics.",
             'if [[ "${1:-}" != "--execute" ]]; then',
             "  cat <<'POET_PLANNED_COMMANDS'", *commands, "POET_PLANNED_COMMANDS",
             '  echo "Pending protocol only. Pass --execute after validation and budget review to run sequentially."',
             "  exit 0", "fi", "", *commands, ""]
    (args.output / "run_matrix.sh").write_text("\n".join(shell))
    print(json.dumps({"output": str(args.output), "evaluation_terrains": bank["evaluation_count"],
                      "controls": bank["control_count"], "duplicates_removed": len(bank["removed_duplicates"]),
                      "commands_generated_not_run": len(commands), "bank_sha256": bank["bank_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
