#!/usr/bin/env python3
"""Validate raw completed runs and aggregate a fixed-task physical comparison.

No simulation is started. Raw runs are read-only; output must be a new path
unless --refresh is explicitly supplied for a live v2 summary.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import statistics

METRICS = ("requested_episodes", "first_episode_steps", "simulated_world_steps")


def read_json(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def transfer_summary(events, iteration):
    selected = [row for row in events if row["iteration"] <= iteration]
    attempts = [row for row in selected if row["event"] == "transfer_attempt"]
    target_scores = {}
    for row in attempts:
        target_scores.setdefault((row["iteration"], row["target"]), []).append(row["direct_score"])
    return {"direct_attempts": len(attempts),
            "adaptation_proposals": sum(row["proposal_selected"] for row in attempts),
            "accepted_targets": sum(row["event"] == "transfer_accepted" for row in selected),
            "target_events_with_direct_score_ties": sum(len(values) != len(set(values))
                                                       for values in target_scores.values())}


def run_environment_path(run):
    folder = Path(run["source_summary"]).parent
    portable = folder / "source_environment.json"
    return portable if portable.exists() else folder.parent / "environment.json"


def validated_run(path):
    path = Path(path)
    result = read_json(path)
    folder = path.parent
    if result.get("status") != "completed" or result["active_tasks"] != 4 or result["archive_size"] != 4:
        raise ValueError(f"Not a completed four-task condition: {path}")
    if result["reproduction_enabled"]:
        raise ValueError(f"Unexpected evolving-task result: {path}")
    raw = [json.loads(line) for line in (folder / "evaluations.jsonl").read_text().splitlines()]
    events = [json.loads(line) for line in (folder / "events.jsonl").read_text().splitlines()]
    if transfer_summary(events, result["iterations_completed"]) != result["transfer_summary"]:
        raise ValueError(f"Raw transfer event mismatch: {path}")
    for phase in ("train", "eval"):
        records = [row for row in raw if row["phase"] == phase]
        actual = {key: sum(row[key] for row in records) for key in METRICS}
        expected = result[f"{phase}_metrics"]
        if any(actual[key] != expected[key] for key in METRICS):
            raise ValueError(f"Raw accounting mismatch in {phase}: {path}")
        if sum(row["policies"] for row in records) != actual["requested_episodes"]:
            raise ValueError(f"Policy/episode count mismatch: {path}")
    if result["algorithm_policy_evaluations"] != result["train_metrics"]["requested_episodes"]:
        raise ValueError(f"Algorithm/evaluator count mismatch: {path}")
    checkpoints = []
    for item in result["checkpoints"]:
        checkpoint = folder / item["checkpoint"]
        data = read_json(checkpoint)
        if data["iteration"] != item["iteration"] or len(data["pairs"]) != 4:
            raise ValueError(f"Checkpoint state mismatch: {checkpoint}")
        # Core saves little-endian numeric doubles as JSON. Check that the
        # measured policy library is exactly this checkpoint's current policy.
        import struct
        payload = b"".join(struct.pack("<d", value) for pair in data["pairs"] for value in pair["theta"])
        if hashlib.sha256(payload).hexdigest() != item["evaluation"]["policy_matrix_sha256"]:
            raise ValueError(f"Checkpoint/evaluation policy mismatch: {checkpoint}")
        for phase in ("train", "eval"):
            metrics = item[f"{phase}_metrics"]
            prefix = [row for row in raw if row["phase"] == phase][:metrics["evaluator_calls"]]
            if any(sum(row[key] for row in prefix) != metrics[key] for key in METRICS):
                raise ValueError(f"Snapshot raw accounting mismatch: {checkpoint}")
        eval_prefix = [row for row in raw if row["phase"] == "eval"][:item["eval_metrics"]["evaluator_calls"]]
        for task, record in zip(item["evaluation"]["tasks"], eval_prefix[-4:]):
            if (record["policy_matrix_sha256"] != item["evaluation"]["policy_matrix_sha256"]
                    or record["terrain"] != task["terrain"]
                    or record["returns"] != task["returns_by_policy"]
                    or record["goal_successes"] != task["goal_successes_by_policy"]
                    or record.get("max_x") != task.get("max_x_by_policy")):
                raise ValueError(f"Snapshot evaluation differs from raw calls: {checkpoint}")
        checkpoints.append({**item, "checkpoint_sha256": sha(checkpoint),
                            "transfer_summary": transfer_summary(events, item["iteration"])})
    all_max_x = [x for row in raw for x in (row.get("max_x") or [])]
    return {**result, "checkpoints": checkpoints,
            "source_summary": str(path.resolve()), "source_summary_sha256": sha(path),
            "raw_evaluations_sha256": sha(folder / "evaluations.jsonl"),
            "task_fingerprint": read_json(folder / "task_fingerprint.json")["sha256"],
            "max_recorded_x": max(all_max_x) if all_max_x else None,
            "left_initial_flat_margin": bool(all_max_x and max(all_max_x) > 4.0)}


def primary_comparison(runs):
    """Select the largest observed checkpoint common to a complete 3x3 matrix.

    The selector depends only on saved iteration indices, never on outcomes.
    Terminal runs remain separate; a partial matrix has no primary comparison.
    """
    seeds = {row["seed"] for row in runs}
    expected = {(seed, mode) for seed in seeds for mode in ("full", "topk", "no_transfer")}
    complete = len(seeds) == 3 and len(runs) == 9 and {(row["seed"], row["mode"]) for row in runs} == expected
    if not complete:
        return None, [], "pending_complete_three_by_three_matrix"
    common = set.intersection(*({item["iteration"] for item in row["checkpoints"]} for row in runs))
    eligible = [value for value in common if value >= 5]
    if not eligible:
        return None, [], "no_common_checkpoint_at_or_after_first_transfer"
    iteration = max(eligible)
    selected = []
    for run in runs:
        item = next(item for item in run["checkpoints"] if item["iteration"] == iteration)
        all_x = [x for task in item["evaluation"]["tasks"] for x in (task.get("max_x_by_policy") or [])]
        selected.append({
            "seed": run["seed"], "mode": run["mode"], "iteration": iteration,
            "actual_terminal_iteration": run["iterations_completed"],
            "source_summary": run["source_summary"],
            "initial_evaluation": run["initial_evaluation"],
            "checkpoint": item["checkpoint"], "checkpoint_sha256": item["checkpoint_sha256"],
            "evaluation": item["evaluation"], "train_metrics": item["train_metrics"],
            "eval_metrics": item["eval_metrics"], "wall_seconds": item["wall_seconds"],
            "target_training_world_steps": run["target_training_world_steps"],
            "training_world_step_overshoot": max(0, item["train_metrics"]["simulated_world_steps"]
                                                  - run["target_training_world_steps"])
            if run["target_training_world_steps"] else 0,
            "heldout_max_x": max(all_x) if all_x else None,
            "heldout_left_initial_flat_margin": bool(all_x and max(all_x) > 4.0),
            "transfer_summary": item.get("transfer_summary"),
        })
    return iteration, selected, "ready"


def aggregate(inputs):
    files = sorted({path.resolve() for folder in inputs for path in Path(folder).glob("*/summary.json")})
    if not files:
        raise ValueError("No completed condition summary.json files found")
    runs = [validated_run(path) for path in files]
    keys = [(row["seed"], row["mode"]) for row in runs]
    if len(set(keys)) != len(keys):
        raise ValueError("Duplicate seed/mode conditions; do not mix pilot and main directories")
    fingerprints = {row["task_fingerprint"] for row in runs}
    if len(fingerprints) != 1:
        raise ValueError("Mixed task fingerprints; aggregate only the same frozen physical protocol")
    per_seed = {}
    for run in runs:
        per_seed.setdefault(run["seed"], []).append(run)
    for seed, conditions in per_seed.items():
        if len({row["initial_policy_sha256"] for row in conditions}) != 1:
            raise ValueError(f"Initial policies differ across modes for seed {seed}")
        initial = conditions[0]["initial_evaluation"]
        for row in conditions[1:]:
            other = row["initial_evaluation"]
            for left, right in zip(initial["tasks"], other["tasks"]):
                if left["task_id"] != right["task_id"]:
                    raise ValueError("Different heldout tasks")
                if any(abs(a-b) > 1e-7 for a,b in zip(left["returns_by_policy"], right["returns_by_policy"])):
                    raise ValueError(f"Initial physical evaluation differs for paired seed {seed}")
    matched_iteration, comparison_runs, primary_status = primary_comparison(runs)
    groups = {}
    for mode in ("full", "topk", "no_transfer"):
        selected = sorted((row for row in comparison_runs if row["mode"] == mode), key=lambda row: row["seed"])
        if not selected:
            continue
        final = [row["evaluation"]["mean_best_library_return"] for row in selected]
        initial = [row["initial_evaluation"]["mean_best_library_return"] for row in selected]
        groups[mode] = {
            "training_seeds": [row["seed"] for row in selected], "n_independent_seeds": len(selected),
            "initial_mean_best_library_return": statistics.mean(initial),
            "final_mean_best_library_return": statistics.mean(final),
            "final_seed_sample_std": statistics.stdev(final) if len(final) > 1 else None,
            "final_returns_by_seed": final,
            "iterations_by_seed": [row["iteration"] for row in selected],
            "actual_terminal_iterations_by_seed": [row["actual_terminal_iteration"] for row in selected],
            "goal_tasks_by_seed": [row["evaluation"]["tasks_reached_goal"] for row in selected],
            "training_world_steps_by_seed": [row["train_metrics"]["simulated_world_steps"] for row in selected],
            "evaluation_world_steps_by_seed": [row["eval_metrics"]["simulated_world_steps"] for row in selected],
            "wall_seconds_by_seed": [row["wall_seconds"] for row in selected],
            "budget_overshoot_by_seed": [row["training_world_step_overshoot"] for row in selected],
            "max_recorded_x_by_seed": [row["heldout_max_x"] for row in selected],
            "transfer_summaries_by_seed": [row["transfer_summary"] for row in selected],
        }
    # Symlinked selected_main may contain a copied environment snapshot from
    # another execution context. Attribute each condition to its real batch.
    environment_files = sorted({run_environment_path(row) for row in runs
                                if run_environment_path(row).exists()})
    protocols = [read_json(Path(folder) / "protocol.json") for folder in inputs if (Path(folder) / "protocol.json").exists()]
    selection_manifests = [Path(folder) / "selection_manifest.json" for folder in inputs
                           if (Path(folder) / "selection_manifest.json").exists()]
    return {
        "schema_version": 2, "evidence_kind": "actual_cpu_physics_fixed_task_control",
        "is_full_open_ended_poet": False, "run_count": len(runs),
        "complete_three_by_three_matrix": primary_status != "pending_complete_three_by_three_matrix",
        "primary_comparison_status": primary_status,
        "matched_checkpoint_iteration": matched_iteration,
        "group_field_semantics": "All groups fields including legacy final_* labels use the largest common saved checkpoint iteration >=5 across the completed 3-seed x 3-mode matrix; max_recorded_x_by_seed uses only the complete 4x4 heldout evaluations at that checkpoint. Groups is empty until this selection is possible. These are equal optimization rounds, not equal compute.",
        "all_runs_reached_first_transfer_iteration": all(row["iterations_completed"] >= 5 for row in runs),
        "task_fingerprint": next(iter(fingerprints)),
        "environments": [read_json(path) for path in environment_files],
        "environment_provenance": [{"source": str(path.resolve()), "sha256": sha(path),
                                    "conditions": [{"seed": row["seed"], "mode": row["mode"]}
                                                   for row in runs
                                                   if run_environment_path(row) == path]}
                                   for path in environment_files],
        "selection_manifests": [{"source": str(path.resolve()), "sha256": sha(path),
                                  "contents": read_json(path)} for path in selection_manifests],
        "protocol": protocols[0] if protocols else None,
        "statistical_unit": "training seed, not test task or library policy",
        "metric_definition": "Mean across four fixed test tasks of the best current-policy return among four evaluated policies; oracle library coverage, no adaptation.",
        "accounting": "World steps include active and padded simulator lanes; train/eval costs separate; wall time also includes setup and task reconstruction.",
        "totals_scope": "All completed work recorded by the selected canonical runs only. Failed/terminated attempts and work inside an unfinished evaluator call are not included in totals; see selection manifests and retained attempt directories.",
        "limitations": ["Fixed tasks, no environment generation or reproduction.",
                        "Small seed count; no general superiority or significance claim.",
                        "Iteration-boundary stopping gives different actual world-step totals; plot actual x-values.",
                        "Primary comparison matches saved optimization rounds; original terminal outcomes and total costs remain in runs and totals.",
                        "Totals do not estimate the entire project's compute, including terminated execution attempts.",
                        "Wall times were measured during concurrent shared-host execution; they are not isolated throughput benchmarks.",
                        "Hardware visibility can differ across launch contexts; all selected conditions explicitly requested CPU Featherstone. Environment snapshots are attributed to their actual batch directories.",
                        "Terminal runs.max_recorded_x includes only calls with position tracking, excluding multi-batch perturbation calls; primary heldout_max_x covers all 16 heldout cases.",
                        "Max-x beyond 4m only indicates leaving the initial flat margin, not clearing an obstacle.",
                        "Current installed simulator version is in environments; do not relabel beta CPU evidence as GA/GPU."],
        "groups": groups, "comparison_runs": comparison_runs, "runs": runs,
        "totals": {phase: {key: sum(row[f"{phase}_metrics"][key] for row in runs) for key in METRICS}
                   for phase in ("train", "eval")},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()
    if args.output.exists() and not args.refresh:
        parser.error("output already exists; use a new filename or explicitly --refresh")
    result = aggregate(args.inputs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    temporary.replace(args.output)
    print(json.dumps({"output": str(args.output), "run_count": result["run_count"],
                      "complete": result["complete_three_by_three_matrix"], "groups": result["groups"]}, indent=2))


if __name__ == "__main__":
    main()
