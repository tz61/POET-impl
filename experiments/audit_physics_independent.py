#!/usr/bin/env python3
"""Read-only independent audit of the completed fixed-task physics conditions.

This does not simulate, change a checkpoint, or impute missing conditions.
Prints a JSON report; --require-complete also requires all nine main conditions.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import struct


def read(path):
    return json.loads(path.read_text())


def digest(matrix):
    return hashlib.sha256(b"".join(struct.pack("<d", x) for row in matrix for x in row)).hexdigest()


def check(condition, message):
    if not condition:
        raise AssertionError(message)


def audit_run(path):
    folder = path.parent
    result = read(path)
    settings = read(folder / "settings.json")
    cfg = settings["settings"]
    protocol = read(folder.parent / "protocol.json")
    fingerprint = read(folder / "task_fingerprint.json")
    canonical = json.dumps(fingerprint["payload"], sort_keys=True, separators=(",", ":"),
                           ensure_ascii=True, allow_nan=False).encode()
    check(hashlib.sha256(canonical).hexdigest() == fingerprint["sha256"],
          "task fingerprint payload checksum mismatch")
    frozen_protocol = fingerprint["payload"]["task_config"].get("fixed_task_protocol")
    if frozen_protocol is not None:
        check(all(protocol[key] == value for key, value in frozen_protocol.items()),
              "batch protocol differs from frozen task fingerprint")
    project = Path(__file__).resolve().parents[1] / "isaaclab_port"
    source_paths = {"algorithm": project / "core.py", "evaluator": project / "runtime.py",
                    "task": project / "env.py", "robot_usd": project / "assets/planar_biped.usda"}
    for name, source in source_paths.items():
        check(hashlib.sha256(source.read_bytes()).hexdigest()
              == fingerprint["payload"]["source_sha256"][name],
              f"packaged {name} source differs from executed fingerprint")
    check(result["status"] == "completed", "condition not complete")
    check(result["active_tasks"] == result["archive_size"] == 4, "population is not four fixed tasks")
    check(not result["reproduction_enabled"], "unexpected reproduction")
    mode, iterations, pairs = result["mode"], result["iterations_completed"], cfg["pairs"]
    transfers = iterations // 5
    proposals_per_event = {"full": 12, "topk": 4, "no_transfer": 0}[mode]
    direct_per_event = 0 if mode == "no_transfer" else 12
    expected_train_evals = 4 + iterations * 4 * (2 * pairs + 1)
    expected_train_evals += transfers * (direct_per_event + proposals_per_event * (2 * pairs + 1))
    check(result["algorithm_policy_evaluations"] == expected_train_evals, "ES/transfer evaluation oracle mismatch")
    records = [json.loads(line) for line in (folder / "evaluations.jsonl").read_text().splitlines()]
    for phase in ("train", "eval"):
        rows = [row for row in records if row["phase"] == phase]
        reported = result[phase + "_metrics"]
        for key in ("requested_episodes", "first_episode_steps", "simulated_world_steps"):
            check(sum(row[key] for row in rows) == reported[key], f"{phase} {key} raw sum mismatch")
        check(len(rows) == reported["evaluator_calls"], f"{phase} evaluator call count mismatch")
    check(result["train_metrics"]["requested_episodes"] == expected_train_evals, "train episode mismatch")
    for row in records:
        n = row["policies"]
        check(row["requested_episodes"] == n, "policy versus episode count")
        check(len(row["returns"]) == len(row["episode_lengths"]) == len(row["goal_successes"]) == n,
              "per-policy array length mismatch")
        check(all(math.isfinite(x) for x in row["returns"]), "non-finite return")
        check(all(0 < x <= settings["episode_steps"] for x in row["episode_lengths"]), "episode exceeds horizon")
        check(sum(row["episode_lengths"]) == row["first_episode_steps"], "first-episode step sum mismatch")
        check(row["simulated_world_steps"] >= row["first_episode_steps"], "actual work below scored work")
        check(row["simulated_world_steps"] % settings["num_envs"] == 0, "world steps not batched")
        if "terminal_x" in row and "max_x" in row:
            check(len(row["terminal_x"]) == len(row["max_x"]) == n, "x diagnostic array mismatch")
            for terminal, maximum, success in zip(row["terminal_x"], row["max_x"], row["goal_successes"]):
                check(math.isfinite(terminal) and math.isfinite(maximum), "non-finite position")
                check(maximum + 1e-6 >= max(2.0, terminal), "max-x tracker inconsistent")
                check(not success or terminal >= 30.0, "success below finish line")
    events = [json.loads(line) for line in (folder / "events.jsonl").read_text().splitlines()]
    attempts = [event for event in events if event["event"] == "transfer_attempt"]
    accepted = [event for event in events if event["event"] == "transfer_accepted"]
    check(len(attempts) == transfers * direct_per_event, "direct attempt count mismatch")
    check(sum(x["proposal_selected"] for x in attempts) == transfers * proposals_per_event,
          "adaptation proposal count mismatch")
    by_target = {}
    for event in attempts:
        check(event["iteration"] % 5 == 0 and 5 <= event["iteration"] <= iterations, "transfer schedule mismatch")
        check(event["target"] != event["donor"], "self-transfer")
        check((event["proposal_score"] is not None) == event["proposal_selected"], "proposal selection/score mismatch")
        if mode == "topk":
            check(event["proposal_selected"] == (event["direct_rank"] == 1), "Top-1 did not screen by direct rank")
        by_target.setdefault((event["iteration"], event["target"]), []).append(event)
    for group in by_target.values():
        check(sorted(x["direct_rank"] for x in group) == [1, 2, 3], "missing/repeated donor ranks")
        ranked = sorted(group, key=lambda x: (-x["direct_score"], x["donor"]))
        check([x["direct_rank"] for x in ranked] == [1, 2, 3], "direct-score rank/tiebreak mismatch")
    for event in accepted:
        check(event["after"] > event["before"], "accepted non-improving transfer")
    check(result["transfer_summary"]["accepted_targets"] == len(accepted), "accept count mismatch")
    snapshots = result["checkpoints"]
    expected_iterations = sorted(set([0, iterations] + list(range(5, iterations + 1, 5))))
    check([item["iteration"] for item in snapshots] == expected_iterations, "checkpoint schedule mismatch")
    check(result["eval_metrics"]["requested_episodes"] == 16 * len(snapshots), "heldout evaluation count mismatch")
    for item in snapshots:
        checkpoint = read(folder / item["checkpoint"])
        check(checkpoint["iteration"] == item["iteration"], "checkpoint iteration mismatch")
        check(checkpoint["evaluations"] == item["train_metrics"]["requested_episodes"], "checkpoint train count mismatch")
        matrix = [pair["theta"] for pair in checkpoint["pairs"]]
        check(len(matrix) == 4 and all(len(row) == 2804 for row in matrix), "policy shape mismatch")
        evaluation = item["evaluation"]
        check(digest(matrix) == evaluation["policy_matrix_sha256"], "evaluated library differs from checkpoint")
        tasks = evaluation["tasks"]
        check(len(tasks) == 4, "heldout task count mismatch")
        check([t["task_id"] for t in tasks] == [t["task_id"] for t in protocol["heldout_tasks"]], "heldout identity mismatch")
        for task in tasks:
            check(task["best_library_return"] == max(task["returns_by_policy"]), "library best return mismatch")
            check(task["any_policy_reached_goal"] == any(task["goal_successes_by_policy"]), "library success mismatch")
        mean = sum(t["best_library_return"] for t in tasks) / 4
        check(abs(mean - evaluation["mean_best_library_return"]) < 1e-10, "heldout mean mismatch")
        check(evaluation["tasks_reached_goal"] == sum(t["any_policy_reached_goal"] for t in tasks), "heldout coverage mismatch")
    initial = read(folder / snapshots[0]["checkpoint"])
    initial_hash = digest([pair["theta"] for pair in initial["pairs"]])
    check(initial_hash == result["initial_policy_sha256"] == settings["initial_policy_sha256"], "initial hash mismatch")
    training_hashes = {x["profile_sha256"] for x in protocol["training_tasks"]}
    heldout_hashes = {x["profile_sha256"] for x in protocol["heldout_tasks"]}
    check(len(training_hashes) == len(heldout_hashes) == 4, "duplicate task geometry")
    check(not (training_hashes & heldout_hashes), "heldout geometry leakage")
    maxima = [x for row in records for x in row.get("max_x", [])]
    prefix_rows = [row for row in records if row["phase"] == "train"][:4 + 8 * min(iterations, 5)]
    prefix = [{key: row[key] for key in ("terrain", "policy_matrix_sha256", "returns",
                                        "episode_lengths", "goal_successes")}
              for row in prefix_rows]
    return {"seed": result["seed"], "mode": mode, "iterations": iterations,
            "transfer_events": transfers if mode != "no_transfer" else 0,
            "train_policy_evaluations": expected_train_evals,
            "train_world_steps": result["train_metrics"]["simulated_world_steps"],
            "eval_world_steps": result["eval_metrics"]["simulated_world_steps"],
            "initial_policy_sha256": initial_hash,
            "pre_transfer_prefix_sha256": hashlib.sha256(json.dumps(prefix, sort_keys=True).encode()).hexdigest(),
            "fingerprint": fingerprint["sha256"],
            "source_hashes_match": True,
            "raw_evaluations_sha256": hashlib.sha256((folder / "evaluations.jsonl").read_bytes()).hexdigest(),
            "checkpoints": [{"iteration": item["iteration"],
                             "checkpoint_sha256": hashlib.sha256((folder / item["checkpoint"]).read_bytes()).hexdigest(),
                             "train_world_steps": item["train_metrics"]["simulated_world_steps"],
                             "mean_best_library_return": item["evaluation"]["mean_best_library_return"]}
                            for item in snapshots],
            "initial_heldout_returns": [x["returns_by_policy"] for x in result["initial_evaluation"]["tasks"]],
            "final_mean_best_library_return": result["final_evaluation"]["mean_best_library_return"],
            "final_goal_tasks": result["final_evaluation"]["tasks_reached_goal"],
            "max_recorded_x": max(maxima) if maxima else None,
            "summary_path": str(path), "summary_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    rows, errors = [], []
    # selected_main contains only the predeclared canonical condition links;
    # never collect unused conditions from the original sequential chains.
    pattern = "*/summary.json" if args.root.name == "selected_main" else "main_seed_*/*/summary.json"
    for path in sorted(args.root.glob(pattern)):
        try:
            rows.append(audit_run(path))
        except Exception as exc:
            errors.append({"path": str(path), "error": repr(exc)})
    keys = {(row["seed"], row["mode"]) for row in rows}
    expected = {(seed, mode) for seed in (24582922, 24582923, 24582924) for mode in ("full", "topk", "no_transfer")}
    matched_checkpoint = None
    selection_hash = None
    replacement_count = None
    replacement_checks = []
    try:
        manifest_path = args.root / "selection_manifest.json"
        if manifest_path.exists():
            manifest = read(manifest_path)
            selection_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
            selection = manifest["selection"]
            check({(item["seed"], item["mode"]) for item in selection} == expected
                  and len(selection) == len(expected), "canonical selection is not a unique 3x3 matrix")
            for item in selection:
                selected_path = args.root / f'{item["mode"]}_seed_{item["seed"]}'
                if selected_path.is_symlink():
                    check(selected_path.resolve() == Path(item["condition"]).resolve(),
                          "canonical link differs from predeclared selection")
            replacement_count = len(manifest.get("execution_replacements", []))
            for item in manifest.get("execution_replacements", []):
                original = Path(item["interrupted_condition"])
                # Both source ZIP and workspace preserve the original batch names.
                old = args.root.parent / original.parent.name / original.name / "evaluations.jsonl"
                new = args.root / f'{item["mode"]}_seed_{item["seed"]}' / "evaluations.jsonl"
                if not old.exists() or not new.exists():
                    check(not args.require_complete, "interrupted/replacement evidence is missing")
                    continue
                old_rows = [json.loads(line) for line in old.read_text().splitlines()]
                new_rows = [json.loads(line) for line in new.read_text().splitlines()]
                count = min(len(old_rows), len(new_rows))
                fields = ("phase", "terrain", "policy_matrix_sha256", "returns",
                          "episode_lengths", "goal_successes")
                check(all(all(left[key] == right[key] for key in fields)
                          for left, right in zip(old_rows[:count], new_rows[:count])),
                      "replacement physical trace differs from interrupted attempt")
                check(not args.require_complete or len(new_rows) >= len(old_rows),
                      "completed replacement does not cover preserved interrupted prefix")
                replacement_checks.append({"seed": item["seed"], "mode": item["mode"],
                    "matched_evaluator_calls": count,
                    "all_preserved_calls_replayed": len(new_rows) >= len(old_rows),
                    "interrupted_raw_sha256": hashlib.sha256(old.read_bytes()).hexdigest(),
                    "prefix_exactly_equal": True})
        check(len(keys) == len(rows), "duplicate seed/mode")
        check(len({row["fingerprint"] for row in rows}) <= 1, "different task fingerprints")
        for seed in sorted({row["seed"] for row in rows}):
            selected = [row for row in rows if row["seed"] == seed]
            check(len({row["initial_policy_sha256"] for row in selected}) == 1, "unpaired initial policies")
            check(all(row["initial_heldout_returns"] == selected[0]["initial_heldout_returns"] for row in selected),
                  "initial physical rollouts differ across paired modes")
            completed_prefixes = [row["pre_transfer_prefix_sha256"] for row in selected if row["iterations"] >= 5]
            check(len(set(completed_prefixes)) <= 1, "physical training differs before the first transfer")
        if args.require_complete:
            check(keys == expected, "incomplete 3x3 experiment")
            check(all(row["iterations"] >= 5 for row in rows), "condition did not reach first transfer")
        if keys == expected:
            common = set.intersection(*({item["iteration"] for item in row["checkpoints"]} for row in rows))
            eligible = [iteration for iteration in common if iteration >= 5]
            check(bool(eligible), "no common measured checkpoint at or after first transfer")
            matched_checkpoint = max(eligible)
    except Exception as exc:
        errors.append({"comparison": repr(exc)})
    print(json.dumps({"status": "failed" if errors else ("passed" if keys == expected else "partial"),
                      "completed_conditions": len(rows), "missing_conditions": sorted(expected - keys),
                      "matched_checkpoint_iteration": matched_checkpoint,
                      "selection_manifest_sha256": selection_hash,
                      "interrupted_attempt_replacements": replacement_count,
                      "replacement_trace_checks": replacement_checks,
                      "errors": errors, "runs": rows}, indent=2, allow_nan=False))
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
