"""Pure analytical tests of the physical runner's protocol and accounting.

These tests do not run physics and are not physical-performance evidence.
"""
import io
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

spec = importlib.util.spec_from_file_location("physics_protocol", Path(__file__).with_name("run_physics_comparison.py"))
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
aggregate_spec = importlib.util.spec_from_file_location("aggregate_physics", Path(__file__).with_name("aggregate_physics_results.py"))
aggregator = importlib.util.module_from_spec(aggregate_spec)
aggregate_spec.loader.exec_module(aggregator)


class AnalyticEvaluator:
    def __init__(self):
        self.counts = dict.fromkeys(runner.METRIC_KEYS, 0)
        self.last_episode_lengths = []
        self.last_episode_successes = []

    def __call__(self, terrain, matrix):
        n = len(matrix)
        self.counts["requested_episodes"] += n
        self.counts["first_episode_steps"] += 3 * n
        self.counts["simulated_world_steps"] += 12 * ((n + 3) // 4)
        self.last_episode_lengths = [3] * n
        self.last_episode_successes = [False] * n
        return 200. - np.square(matrix - terrain.roughness).mean(axis=1)

    def metrics(self):
        return self.counts.copy()


TRAIN = [runner.Terrain(roughness=.02 * i, seed=i) for i in range(4)]
TEST = [(str(i), runner.Terrain(roughness=.1 + i * .02, seed=100 + i)) for i in range(4)]


class PhysicalProtocolTests(unittest.TestCase):
    def test_modes_share_identical_but_independent_initial_policies(self):
        states = []
        for mode in ("full", "topk", "no_transfer"):
            poet, digest = runner.fresh_population(123, mode, 2, TRAIN, AnalyticEvaluator(), None)
            states.append((poet, digest))
        self.assertEqual(len({digest for _, digest in states}), 1)
        for index in range(4):
            for poet, _ in states[1:]:
                np.testing.assert_array_equal(states[0][0].active[index].theta, poet.active[index].theta)
        self.assertFalse(np.array_equal(states[0][0].active[0].theta, states[0][0].active[1].theta))
        self.assertFalse(np.shares_memory(states[0][0].active[0].theta, states[0][0].active[1].theta))

    def test_heldout_only_evaluates_and_preserves_algorithm_rng_and_state(self):
        meter = runner.MeteredEvaluator(AnalyticEvaluator(), io.StringIO())
        poet, before = runner.fresh_population(123, "topk", 2, TRAIN, meter, None)
        rng_before = json.dumps(poet.rng.bit_generator.state, sort_keys=True)
        results = runner.evaluate_library(poet, meter, TEST)
        self.assertEqual(before, runner.hash_array(np.stack([p.theta for p in poet.active])))
        self.assertEqual(rng_before, json.dumps(poet.rng.bit_generator.state, sort_keys=True))
        self.assertTrue(all(p.optimizer.t == 0 for p in poet.active))
        self.assertEqual(poet.evaluations, 0)
        self.assertEqual(meter.totals["train"]["requested_episodes"], 0)
        self.assertEqual(meter.totals["eval"]["requested_episodes"], 16)
        self.assertEqual(results["task_count"], 4)
        self.assertEqual(results["tasks_reached_goal"], 0)
        self.assertEqual(meter.phase, "train")

    def test_fifth_iteration_transfer_accounting_and_snapshots(self):
        args = SimpleNamespace(pairs=2, episode_steps=10, num_envs=4,
                               world_step_budget=0, max_seconds=0, iterations=5)
        expected = {"full": (176, 12), "topk": (136, 4), "no_transfer": (104, 0)}
        for mode, (episodes, proposals) in expected.items():
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                destination = Path(directory) / "run"
                result = runner.run_condition(123, mode, args, TRAIN, TEST,
                                              AnalyticEvaluator(), destination, {})
                self.assertEqual(result["iterations_completed"], 5)
                self.assertEqual(result["train_metrics"]["requested_episodes"], episodes)
                self.assertEqual(result["algorithm_policy_evaluations"], episodes)
                self.assertEqual(result["eval_metrics"]["requested_episodes"], 32)
                self.assertEqual(result["transfer_summary"]["adaptation_proposals"], proposals)
                self.assertEqual(result["active_tasks"], 4)
                self.assertEqual(result["archive_size"], 4)
                self.assertFalse(result["reproduction_enabled"])
                self.assertEqual([item["iteration"] for item in result["checkpoints"]], [0, 5])
                events = [json.loads(line) for line in (destination / "events.jsonl").read_text().splitlines()]
                self.assertTrue(all(event["iteration"] == 5 for event in events))
                self.assertTrue((destination / "checkpoint_000.json").is_file())
                self.assertTrue((destination / "checkpoint_005.json").is_file())

    def test_budget_stops_between_iterations_and_reports_actual_overshoot(self):
        args = SimpleNamespace(pairs=2, episode_steps=10, num_envs=4,
                               world_step_budget=120, max_seconds=0, iterations=20)
        with tempfile.TemporaryDirectory() as directory:
            result = runner.run_condition(123, "full", args, TRAIN, TEST,
                                          AnalyticEvaluator(), Path(directory) / "run", {})
        self.assertEqual(result["iterations_completed"], 1)
        self.assertEqual(result["train_metrics"]["simulated_world_steps"], 144)
        self.assertEqual(result["training_world_step_overshoot"], 24)
        self.assertEqual(result["stop_reason"], "training_world_step_budget_at_iteration_boundary")
        self.assertEqual([item["iteration"] for item in result["checkpoints"]], [0, 1])

    def test_aggregation_checks_raw_counts_and_checkpoint_policy_identity(self):
        args = SimpleNamespace(pairs=2, episode_steps=10, num_envs=4,
                               world_step_budget=0, max_seconds=0, iterations=1)
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory) / "full_seed_123"
            runner.run_condition(123, "full", args, TRAIN, TEST,
                                 AnalyticEvaluator(), folder, {"sha256": "test-fixture"})
            result = aggregator.aggregate([Path(directory)])
            self.assertEqual(result["run_count"], 1)
            self.assertFalse(result["complete_three_by_three_matrix"])
            self.assertIsNone(result["matched_checkpoint_iteration"])
            self.assertEqual(result["groups"], {})
            self.assertEqual(result["comparison_runs"], [])
            self.assertEqual(result["totals"]["train"]["requested_episodes"], 24)
            checkpoint_path = folder / "checkpoint_001.json"
            data = json.loads(checkpoint_path.read_text())
            data["pairs"][0]["theta"][0] += 1.
            checkpoint_path.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, "Checkpoint/evaluation policy mismatch"):
                aggregator.aggregate([Path(directory)])

    def test_aggregation_rejects_summary_count_not_supported_by_raw_log(self):
        args = SimpleNamespace(pairs=2, episode_steps=10, num_envs=4,
                               world_step_budget=0, max_seconds=0, iterations=1)
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory) / "full_seed_123"
            runner.run_condition(123, "full", args, TRAIN, TEST,
                                 AnalyticEvaluator(), folder, {"sha256": "test-fixture"})
            path = folder / "summary.json"
            data = json.loads(path.read_text())
            data["train_metrics"]["requested_episodes"] += 1
            path.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, "Raw accounting mismatch"):
                aggregator.aggregate([Path(directory)])

    def test_primary_selection_requires_full_matrix_and_matches_actual_snapshot_sets(self):
        runs = []
        for seed in (1, 2, 3):
            for mode in ("full", "topk", "no_transfer"):
                indices = [0, 5, 10, 13] if seed == 1 and mode == "full" else [0, 5, 10, 15, 20]
                snapshots = [{"iteration": index, "checkpoint": f"checkpoint_{index:03d}.json",
                              "checkpoint_sha256": str(index), "wall_seconds": index * 2,
                              "train_metrics": {"simulated_world_steps": index * 100},
                              "eval_metrics": {"simulated_world_steps": index * 20},
                              "evaluation": {"mean_best_library_return": index,
                                             "tasks": [{"max_x_by_policy": [2., 2.5, 3., 4.1]}]}}
                             for index in indices]
                runs.append({"seed": seed, "mode": mode, "iterations_completed": max(indices),
                             "initial_evaluation": snapshots[0]["evaluation"],
                             "checkpoints": snapshots, "source_summary": "analytical_fixture",
                             "target_training_world_steps": 900})
        index, selected, status = aggregator.primary_comparison(runs)
        self.assertEqual((index, status), (10, "ready"))
        self.assertEqual(len(selected), 9)
        self.assertEqual(selected[0]["actual_terminal_iteration"], 13)
        self.assertEqual(selected[1]["actual_terminal_iteration"], 20)
        for item in selected:
            self.assertEqual(item["evaluation"]["mean_best_library_return"], 10)
            self.assertEqual(item["train_metrics"]["simulated_world_steps"], 1000)
            self.assertEqual(item["eval_metrics"]["simulated_world_steps"], 200)
            self.assertEqual(item["training_world_step_overshoot"], 100)
            self.assertEqual(item["heldout_max_x"], 4.1)
        self.assertEqual(aggregator.primary_comparison(runs[:-1]),
                         (None, [], "pending_complete_three_by_three_matrix"))
        # An irregular terminal checkpoint cannot be inferred from min(final).
        runs[0]["checkpoints"] = [item for item in runs[0]["checkpoints"] if item["iteration"] != 10]
        self.assertEqual(aggregator.primary_comparison(runs)[0], 5)
        runs[0]["checkpoints"] = [runs[0]["checkpoints"][0]]
        self.assertEqual(aggregator.primary_comparison(runs),
                         (None, [], "no_common_checkpoint_at_or_after_first_transfer"))

    def test_aggregated_groups_use_common_checkpoint_costs_and_keep_terminal_records(self):
        with tempfile.TemporaryDirectory() as directory:
            for seed in (11, 12, 13):
                for mode in ("full", "topk", "no_transfer"):
                    iterations = 6 if seed == 11 and mode == "full" else 10
                    args = SimpleNamespace(pairs=2, episode_steps=10, num_envs=4,
                                           world_step_budget=0, max_seconds=0, iterations=iterations)
                    runner.run_condition(seed, mode, args, TRAIN, TEST, AnalyticEvaluator(),
                                         Path(directory) / f"{mode}_{seed}", {"sha256": "test-fixture"})
            result = aggregator.aggregate([Path(directory)])
            self.assertTrue(result["complete_three_by_three_matrix"])
            self.assertEqual(result["matched_checkpoint_iteration"], 5)
            self.assertEqual(len(result["comparison_runs"]), 9)
            self.assertEqual(result["groups"]["full"]["iterations_by_seed"], [5, 5, 5])
            self.assertEqual(result["groups"]["full"]["actual_terminal_iterations_by_seed"], [6, 10, 10])
            originals = {(row["seed"], row["mode"]): row for row in result["runs"]}
            for primary in result["comparison_runs"]:
                original = originals[(primary["seed"], primary["mode"])]
                snapshot = next(row for row in original["checkpoints"] if row["iteration"] == 5)
                self.assertEqual(primary["evaluation"], snapshot["evaluation"])
                self.assertEqual(primary["train_metrics"], snapshot["train_metrics"])
                self.assertLess(primary["train_metrics"]["requested_episodes"],
                                original["train_metrics"]["requested_episodes"])
            self.assertGreater(result["totals"]["train"]["requested_episodes"],
                               sum(row["train_metrics"]["requested_episodes"] for row in result["comparison_runs"]))


if __name__ == "__main__":
    unittest.main()
