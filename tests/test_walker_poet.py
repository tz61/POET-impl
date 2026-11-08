"""Check the batched POET loop on analytic objectives; no simulator is involved."""
import json
import sys
from pathlib import Path
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "upstream"))
from isaaclab_port.walker.poet import CurriculumRun, Niche, Reproducer, Settings, WalkerPOET
from isaaclab_port.walker.terrain import EnvConfig
from run_local import reject_unbuildable, trim_logs
from train_walker import trim_log
sys.path.insert(0, str(ROOT / "tools"))
from make_targets import distance_from_flat, pick_targets
from compare_walker import at_budget, at_iteration, by_run, common_budget, list_checkpoints, report, run_name


class Result:
    def __init__(self, returns, finished=None):
        self.returns = np.asarray(returns, dtype=float)
        self.lengths = np.ones(len(self.returns), dtype=np.int64)
        self.finished = np.zeros(len(self.returns), dtype=bool) if finished is None else finished


def niche(roughness=0.0, seed=1):
    return Niche(EnvConfig(ground_roughness=roughness), seed)


class Recorder:
    """Return target - |theta| distance, and remember what was asked."""

    def __init__(self, fn):
        self.fn, self.calls = fn, []

    def __call__(self, tasks):
        self.calls.append(len(tasks))
        return Result([self.fn(config, theta) for config, _, theta in tasks])


def fast(**kw):
    defaults = dict(pairs=8, children=16, mc_lower=-1e9, mc_upper=1e9, reproduce_score=-1e9, noise=0.05,
                    eval_rollouts=1)
    defaults.update(kw)
    return Settings(**defaults)


class WalkerPoetTests(unittest.TestCase):
    def test_es_improves_an_analytic_objective(self):
        evaluate = Recorder(lambda c, theta: 10.0 - float(np.sum((theta - 0.3) ** 2)))
        poet = WalkerPOET(fast(pairs=32, reproduction=False, l2=0.0), evaluate, parameter_count=6)
        first = None
        for _ in range(150):
            poet.step()
            first = poet.active[0].score if first is None else first
        self.assertGreater(poet.active[0].score, first)
        self.assertAlmostEqual(poet.active[0].theta.mean(), 0.3, delta=0.1)

    def test_one_iteration_is_one_batch_with_scores(self):
        evaluate = Recorder(lambda c, theta: 0.0)
        poet = WalkerPOET(fast(reproduction=False), evaluate, parameter_count=4)
        poet.step()
        self.assertEqual(evaluate.calls, [1 + 2 * 8])
        self.assertEqual(poet.rollouts, 17)

    def make_population(self, settings, fn):
        evaluate = Recorder(fn)
        poet = WalkerPOET(settings, evaluate, parameter_count=4)
        poet.active = [poet.new_pair(niche(r, 1), np.full(4, r), 0) for r in (0.0, 1.0, 2.0, 3.0)]
        for pair in poet.active:
            pair.score = -5.0
            pair.optimizer.t = 1
        return poet, evaluate

    def test_full_adapts_every_donor_and_topk_only_the_best(self):
        evaluations = {}
        for mode, expected in (("full", 12), ("topk", 4)):
            poet, evaluate = self.make_population(fast(transfer_mode=mode), lambda c, theta: float(theta.mean()))
            poet.transfer()
            evaluations[mode] = evaluate.calls
            self.assertEqual(evaluate.calls[0], 12)
            self.assertEqual(evaluate.calls[1], expected * 16)
            self.assertEqual(evaluate.calls[2], expected)
        self.assertLess(sum(evaluations["topk"]), sum(evaluations["full"]))

    def test_topk_ranks_by_direct_score_with_stable_ties(self):
        poet, _ = self.make_population(fast(transfer_mode="topk"), lambda c, theta: float(theta.mean()))
        attempts = []
        poet.event = lambda kind, **data: attempts.append(data) if kind == "transfer_attempt" else None
        poet.transfer()
        for target in {a["target"] for a in attempts}:
            chosen = [a for a in attempts if a["target"] == target and a["proposal_selected"]]
            self.assertEqual(len(chosen), 1)
            self.assertEqual(chosen[0]["direct_rank"], 1)

    def test_transfer_needs_strict_improvement_and_does_not_cascade(self):
        poet, _ = self.make_population(fast(transfer_mode="topk", l2=0.0), lambda c, theta: float(theta.mean()))
        before = {p.niche.name: p.theta.copy() for p in poet.active}
        best = poet.active[-1]
        best.score = 1e9
        poet.transfer()
        np.testing.assert_array_equal(best.theta, before[best.niche.name])
        # Each replacement comes from the frozen population, so nobody exceeds its best donor by chaining.
        for pair in poet.active[:-1]:
            self.assertLessEqual(float(pair.theta.mean()), 3.0 + 0.5)

    def test_confirmation_is_off_by_default(self):
        poet, evaluate = self.make_population(fast(), lambda c, theta: float(theta.mean()))
        self.assertEqual(poet.confirm_transfers([("anything",)]), [("anything",)])
        poet.transfer()
        self.assertEqual(len(evaluate.calls), 3)

    def test_confirmation_keeps_only_winners_that_hold_up_on_fresh_rollouts(self):
        poet, evaluate = self.make_population(fast(transfer_confirm_rollouts=3), lambda c, theta: float(theta.mean()))
        events = []
        poet.event = lambda kind, **data: events.append((kind, data))
        keep, drop = poet.active[:2]
        keep.theta, drop.theta = np.full(keep.theta.shape, 1.0), np.full(drop.theta.shape, 1.0)
        better, worse = np.full(keep.theta.shape, 2.0), np.zeros(drop.theta.shape)
        evaluate.calls.clear()
        confirmed = poet.confirm_transfers([(keep, 999.0, better, "a", False), (drop, 999.0, worse, "b", False)])
        self.assertEqual(evaluate.calls, [12])
        self.assertEqual([(item[0], item[1], item[3]) for item in confirmed], [(keep, 2.0, "a")])
        rejected = [data for kind, data in events if kind == "transfer_rejected"]
        self.assertEqual([(r["donor"], r["claimed"], r["incumbent"], r["rescored"]) for r in rejected], [("b", 999.0, 1.0, 0.0)])

    def test_reproduction_requires_eligibility_and_window(self):
        evaluate = Recorder(lambda c, theta: 100.0)
        poet = WalkerPOET(fast(reproduce_score=200.0, mc_lower=50.0, mc_upper=300.0), evaluate, parameter_count=4)
        events = []
        poet.event = lambda kind, **data: events.append(kind)
        poet.active[0].score = 150.0
        poet.reproduce()
        self.assertEqual(events, ["reproduction_ineligible"])
        self.assertEqual(len(poet.active), 1)
        poet.active[0].score = 250.0
        poet.reproduce()
        self.assertEqual(len(poet.active), 2)
        self.assertIn("admitted", events)
        self.assertEqual(len(poet.archive), 2)

    def test_child_outside_the_window_is_rejected(self):
        evaluate = Recorder(lambda c, theta: 400.0)
        poet = WalkerPOET(fast(reproduce_score=200.0, mc_lower=50.0, mc_upper=300.0), evaluate, parameter_count=4)
        poet.active[0].score = 250.0
        poet.reproduce()
        self.assertEqual(len(poet.active), 1)

    def test_no_transfer_inherits_parent_without_proposals(self):
        evaluate = Recorder(lambda c, theta: 100.0)
        poet = WalkerPOET(fast(reproduce_score=200.0, mc_lower=50.0, mc_upper=300.0, transfers=False), evaluate, parameter_count=4)
        poet.active[0].score = 250.0
        theta = poet.active[0].theta.copy()
        poet.reproduce()
        np.testing.assert_array_equal(poet.active[1].theta, theta)
        self.assertEqual(max(evaluate.calls[1:]), 1)

    def test_capacity_retires_oldest_but_archive_keeps_it(self):
        evaluate = Recorder(lambda c, theta: 100.0)
        poet = WalkerPOET(fast(reproduce_score=200.0, mc_lower=50.0, mc_upper=300.0, max_active=2), evaluate, parameter_count=4)
        first = poet.active[0].niche.name
        for _ in range(3):
            for pair in poet.active:
                pair.score = 250.0
            poet.reproduce()
        self.assertEqual(len(poet.active), 2)
        self.assertNotIn(first, [p.niche.name for p in poet.active])
        self.assertEqual(len(poet.archive), 4)

    def test_scores_average_repeated_rollouts(self):
        values = iter(range(100))
        poet = WalkerPOET(fast(pairs=1, eval_rollouts=3, reproduction=False, l2=0.0),
                          Recorder(lambda c, theta: float(next(values))), parameter_count=4)
        poet.refresh_scores()
        self.assertAlmostEqual(poet.active[0].score, 1.0)
        poet.step()
        self.assertEqual(poet.rollouts, 3 + 3 + 2)

    def test_finish_needs_a_majority_of_rollouts(self):
        poet = WalkerPOET(fast(eval_rollouts=3), Recorder(lambda c, t: 0.0), parameter_count=4)
        result = Result([300.0] * 6, np.array([True, True, False, True, False, False]))
        returns, finished = poet.read_scores(result, 2)
        self.assertEqual(finished.tolist(), [True, False])

    def test_solved_needs_finish_and_score(self):
        poet = WalkerPOET(fast(), Recorder(lambda c, t: 0.0), parameter_count=4)
        pair = poet.active[0]
        poet.record(pair, 400.0, False, 3, poet.settings)
        self.assertEqual(pair.solved_at, -1)
        poet.record(pair, 100.0, True, 4, poet.settings)
        self.assertEqual(pair.solved_at, -1)
        poet.record(pair, 231.0, True, 5, poet.settings)
        self.assertEqual(pair.solved_at, 5)

    def test_checkpoint_resumes_identically(self):
        fn = lambda c, theta: 10.0 - float(np.sum(theta ** 2))
        settings = fast(pairs=4, children=4, mutation_interval=3, transfer_interval=2)
        original = WalkerPOET(settings, Recorder(fn), parameter_count=4)
        for _ in range(5):
            original.step()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "checkpoint.json"
            original.save(path)
            resumed = WalkerPOET.load(path, Recorder(fn))
        for _ in range(4):
            original.step()
            resumed.step()
        self.assertEqual(len(original.active), len(resumed.active))
        for a, b in zip(original.active, resumed.active):
            np.testing.assert_allclose(a.theta, b.theta)

    def curriculum(self):
        target = Niche(EnvConfig(ground_roughness=1.2, pit_gap=(0.4, 0.8), stump_width=(1, 2),
                                 stump_height=(0.2, 0.4), stump_float=(0, 1)), 9)
        settings = fast(reproduction=False, transfers=False, pairs=2, reproduce_score=200.0)
        return CurriculumRun(settings, Recorder(lambda c, t: 250.0), None, target, parameter_count=4), target

    def test_curriculum_moves_monotonically_and_stops_at_the_target(self):
        run, target = self.curriculum()
        finished_course = lambda tasks: Result([250.0] * len(tasks), np.ones(len(tasks), dtype=bool))
        run.evaluate_fn = finished_course
        previous = np.zeros(5)
        for _ in range(60):
            run.step()
            current = run.active[0].niche.config.vector
            self.assertTrue((current >= previous - 1e-9).all())
            self.assertTrue((current <= target.config.vector + 1e-9).all())
            previous = current
            if run.done:
                break
        self.assertTrue(run.done)
        self.assertEqual(run.closest_solved, 0.0)
        np.testing.assert_allclose(run.active[0].niche.config.vector, target.config.vector)

    def test_curriculum_waits_for_the_threshold(self):
        run, _ = self.curriculum()
        run.evaluate_fn = lambda tasks: Result([150.0] * len(tasks))
        run.step()
        self.assertEqual(run.active[0].niche.config, EnvConfig())
        self.assertIsNone(run.closest_solved)

    def test_mutation_matches_official_operator(self):
        try:
            from poet_distributed import reproduce_ops
            from poet_distributed.niches.box2d.env import Env_config
        except ImportError as exc:
            self.skipTest(f"Box2D is unavailable: {exc}")

        class Args:
            master_seed = 123
            envs = ["stump", "pit", "roughness"]

        official = reproduce_ops.Reproducer(Args())
        ours = Reproducer(123)
        config = EnvConfig()
        parent = Env_config(name="flat", ground_roughness=0, pit_gap=[], stump_width=[], stump_height=[],
                            stump_float=[], stair_height=[], stair_width=[], stair_steps=[])
        for _ in range(300):
            parent = official.mutate(parent)
            config = ours.mutate(config)
            self.assertAlmostEqual(parent.ground_roughness, config.ground_roughness)
            np.testing.assert_allclose(parent.pit_gap, config.pit_gap)
            np.testing.assert_allclose(parent.stump_height, config.stump_height)
            np.testing.assert_allclose(parent.stump_width, config.stump_width)


class ResumeLogTests(unittest.TestCase):
    def test_walker_log_drops_rows_past_the_checkpoint_and_a_partial_line(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "progress.jsonl"
            path.write_text('{"iteration": 24, "wall_seconds": 5}\n{"iteration": 25, "wall_seconds": 6}\n{"iterat')
            self.assertEqual(trim_log(path, 25), dict(iteration=24, wall_seconds=5))
            self.assertEqual(path.read_text(), '{"iteration": 24, "wall_seconds": 5}\n')

    def test_walker_log_is_emptied_when_a_run_restarts(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "events.jsonl"
            path.write_text('{"iteration": 3}\n')
            self.assertIsNone(trim_log(path, 0))
            self.assertEqual(path.read_text(), "")
            self.assertIsNone(trim_log(Path(folder) / "absent.jsonl", 0))

    def test_box2d_logs_keep_the_header_and_events_before_the_loop(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            (output / "metrics.csv").write_text("iteration,score\n24,1.0\n25,2.0\n26,3")
            (output / "events.jsonl").write_text('{"iteration": -1}\n{"iteration": 24}\n{"iteration": 25}\n{"it')
            trim_logs(output, 25)
            self.assertEqual((output / "metrics.csv").read_text(), "iteration,score\n24,1.0\n")
            self.assertEqual((output / "events.jsonl").read_text(), '{"iteration": -1}\n{"iteration": 24}\n')


class UnbuildableCourseTests(unittest.TestCase):
    class Candidate:
        optim_id = "thin_stump"

    def test_a_course_box2d_cannot_build_fails_the_minimal_criterion(self):
        def evaluate(_, theta):
            raise AssertionError("area > b2_epsilon")
        reported = []
        score = reject_unbuildable(evaluate, reported.append)(self.Candidate(), None)
        self.assertEqual(score, float("-inf"))
        self.assertEqual(reported, ["thin_stump"])

    def test_other_assertions_and_scores_pass_through(self):
        def evaluate(_, theta):
            raise AssertionError("something else")
        with self.assertRaises(AssertionError):
            reject_unbuildable(evaluate, None)(self.Candidate(), None)
        self.assertEqual(reject_unbuildable(lambda _, theta: 7.0, None)(self.Candidate(), None), 7.0)


class TargetTests(unittest.TestCase):
    @staticmethod
    def write_run(root, name, niches):
        folder = root / name
        folder.mkdir()
        solved = {niche: dict(score=240, finished=True, solved_at=at) for niche, (_, _, at) in niches.items()}
        solved["flat"] = dict(score=300, finished=True, solved_at=100)
        (folder / "progress.jsonl").write_text(json.dumps(dict(iteration=0, active=1, archive=1, rollouts=1, steps=1,
                                                               niches=solved)) + "\n")
        (folder / "events.jsonl").write_text("".join(
            json.dumps(dict(event="admitted", niche=niche, config=config, seed=seed)) + "\n"
            for niche, (config, seed, _) in niches.items()))

    def test_picks_the_hardest_at_each_level_from_requested_modes_only(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self.write_run(root, "full_seed_1", {"easy": (dict(ground_roughness=0.6), 1, 300),
                                                 "mid": (dict(ground_roughness=1.2), 2, 500),
                                                 "hard": (dict(ground_roughness=4.8), 3, 900)})
            self.write_run(root, "topk_seed_1", {"other": (dict(ground_roughness=3.0), 4, 400)})
            targets = pick_targets(root, per_level=1)
            self.assertEqual({(t["name"], t["level"], t["budget"]) for t in targets},
                             {("full_seed_1_mid", "below_challenging", 500), ("full_seed_1_hard", "challenging", 900)})

    def test_a_course_solved_twice_keeps_its_earliest_budget(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            config = dict(ground_roughness=1.2)
            self.write_run(root, "full_seed_1", {"a": (config, 7, 800)})
            self.write_run(root, "full_seed_2", {"b": (config, 7, 600)})
            targets = pick_targets(root)
            self.assertEqual([(t["name"], t["budget"]) for t in targets], [("full_seed_2_b", 600)])

    def test_distance_uses_the_curriculum_scale(self):
        self.assertAlmostEqual(distance_from_flat(dict(ground_roughness=8.0)), 1.0)
        self.assertAlmostEqual(distance_from_flat(dict(stump_height=(0.0, 3.0))), 1.0)
        self.assertEqual(distance_from_flat({}), 0.0)


class CompareTests(unittest.TestCase):
    @staticmethod
    def row(run, iteration, steps, solved=1):
        return dict(run=run, iteration=iteration, steps=steps, rollouts=steps // 10, policies=2,
                    mean_library_return=float(solved), courses_solved_by_library=solved,
                    courses_solved_by_best_policy=solved)

    def test_run_name_reads_folders_and_snapshots(self):
        self.assertEqual(run_name("runs/x/topk_seed_1/checkpoint.json"), "topk_seed_1")
        self.assertEqual(run_name("runs/snapshots/no_transfer_seed_1_01400.json"), "no_transfer_seed_1")

    def test_list_keeps_one_checkpoint_per_run_and_iteration(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / "main"
            snapshots = Path(folder) / "snapshots"
            for run in ("full_seed_1", "direct_seed_1"):
                (root / run).mkdir(parents=True)
                (root / run / "checkpoint.json").write_text(json.dumps(dict(iteration=200)))
            snapshots.mkdir()
            for iteration in (100, 200):
                (snapshots / f"full_seed_1_{iteration:05d}.json").write_text(json.dumps(dict(iteration=iteration)))
            self.assertEqual([path.name for path in list_checkpoints(root, snapshots)],
                             ["full_seed_1_00100.json", "full_seed_1_00200.json"])

    def test_budget_takes_the_latest_checkpoint_within_it(self):
        runs = by_run([self.row("full_seed_1", 200, 900), self.row("full_seed_1", 100, 500),
                       self.row("topk_seed_1", 100, 300), self.row("topk_seed_1", 200, 700)])
        self.assertEqual(common_budget(runs), 700)
        view = at_budget(runs, 700)
        self.assertEqual((view["full_seed_1"]["iteration"], view["topk_seed_1"]["iteration"]), (100, 200))
        self.assertEqual(at_iteration(runs, 200)["full_seed_1"]["steps"], 900)

    def test_report_names_runs_missing_from_a_view(self):
        result = report([self.row("full_seed_1", 200, 900, solved=4), self.row("full_seed_2", 200, 800, solved=2),
                         self.row("topk_seed_1", 100, 500)], iteration=200, budget=850)
        self.assertEqual(result["equal_iteration"]["missing"], ["topk_seed_1"])
        self.assertEqual(result["equal_iteration"]["modes"]["full"]["courses_solved_by_library"],
                         dict(seeds={"1": 4, "2": 2}, mean=3.0))
        self.assertEqual(result["equal_compute"]["missing"], ["full_seed_1"])


if __name__ == "__main__":
    unittest.main()
