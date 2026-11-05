"""Analytical tests of periodic donor screening, not locomotion experiments."""
import copy
from dataclasses import replace
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from isaaclab_port.core import POET, Settings, Terrain


def objective(terrain, parameters):
    return 100. - np.square(parameters - (0.5 + terrain.roughness)).sum(axis=1)


class CountedObjective:
    def __init__(self, function=objective):
        self.function = function
        self.calls = []

    def __call__(self, terrain, parameters):
        self.calls.append((terrain, parameters.copy()))
        return self.function(terrain, parameters)

    @property
    def evaluations(self):
        return sum(len(parameters) for _, parameters in self.calls)


def population(mode="full", count=4, k=1, function=objective, pairs=32):
    evaluator = CountedObjective(function)
    events = []
    settings = Settings(seed=81, pairs=pairs, learning_rate=0.03, l2=0.,
                        transfers=mode != "no_transfer",
                        periodic_transfer_mode="topk" if mode == "topk" else "full",
                        transfer_top_k=k)
    poet = POET(settings, evaluator, lambda kind, **data: events.append((kind, data)),
                parameter_count=2)
    poet.active = [poet.new_pair(Terrain(i * 0.1, seed=i), np.array([i * 0.3, -0.1]), 0)
                   for i in range(count)]
    poet.archive = [pair.terrain for pair in poet.active]
    for pair in poet.active:
        pair.score = float(function(pair.terrain, pair.theta[None])[0])
        poet.update_best(pair)
    return poet, evaluator, events


class TopKTransferTests(unittest.TestCase):
    def assert_population_equal(self, first, second):
        self.assertEqual(first.evaluations, second.evaluations)
        self.assertEqual(first.rng.bit_generator.state, second.rng.bit_generator.state)
        self.assertEqual(first.archive, second.archive)
        self.assertEqual(len(first.active), len(second.active))
        for left, right in zip(first.active, second.active):
            self.assertEqual(left.terrain, right.terrain)
            self.assertEqual(left.score, right.score)
            self.assertEqual(left.sigma, right.sigma)
            np.testing.assert_array_equal(left.theta, right.theta)
            np.testing.assert_array_equal(left.best_theta, right.best_theta)
            for key, value in left.optimizer.__dict__.items():
                if isinstance(value, np.ndarray):
                    np.testing.assert_array_equal(value, right.optimizer.__dict__[key])
                else:
                    self.assertEqual(value, right.optimizer.__dict__[key])

    def test_real_es_cost_and_all_direct_candidates_are_logged(self):
        for mode, expected, selected in (("full", 792, 12), ("topk", 272, 4),
                                         ("no_transfer", 0, 0)):
            with self.subTest(mode=mode):
                poet, evaluator, events = population(mode)
                poet.transfer()
                self.assertEqual(poet.evaluations, expected)
                self.assertEqual(evaluator.evaluations, expected)
                attempts = [data for kind, data in events if kind == "transfer_attempt"]
                self.assertEqual(len(attempts), 0 if mode == "no_transfer" else 12)
                self.assertEqual(sum(data["proposal_selected"] for data in attempts), selected)
                for data in attempts:
                    self.assertEqual(data["proposal_score"] is not None, data["proposal_selected"])
                for target in poet.active:
                    ranked = [data["direct_rank"] for data in attempts
                              if data["target"] == target.terrain.name]
                    self.assertEqual(sorted(ranked), [] if mode == "no_transfer" else [1, 2, 3])

    def test_screening_can_miss_a_lower_ranked_donors_better_proposal(self):
        # Explicit lookup-like analytical proposal fixture: the lower direct
        # donor has the better adapted score. It is not a physical policy.
        def linear(terrain, parameters):
            return parameters[:, 0]

        outcomes = {}
        for mode in ("full", "topk"):
            poet, _, events = population(mode, count=3, function=linear)
            for pair, x in zip(poet.active, (0., 2., 1.)):
                pair.theta = np.array([x, 0.])
                pair.score = x
            target = poet.active[0]
            poet.proposal = lambda to, donor, use_adam=True: (
                np.array([10. if donor[0] == 1. else 2.5, 0.])
                if to is target else donor.copy())
            poet.transfer()
            outcomes[mode] = target.score
            attempts = [data for kind, data in events
                        if kind == "transfer_attempt" and data["target"] == target.terrain.name]
            self.assertEqual(sorted(data["direct_score"] for data in attempts), [1., 2.])
        self.assertEqual(outcomes, {"full": 10., "topk": 2.5})

    def test_direct_winner_survives_even_when_its_adaptation_is_worse(self):
        def linear(terrain, parameters):
            return parameters[:, 0]
        poet, _, _ = population("topk", count=3, function=linear)
        target = poet.active[0]
        direct_best = max(pair.theta[0] for pair in poet.active[1:])
        poet.proposal = lambda to, donor, use_adam=True: np.full_like(donor, -100.)
        poet.transfer()
        self.assertEqual(target.score, direct_best)

    def test_tied_donors_use_stable_names_not_population_order(self):
        choices = []
        for reverse in (False, True):
            poet, _, events = population("topk", function=lambda terrain, parameters: np.zeros(len(parameters)))
            expected = {target.terrain.name: min(other.terrain.name for other in poet.active if other is not target)
                        for target in poet.active}
            if reverse:
                poet.active.reverse()
            poet.proposal = lambda target, donor, use_adam=True: donor.copy()
            poet.transfer()
            actual = {data["target"]: data["donor"] for kind, data in events
                      if kind == "transfer_attempt" and data["proposal_selected"]}
            self.assertEqual(actual, expected)
            choices.append(actual)
        self.assertEqual(*choices)

    def test_small_populations_and_k_covering_all_donors_match_full(self):
        for count, k in ((1, 1), (2, 1), (4, 3), (4, 8)):
            with self.subTest(count=count, k=k):
                full, _, _ = population("full", count=count)
                topk, _, _ = population("topk", count=count, k=k)
                full.transfer()
                topk.transfer()
                self.assert_population_equal(full, topk)

    def test_full_matches_original_interleaved_transfer_with_warm_adam(self):
        current, _, _ = population("full", pairs=8)
        for pair in current.active:
            current.optimize(pair)
        legacy = copy.deepcopy(current)
        current.transfer()
        # Original direct -> proposal interleaving, retained here as a baseline
        # oracle. Evaluation order differs; deterministic outcomes must not.
        donors = [(pair.terrain.name, pair.theta.copy()) for pair in legacy.active]
        replacements = []
        for target in legacy.active:
            best_score, best_theta = target.score, target.theta
            for name, theta in donors:
                if name == target.terrain.name:
                    continue
                direct = float(legacy.evaluate(target.terrain, theta)[0])
                proposed = legacy.proposal(target, theta)
                adapted = float(legacy.evaluate(target.terrain, proposed)[0])
                if direct > best_score:
                    best_score, best_theta = direct, theta
                if adapted > best_score:
                    best_score, best_theta = adapted, proposed
            replacements.append((target, best_score, best_theta.copy()))
        for target, score, theta in replacements:
            if score > target.score:
                target.theta, target.score = theta, score
                target.optimizer.reset()
                target.sigma = legacy.settings.noise
                legacy.update_best(target)
        self.assert_population_equal(current, legacy)

    def test_topk_uses_frozen_donors_without_acceptance_cascade(self):
        def quadratic(terrain, parameters):
            optimum = {0.: 3., .1: 1., .2: -100.}[terrain.roughness]
            return -np.square(parameters[:, 0] - optimum)

        for reverse in (False, True):
            poet, _, _ = population("topk", count=3, function=quadratic)
            a, b, c = poet.active
            for pair, x in zip(poet.active, (0., 1., -100.)):
                pair.theta = np.array([x, 0.])
                pair.score = float(quadratic(pair.terrain, pair.theta[None])[0])
            seen_by_b = []
            def proposal(target, donor, use_adam=True):
                if target is b:
                    seen_by_b.append(donor.copy())
                return np.array([3., 0.]) if target is a else donor.copy()
            poet.proposal = proposal
            if reverse:
                poet.active.reverse()
            poet.transfer()
            np.testing.assert_array_equal(seen_by_b, [[0., 0.]])
            np.testing.assert_array_equal(a.theta, [3., 0.])
            np.testing.assert_array_equal(b.theta, [1., 0.])

    def test_child_initialization_is_identical_for_full_and_topk(self):
        results = []
        for mode in ("full", "topk"):
            poet, evaluator, events = population(mode, count=3, pairs=8)
            poet.settings = replace(poet.settings, children=1, reproduce_score=0., max_active=4)
            poet.mutate = lambda terrain: Terrain(.6, seed=10)
            poet.reproduce()
            self.assertEqual(len(poet.active), 4)
            attempts = [data for kind, data in events if kind == "child_transfer_attempt"]
            self.assertEqual(len(attempts), 3)
            self.assertEqual(evaluator.evaluations, 1 + 3 * (2 * 8 + 2))
            results.append((poet, attempts))
        self.assert_population_equal(results[0][0], results[1][0])
        self.assertEqual(results[0][1], results[1][1])

    def test_old_checkpoint_defaults_to_full_and_keeps_no_transfer(self):
        for transfers in (False, True):
            poet, _, _ = population()
            poet.settings = replace(poet.settings, transfers=transfers)
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "old.json"
                poet.save(path)
                data = json.loads(path.read_text())
                del data["settings"]["periodic_transfer_mode"]
                del data["settings"]["transfer_top_k"]
                path.write_text(json.dumps(data))
                restored = POET.load(path, objective)
            self.assertEqual(restored.settings.periodic_transfer_mode, "full")
            self.assertEqual(restored.settings.transfer_top_k, 1)
            self.assertEqual(restored.settings.transfers, transfers)
            self.assert_population_equal(poet, restored)

    def test_topk_checkpoint_resume_is_identical(self):
        continuous, _, _ = population("topk")
        partial, _, _ = population("topk")
        for poet in (continuous, partial):
            poet.settings = replace(poet.settings, transfer_interval=1, mutation_interval=100)
        for _ in range(4):
            continuous.step()
        for _ in range(2):
            partial.step()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "saved.json"
            partial.save(path)
            resumed = POET.load(path, objective)
            for _ in range(2):
                resumed.step()
            continuous.save(Path(directory) / "continuous.json")
            resumed.save(Path(directory) / "resumed.json")
            self.assertEqual(json.loads((Path(directory) / "continuous.json").read_text()),
                             json.loads((Path(directory) / "resumed.json").read_text()))

    def test_invalid_screening_settings_are_rejected(self):
        for invalid in (0, -1, True, 1.5):
            with self.subTest(k=invalid), self.assertRaisesRegex(ValueError, "positive integer"):
                POET(Settings(transfer_top_k=invalid), objective, parameter_count=2)
        with self.assertRaisesRegex(ValueError, "periodic_transfer_mode"):
            POET(Settings(periodic_transfer_mode="unknown"), objective, parameter_count=2)


if __name__ == "__main__":
    unittest.main()
