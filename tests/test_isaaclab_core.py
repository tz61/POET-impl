"""Analytical unit tests of the Isaac Lab port's simulator-independent loop.

No Isaac Sim, PhysX, robot, or locomotion result is mocked or claimed here.
The evaluator fixtures are explicit scalar functions of policy parameters.
"""

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


def quadratic_scores(terrain, parameters):
    """Analytical smooth objective with a terrain-dependent optimum."""
    target = np.linspace(.25, .75, parameters.shape[1]) + terrain.roughness
    return 200. - np.square(parameters - target).sum(axis=1)


class IsaacLabCoreTests(unittest.TestCase):
    def settings(self, **kwargs):
        base = Settings(seed=81, pairs=32, learning_rate=.025, l2=0.,
                        transfer_interval=1, mutation_interval=2,
                        children=5, max_active=3, mc_lower=0., mc_upper=300.,
                        reproduce_score=150.)
        return replace(base, **kwargs)

    def assert_poet_equal(self, expected, actual):
        self.assertEqual(expected.iteration, actual.iteration)
        self.assertEqual(expected.evaluations, actual.evaluations)
        self.assertEqual(expected.settings, actual.settings)
        self.assertEqual(expected.archive, actual.archive)
        self.assertEqual(expected.rng.bit_generator.state, actual.rng.bit_generator.state)
        self.assertEqual(len(expected.active), len(actual.active))
        for a, b in zip(expected.active, actual.active):
            self.assertEqual(a.terrain, b.terrain)
            self.assertEqual(a.born, b.born)
            self.assertEqual(a.sigma, b.sigma)
            self.assertEqual(a.score, b.score)
            self.assertEqual(a.best_score, b.best_score)
            np.testing.assert_array_equal(a.theta, b.theta)
            if a.best_theta is None:
                self.assertIsNone(b.best_theta)
            else:
                np.testing.assert_array_equal(a.best_theta, b.best_theta)
            self.assertEqual(a.optimizer.t, b.optimizer.t)
            self.assertEqual(a.optimizer.stepsize, b.optimizer.stepsize)
            np.testing.assert_array_equal(a.optimizer.m, b.optimizer.m)
            np.testing.assert_array_equal(a.optimizer.v, b.optimizer.v)

    def test_checkpoint_resume_matches_uninterrupted_rng_and_evolution(self):
        continuous_events, resumed_events = [], []
        record_continuous = lambda name, **data: continuous_events.append((name, data))
        record_resumed = lambda name, **data: resumed_events.append((name, data))
        uninterrupted = POET(self.settings(), quadratic_scores, record_continuous, parameter_count=6)
        partial = POET(self.settings(), quadratic_scores, record_resumed, parameter_count=6)
        for _ in range(8):
            uninterrupted.step()
        for _ in range(3):
            partial.step()
        with tempfile.TemporaryDirectory(prefix="poet-port-checkpoint-") as directory:
            path = Path(directory) / "state.json"
            partial.save(path)
            resumed = POET.load(path, quadratic_scores, record_resumed)
            self.assert_poet_equal(partial, resumed)
            for _ in range(5):
                resumed.step()
            self.assert_poet_equal(uninterrupted, resumed)
            self.assertEqual(continuous_events, resumed_events)
            # Ensure the comparison exercised evolution, not only one ES run.
            self.assertGreater(len(uninterrupted.archive), 1)
            self.assertIn("transfer_attempt", [name for name, _ in continuous_events])
            uninterrupted.save(Path(directory) / "continuous.json")
            resumed.save(Path(directory) / "resumed.json")
            self.assertEqual(json.loads((Path(directory) / "continuous.json").read_text()),
                             json.loads((Path(directory) / "resumed.json").read_text()))

    def test_checkpoint_before_first_evaluation_round_trips(self):
        original = POET(self.settings(), quadratic_scores, parameter_count=6)
        with tempfile.TemporaryDirectory(prefix="poet-port-initial-") as directory:
            path = Path(directory) / "initial.json"
            original.save(path)
            restored = POET.load(path, quadratic_scores)
            self.assert_poet_equal(original, restored)
            original.step()
            restored.step()
            self.assert_poet_equal(original, restored)

    def test_es_improves_fixed_quadratic_and_uses_antithetic_pairs(self):
        calls = []

        def objective(terrain, parameters):
            calls.append(parameters.copy())
            return -np.square(parameters - .5).sum(axis=1)

        poet = POET(self.settings(pairs=512), objective, parameter_count=6)
        pair = poet.active[0]
        before = pair.theta.copy()
        original_score = float(objective(pair.terrain, before[None])[0])
        calls.clear()
        poet.optimize(pair)
        cloud = calls[0].reshape(512, 2, 6)
        np.testing.assert_allclose(cloud[:, 0] + cloud[:, 1],
                                   np.broadcast_to(2. * before, (512, 6)), atol=1e-15)
        self.assertGreater(pair.score, original_score)
        self.assertEqual(pair.optimizer.t, 1)
        self.assertEqual(poet.evaluations, 2 * 512 + 1)
        self.assertEqual(pair.best_score, pair.score)
        np.testing.assert_array_equal(pair.best_theta, pair.theta)

    def test_proposals_preserve_target_optimizer_and_both_policies(self):
        poet = POET(self.settings(), quadratic_scores, parameter_count=6)
        target = poet.active[0]
        poet.optimize(target)
        donor = np.linspace(-.8, -.2, 6)
        donor_before = donor.copy()
        policy_before = target.theta.copy()
        state_before = copy.deepcopy(target.optimizer.__dict__)
        sigma_before = target.sigma
        for use_adam in (False, True):
            with self.subTest(use_adam=use_adam):
                proposal = poet.proposal(target, donor, use_adam=use_adam)
                self.assertTrue(np.isfinite(proposal).all())
                self.assertFalse(np.array_equal(proposal, donor))
                np.testing.assert_array_equal(donor, donor_before)
                np.testing.assert_array_equal(target.theta, policy_before)
                self.assertEqual(target.sigma, sigma_before)
                for key, before in state_before.items():
                    after = target.optimizer.__dict__[key]
                    if isinstance(before, np.ndarray):
                        np.testing.assert_array_equal(before, after)
                    else:
                        self.assertEqual(before, after)
        fresh = poet.new_pair(Terrain(.1), donor, born=0)
        self.assertEqual(fresh.optimizer.t, 0)
        self.assertTrue(np.isfinite(poet.proposal(fresh, donor, use_adam=True)).all())
        self.assertEqual(fresh.optimizer.t, 0)

    def test_transfer_uses_frozen_donors_without_acceptance_cascade(self):
        def run(reverse):
            def objective(terrain, parameters):
                return -np.square(parameters[:, 0] - 3.)

            poet = POET(self.settings(), objective, parameter_count=1)
            a = poet.new_pair(Terrain(0., seed=1), np.array([0.]), 0)
            b = poet.new_pair(Terrain(.1, seed=2), np.array([1.]), 0)
            for pair in (a, b):
                pair.score = float(objective(pair.terrain, pair.theta[None])[0])
            poet.active = [b, a] if reverse else [a, b]
            proposal_inputs = []

            def proposal(target, donor, use_adam=True):
                proposal_inputs.append((target.terrain.key, donor.copy()))
                # Analytical transfer fixture: only A can adapt B to x=3.
                return np.array([3.]) if target is a else donor.copy()

            poet.proposal = proposal
            poet.transfer()
            # B must see A's old x=0, never A's accepted proposal x=3.
            seen_by_b = [value for key, value in proposal_inputs if key == b.terrain.key]
            np.testing.assert_array_equal(seen_by_b, [[0.]])
            np.testing.assert_array_equal(a.theta, [3.])
            np.testing.assert_array_equal(b.theta, [1.])
            return {pair.terrain.key: pair.theta.tolist() for pair in poet.active}

        self.assertEqual(run(False), run(True))

    def test_duplicate_and_mc_filtering_retire_fifo_but_retain_archive(self):
        scores = {.2: -20., .1: 30., .6: 50., .5: 400.}
        events, evaluated = [], []

        def objective(terrain, parameters):
            evaluated.append(terrain.key)
            return np.full(len(parameters), scores[terrain.roughness])

        poet = POET(self.settings(transfers=False, max_active=2, children=6,
                                  mc_lower=0., mc_upper=100., reproduce_score=10.),
                    objective, lambda name, **data: events.append((name, data)), parameter_count=2)
        oldest = poet.active[0]
        oldest.score = 50.
        younger = poet.new_pair(Terrain(.03), np.array([.4, .6]), born=1)
        younger.score = 0.
        retired = Terrain(.3, seed=10)
        poet.active.append(younger)
        poet.archive.extend([younger.terrain, retired])
        candidates = iter([Terrain(.3, seed=999), Terrain(.2), Terrain(.1),
                           Terrain(.6, seed=5), Terrain(.6, seed=6), Terrain(.5)])
        poet.mutate = lambda parent: next(candidates)
        poet.iteration = 10
        poet.reproduce()
        self.assertNotIn(retired.key, evaluated)
        # The repeated genotype, despite a new seed, is not evaluated twice
        # in the candidate-generation phase (the accepted child is reevaluated).
        mutations = [data for name, data in events if name == "mutation"]
        self.assertEqual([data["child"]["roughness"] for data in mutations], [.2, .1, .6, .5])
        self.assertEqual([data["passed"] for data in mutations], [False, True, True, False])
        self.assertEqual([pair.terrain.roughness for pair in poet.active], [.03, .6])
        self.assertEqual(poet.active[-1].born, 10)
        self.assertIn(oldest.terrain, poet.archive)
        self.assertIn(retired, poet.archive)
        self.assertEqual([data["terrain"] for name, data in events if name == "retired"],
                         [oldest.terrain.name])

    def test_no_transfer_child_inherits_only_its_parent(self):
        queried = []

        def objective(terrain, parameters):
            queried.extend(parameters.copy())
            # An alternative donor would score better, but must not be queried.
            return 50. + parameters[:, 0]

        poet = POET(self.settings(transfers=False, children=1, reproduce_score=100.),
                    objective, parameter_count=2)
        parent = poet.active[0]
        parent.theta = np.array([1., 2.])
        parent.score = 150.
        alternative = poet.new_pair(Terrain(.1), np.array([9., 8.]), born=0)
        alternative.score = 0.
        poet.active.append(alternative)
        poet.archive.append(alternative.terrain)
        poet.mutate = lambda terrain: Terrain(.2, seed=27)

        def forbidden_proposal(*args, **kwargs):
            self.fail("No-transfer condition must not make adaptation proposals")

        poet.proposal = forbidden_proposal
        poet.reproduce()
        self.assertEqual(len(poet.active), 3)
        child = poet.active[-1]
        np.testing.assert_array_equal(child.theta, parent.theta)
        self.assertFalse(np.shares_memory(child.theta, parent.theta))
        self.assertEqual(len(queried), 2)  # inherited MC test and admission evaluation
        np.testing.assert_array_equal(queried, [[1., 2.], [1., 2.]])

    def test_post_transfer_minimal_criterion_rejects_over_easy_child(self):
        events = []

        def objective(terrain, parameters):
            return np.where(parameters[:, 0] > 5., 400., 50.)

        poet = POET(self.settings(children=1, mc_lower=0., mc_upper=100., reproduce_score=100.),
                    objective, lambda name, **data: events.append(name), parameter_count=2)
        parent = poet.active[0]
        parent.theta = np.array([1., 2.])
        parent.score = 150.
        alternate = poet.new_pair(Terrain(.1), np.array([9., 8.]), born=0)
        alternate.score = 0.
        poet.active.append(alternate)
        poet.archive.append(alternate.terrain)
        poet.mutate = lambda terrain: Terrain(.2)
        poet.proposal = lambda target, donor, use_adam=False: donor.copy()
        poet.reproduce()
        self.assertEqual(len(poet.active), 2)
        self.assertEqual(len(poet.archive), 2)
        self.assertIn("mutation", events)
        self.assertIn("child_transfer_attempt", events)
        self.assertNotIn("admitted", events)

    def test_evaluator_rejects_invalid_shape_and_nonfinite_scores(self):
        for evaluator in (lambda terrain, theta: np.zeros((len(theta), 1)),
                          lambda terrain, theta: np.full(len(theta), np.nan)):
            poet = POET(self.settings(), evaluator, parameter_count=2)
            with self.assertRaisesRegex(ValueError, "finite score"):
                poet.evaluate(poet.active[0].terrain, poet.active[0].theta)
            self.assertEqual(poet.evaluations, 0)


if __name__ == "__main__":
    unittest.main()
