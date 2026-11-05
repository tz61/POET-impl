"""Analytical unit fixtures for the unchanged original POET implementation.

These tests check optimization and population bookkeeping, not locomotion or
paper-level performance.  A tiny quadratic objective replaces rollouts only
inside this test file.  Import-only Env_config fixtures allow the bookkeeping
methods to be tested without importing the Box2D simulator.
"""

from collections import OrderedDict, namedtuple
import copy
import importlib.util
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest import mock

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / "upstream"
if str(UPSTREAM) not in sys.path:
    sys.path.insert(0, str(UPSTREAM))

from poet_distributed import es, novelty
from poet_distributed.noise import SharedNoiseTable


def _load_file(name, filename):
    spec = importlib.util.spec_from_file_location(name, filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


fiber = _load_file("_analytical_serial_fiber", ROOT / "local_runtime" / "fiber.py")

EnvConfig = namedtuple("EnvConfig", [
    "name", "ground_roughness", "pit_gap", "stump_width", "stump_height",
    "stump_float", "stair_height", "stair_width", "stair_steps",
])


def environment(name, roughness=0., pit_gap=None, stump_height=None):
    return EnvConfig(name, roughness, list(pit_gap or []), [],
                     list(stump_height or []), [], [], [], [])


def _load_bookkeeping_modules():
    """Inject only a configuration type while importing actual POET methods.

    All module replacements are scoped to this import.  No production module
    or simulator is edited, and no fake environment is installed afterwards.
    """
    env_fixture = ModuleType("poet_distributed.niches.box2d.env")
    env_fixture.Env_config = EnvConfig
    with mock.patch.dict(sys.modules, {
        "poet_distributed.niches.box2d.env": env_fixture,
    }):
        reproduction = _load_file("poet_distributed._analytical_reproduce_ops",
                                  UPSTREAM / "poet_distributed" / "reproduce_ops.py")
        with mock.patch.dict(sys.modules, {
            "poet_distributed.reproduce_ops": reproduction,
        }):
            algorithm = _load_file("poet_distributed._analytical_poet_algo",
                                   UPSTREAM / "poet_distributed" / "poet_algo.py")
    return reproduction, algorithm


reproduction, algorithm = _load_bookkeeping_modules()


class QuadraticNiche:
    """Deterministic analytical objective; deliberately not a physics fixture."""

    def __init__(self, target):
        self.target = np.asarray(target).copy()

    def score(self, theta):
        return -float(np.square(np.asarray(theta) - self.target).sum())

    def rollout_batch(self, thetas, batch_size, random_state, eval=False):
        scores = np.asarray([self.score(theta) for theta in thetas])
        if len(scores) != batch_size:
            raise AssertionError("Incorrect analytical rollout batch size")
        return scores, np.ones(batch_size, dtype=int)


class EvolutionStrategyTests(unittest.TestCase):
    def setUp(self):
        self.master_rng = np.random.get_state()
        np.random.seed(1234)
        self.directory = tempfile.TemporaryDirectory(prefix="poet-es-test-")
        self.addCleanup(self.directory.cleanup)
        self.manager = fiber.Manager()
        self.shared = {"thetas": self.manager.dict(), "niches": self.manager.dict()}
        self.pool = fiber.Pool(1)
        self.addCleanup(self.pool.terminate)

        # Use the upstream noise lookup/indexing methods without allocating
        # its production 250-million-element shared-memory table.
        table = object.__new__(SharedNoiseTable)
        table.noise = np.random.RandomState(42).randn(100_000).astype(np.float32)
        for name, value in (("noise", table), ("thetas", self.shared["thetas"]),
                            ("niches", self.shared["niches"])):
            patcher = mock.patch.object(es, name, value, create=True)
            patcher.start()
            self.addCleanup(patcher.stop)
        es.fiber_get_theta.cache_clear()
        es.fiber_get_niche.cache_clear()
        self.addCleanup(es.fiber_get_theta.cache_clear)
        self.addCleanup(es.fiber_get_niche.cache_clear)
        self.addCleanup(np.random.set_state, self.master_rng)

        self.initial = np.array([-1.5, -1., -.5, .2, 1.3, 2.])
        self.target = np.array([.2, -.2, .4, -.3, .6, .1])
        self.niche = QuadraticNiche(self.target)
        self.optimizer = es.ESOptimizer(
            fiber_pool=self.pool, fiber_shared=self.shared,
            theta=self.initial.copy(), make_niche=lambda: self.niche,
            learning_rate=.05, batches_per_chunk=4, batch_size=128,
            eval_batch_size=2, eval_batches_per_step=2, l2_coeff=0.,
            noise_std=.1, normalize_grads_by_noise_std=True,
            returns_normalization="centered_ranks", optim_id="quadratic",
            log_file=self.directory.name, is_candidate=True,
        )

    def test_antithetic_workers_evaluate_both_signs_of_same_noise(self):
        self.optimizer.broadcast_theta(self.initial)
        result = es.run_po_batch_fiber(self.optimizer.iteration, "quadratic", 8, 19, .1)
        for row, index in enumerate(result.noise_inds):
            perturbation = es.noise.get(index, len(self.initial))
            self.assertAlmostEqual(result.returns[row, 0],
                                   self.niche.score(self.initial + .1 * perturbation))
            self.assertAlmostEqual(result.returns[row, 1],
                                   self.niche.score(self.initial - .1 * perturbation))
        np.testing.assert_array_equal(result.lengths, np.ones((8, 2), dtype=int))

    def test_one_es_update_improves_the_known_quadratic_objective(self):
        task = self.optimizer.start_step()
        gradient, _ = self.optimizer.compute_grads(self.optimizer.get_chunk(task[0]),
                                                   self.initial)
        # The true ascent direction is 2*(target-theta).  This independently
        # checks the sign convention before checking the actual Adam update.
        self.assertGreater(float(gradient @ (self.target - self.initial)), 0.)
        improved, stats = self.optimizer.get_step(task)
        self.assertGreater(self.niche.score(improved), self.niche.score(self.initial))
        self.assertTrue(np.isfinite(improved).all())
        self.assertGreater(stats.grad_norm, 0.)
        self.assertEqual(self.optimizer.optimizer.t, 1)
        # The caller, not get_step, owns replacing the policy vector.
        np.testing.assert_array_equal(self.optimizer.theta, self.initial)
        self.assertAlmostEqual(self.optimizer.evaluate_theta(improved),
                               self.niche.score(improved))

    def test_proposal_only_preserves_optimizer_moments_and_live_policy(self):
        self.optimizer.theta, _ = self.optimizer.get_step(self.optimizer.start_step())
        live_policy = self.optimizer.theta.copy()
        adam = self.optimizer.optimizer
        before = (adam.t, adam.m.copy(), adam.v.copy(), adam.stepsize,
                  self.optimizer.noise_std)
        task = self.optimizer.start_step()
        for use_adam in (False, True):
            with self.subTest(propose_with_adam=use_adam):
                proposal, _ = self.optimizer.get_step(
                    task, propose_only=True, propose_with_adam=use_adam)
                self.assertTrue(np.isfinite(proposal).all())
                self.assertFalse(np.array_equal(proposal, live_policy))
                np.testing.assert_array_equal(self.optimizer.theta, live_policy)
                self.assertEqual(adam.t, before[0])
                np.testing.assert_array_equal(adam.m, before[1])
                np.testing.assert_array_equal(adam.v, before[2])
                self.assertEqual(adam.stepsize, before[3])
                self.assertEqual(self.optimizer.noise_std, before[4])


class PopulationBookkeepingTests(unittest.TestCase):
    def test_novelty_is_mean_distance_to_nearest_archive_members(self):
        query = environment("query", roughness=1.)
        archive = OrderedDict((name, environment(name, roughness=value))
                              for name, value in (("near", 0.), ("middle", 3.), ("far", 11.)))
        self.assertAlmostEqual(novelty.compute_novelty_vs_archive(archive, query, k=2), 1.5)
        self.assertAlmostEqual(novelty.compute_novelty_vs_archive(archive, query, k=5), 13. / 3.)
        # Pit and stump dimensions also participate, not just roughness.
        a = environment("a", pit_gap=[0., 0.], stump_height=[0., 0.])
        b = environment("b", pit_gap=[3., 0.], stump_height=[0., 4.])
        self.assertAlmostEqual(novelty.euclidean_distance(a, b), 5.)

    def test_reproduction_is_seeded_bounded_and_does_not_mutate_parent(self):
        args = SimpleNamespace(master_seed=11, envs=["roughness", "pit", "stump"])
        left, right = reproduction.Reproducer(args), reproduction.Reproducer(args)
        parent = EnvConfig("parent", 5., [1.2, 2.], [1, 2], [.8, 1.2],
                           [0, 1], [], [], [])
        original = copy.deepcopy(parent)
        for _ in range(20):
            child = left.mutate(parent)
            self.assertEqual(child, right.mutate(parent))
            self.assertEqual(parent, original)
            self.assertGreaterEqual(child.ground_roughness, 0.)
            self.assertLessEqual(child.ground_roughness, 10.)
            self.assertLessEqual(child.pit_gap[0], child.pit_gap[1])
            self.assertTrue(all(0. <= value <= 8. for value in child.pit_gap))
            self.assertLessEqual(child.stump_height[0], child.stump_height[1])
            self.assertTrue(all(0. <= value <= 5. for value in child.stump_height))
            self.assertEqual(child.name, reproduction.name_env_config(
                child.ground_roughness, child.pit_gap, child.stump_width,
                child.stump_height, child.stump_float, child.stair_width,
                child.stair_height, child.stair_steps))

    def test_minimal_criterion_includes_boundaries_and_rejects_outside(self):
        zoo = object.__new__(algorithm.MultiESOptimizer)
        zoo.args = SimpleNamespace(mc_lower=25., mc_upper=340.)
        for value in (25., 100., 340.):
            self.assertTrue(zoo.pass_mc(value))
        for value in (24.999, 340.001):
            self.assertFalse(zoo.pass_mc(value))

    def test_children_are_deduplicated_filtered_and_ranked_by_novelty(self):
        zoo = object.__new__(algorithm.MultiESOptimizer)
        zoo.args = SimpleNamespace(mc_lower=25., mc_upper=340.)
        zoo.env_registry = {"active": environment("active")}
        zoo.env_archive = {"origin": environment("origin")}
        parent_theta = np.array([1., 2.])
        zoo.optimizers = {"parent": SimpleNamespace(theta=parent_theta)}
        candidates = [environment("active", 9.), environment("too_hard", 8.),
                      environment("near", .6), environment("far", 1.5),
                      environment("too_easy", 7.)]
        queued = iter((env, 0, "parent") for env in candidates)
        zoo.get_new_env = lambda parents: next(queued)
        scores = {"too_hard": 10., "near": 100., "far": 150., "too_easy": 400.}
        evaluated = []

        def candidate_optimizer(env, seed, is_candidate):
            self.assertTrue(is_candidate)
            evaluated.append(env.name)

            def evaluate(theta):
                np.testing.assert_array_equal(theta, parent_theta)
                return scores[env.name]

            return SimpleNamespace(evaluate_theta=evaluate)

        zoo.create_optimizer = candidate_optimizer
        children = zoo.get_child_list(["parent"], max_children=5)
        self.assertEqual(evaluated, ["too_hard", "near", "far", "too_easy"])
        self.assertEqual([child[0].name for child in children], ["far", "near"])
        np.testing.assert_allclose([child[3] for child in children], [1.5, .6])
        np.testing.assert_array_equal(parent_theta, [1., 2.])

    def test_fifo_retirement_keeps_archive_but_removes_oldest_active_niches(self):
        zoo = object.__new__(algorithm.MultiESOptimizer)
        names = ["zeta_oldest", "alpha_second", "middle_newest"]
        zoo.env_registry = OrderedDict((name, environment(name)) for name in names)
        zoo.env_archive = zoo.env_registry.copy()
        zoo.optimizers = OrderedDict((name, SimpleNamespace()) for name in names)
        zoo.remove_oldest(2)
        self.assertEqual(list(zoo.env_registry), ["middle_newest"])
        self.assertEqual(list(zoo.optimizers), ["middle_newest"])
        self.assertEqual(list(zoo.env_archive), names)


if __name__ == "__main__":
    unittest.main()
