"""Tensor/batching tests with a synthetic vector environment, not physics claims."""
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from isaaclab_port.core import PARAMETER_COUNT, Terrain
from isaaclab_port.runtime import Evaluator, policy_actions


class VectorFixture:
    device = "cpu"
    max_episode_length = 5

    def __init__(self, cfg):
        self.actions = torch.zeros((2, 4))

    def reset(self, seed):
        self.actions.zero_()
        self.t = 0
        return {"policy": torch.zeros((2, 24))}, {}

    def step(self, actions):
        self.actions = actions.clone()
        self.t += 1
        rewards = torch.tensor([1. if self.t <= 2 else 100., 2. if self.t <= 3 else 100.])
        done = torch.tensor([self.t == 2, self.t == 3])
        return ({"policy": torch.zeros((2, 24))}, rewards, done, torch.zeros(2, dtype=torch.bool),
                {"terminal_success": done})

    def close(self):
        pass


class RuntimeTests(unittest.TestCase):
    def test_reset_between_batches_and_only_first_episode_counted(self):
        with patch.dict(sys.modules, {"isaaclab_port.env": SimpleNamespace(PoetBipedEnv=VectorFixture)}):
            evaluator = Evaluator(2, lambda t: None)
            scores = evaluator(Terrain(), np.zeros((3, PARAMETER_COUNT)))
            np.testing.assert_array_equal(scores, [2., 6., 2.])
            self.assertEqual(evaluator.last_episode_lengths, [2, 3, 2])
            self.assertEqual(evaluator.last_episode_successes, [True, True, True])
            self.assertEqual(evaluator.first_episode_steps, 7)
            self.assertEqual(evaluator.simulated_world_steps, 10)
            self.assertEqual(evaluator.requested_episodes, 3)
            np.testing.assert_array_equal(evaluator(Terrain(), np.zeros((1, PARAMETER_COUNT))), [2.])

    def test_batch_network_matches_independent_numpy_networks(self):
        rng = np.random.default_rng(16)
        obs = rng.normal(size=(3, 24)).astype(np.float32)
        params = rng.normal(0, .1, size=(3, PARAMETER_COUNT)).astype(np.float32)
        expected = []
        for x, p in zip(obs, params):
            offset = 0
            for inputs, outputs in [(24, 40), (40, 40), (40, 4)]:
                w = p[offset:offset + inputs * outputs].reshape(inputs, outputs)
                offset += inputs * outputs
                b = p[offset:offset + outputs]
                offset += outputs
                x = np.tanh(x @ w + b)
            expected.append(x)
        actual = policy_actions(torch.from_numpy(obs), torch.from_numpy(params)).numpy()
        np.testing.assert_allclose(actual, expected, atol=1e-6)

    def test_numerical_failure_in_active_episode_is_reported(self):
        class BrokenFixture(VectorFixture):
            def step(self, actions):
                obs, reward, done, truncated, info = super().step(actions)
                info["numerical_failure"] = torch.tensor([True, False])
                return obs, reward, done, truncated, info

        with patch.dict(sys.modules, {"isaaclab_port.env": SimpleNamespace(PoetBipedEnv=BrokenFixture)}):
            with self.assertRaisesRegex(FloatingPointError, "Non-finite simulator state"):
                Evaluator(2, lambda t: None)(Terrain(), np.zeros((2, PARAMETER_COUNT)))

    def test_reset_episode_nan_does_not_change_completed_return(self):
        class ResetNanFixture(VectorFixture):
            def step(self, actions):
                obs, reward, done, truncated, info = super().step(actions)
                if self.t > 2:
                    reward[0] = float("nan")
                    info["numerical_failure"] = torch.tensor([True, False])
                return obs, reward, done, truncated, info

        with patch.dict(sys.modules, {"isaaclab_port.env": SimpleNamespace(PoetBipedEnv=ResetNanFixture)}):
            values = Evaluator(2, lambda t: None)(Terrain(), np.zeros((2, PARAMETER_COUNT)))
            np.testing.assert_array_equal(values, [2., 6.])


if __name__ == "__main__":
    unittest.main()
