"""Behavioral checks for the socket-free worker transport adapter."""

import importlib.util
from pathlib import Path
import random
import unittest

import numpy as np


_path = Path(__file__).resolve().parents[1] / "local_runtime" / "fiber.py"
_spec = importlib.util.spec_from_file_location("poet_serial_fiber_test", _path)
fiber = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fiber)


class SerialFiberTests(unittest.TestCase):
    def test_manager_values_are_snapshots(self):
        with fiber.get_context("spawn").Manager() as manager:
            shared = manager.dict()
            original = {"theta": np.array([1., 2.])}
            shared["optim"] = original
            original["theta"][0] = 99.
            received = shared["optim"]
            np.testing.assert_array_equal(received["theta"], [1., 2.])
            received["theta"][1] = 99.
            np.testing.assert_array_equal(shared["optim"]["theta"], [1., 2.])
            self.assertEqual(list(shared), ["optim"])
            del shared["optim"]
            self.assertEqual(len(shared), 0)

    def test_worker_seeding_does_not_change_master_random_streams(self):
        random.seed(16)
        np.random.seed(17)
        expected_python = random.Random(16).random()
        expected_numpy = np.random.RandomState(17).random_sample()

        def rollout():
            random.seed(987)
            np.random.seed(654)
            return random.random(), np.random.random_sample()

        with fiber.Pool(8) as pool:
            result = pool.apply_async(rollout)
            self.assertTrue(result.ready())
            self.assertTrue(result.successful())
            self.assertEqual(result.get(), (random.Random(987).random(),
                                          np.random.RandomState(654).random_sample()))
        self.assertEqual(random.random(), expected_python)
        self.assertEqual(np.random.random_sample(), expected_numpy)

    def test_initializer_and_worker_random_state_persist(self):
        def initializer(seed):
            np.random.seed(seed)

        with fiber.Pool(initializer=initializer, initargs=(123,)) as pool:
            actual = [pool.apply_async(np.random.random_sample).get()
                      for _ in range(3)]
        np.testing.assert_array_equal(actual, np.random.RandomState(123).random_sample(3))

    def test_jobs_observe_the_published_theta_at_submission(self):
        shared = fiber.Manager().dict()
        shared["theta"] = 1
        with fiber.Pool() as pool:
            first = pool.apply_async(lambda: shared["theta"])
            shared["theta"] = 2
            second = pool.apply_async(lambda: shared["theta"])
            self.assertEqual(first.get(), 1)
            self.assertEqual(second.get(), 2)

    def test_failed_job_restores_rng_and_reraises_from_get(self):
        random.seed(1)
        np.random.seed(2)

        def failing_job():
            random.seed(100)
            np.random.seed(200)
            raise RuntimeError("rollout failed")

        errors = []
        pool = fiber.Pool()
        failed = pool.apply_async(failing_job, error_callback=errors.append)
        self.assertFalse(failed.successful())
        with self.assertRaisesRegex(RuntimeError, "rollout failed"):
            failed.get(timeout=0)
        self.assertEqual(len(errors), 1)
        self.assertEqual(random.random(), random.Random(1).random())
        self.assertEqual(np.random.random_sample(), np.random.RandomState(2).random_sample())
        self.assertEqual(pool.apply_async(lambda x: x + 1, args=(2,)).get(), 3)
        pool.close()
        pool.join()
        with self.assertRaisesRegex(ValueError, "Pool not running"):
            pool.apply_async(lambda: None)


if __name__ == "__main__":
    unittest.main()
