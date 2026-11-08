"""Compare the port's terrain and constants with the official environment.

Run with poet-reproduction/.venv, which has Box2D; skipped elsewhere.
"""
import random
import sys
from pathlib import Path
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "upstream"))
from isaaclab_port.walker import spec, terrain

CONFIGS = [
    terrain.EnvConfig(),
    terrain.EnvConfig(ground_roughness=2.3),
    terrain.EnvConfig(ground_roughness=0.7, pit_gap=(1.2, 2.8)),
    terrain.EnvConfig(stump_width=(1, 2), stump_height=(0.4, 1.6), stump_float=(0, 1)),
    terrain.EnvConfig(ground_roughness=4.5, pit_gap=(4.0, 7.6), stump_width=(1, 2),
                      stump_height=(2.0, 4.8), stump_float=(0, 1)),
]


class WalkerPortTests(unittest.TestCase):
    def setUp(self):
        try:
            from poet_distributed.niches.box2d import bipedal_walker_custom as walker
        except ImportError as exc:
            self.skipTest(f"Box2D is unavailable: {exc}")
        from poet_distributed.niches.box2d.env import Env_config
        from local_runtime import legacy_seeding
        walker.seeding = legacy_seeding
        self.walker, self.Env_config = walker, Env_config

    def official(self, config, seed):
        env = self.walker.BipedalWalkerCustom(self.Env_config(
            name="test", ground_roughness=config.ground_roughness, pit_gap=list(config.pit_gap),
            stump_width=list(config.stump_width), stump_height=list(config.stump_height),
            stump_float=list(config.stump_float), stair_height=[], stair_width=[], stair_steps=[]))
        env.seed(seed)
        env.reset()
        return env

    def test_terrain_matches_official_generator(self):
        for config in CONFIGS:
            for seed in (1, 77, 424242):
                env = self.official(config, seed)
                course = terrain.generate_course(config, seed)
                np.testing.assert_allclose(course.x, env.terrain_x, atol=1e-9)
                np.testing.assert_allclose(course.y, env.terrain_y, atol=1e-9)
                # Box2D stores polygons and edges in terrain; polygons have four vertices.
                polygons = [f.shape.vertices for body in env.terrain for f in body.fixtures
                            if len(f.shape.vertices) == 4]
                self.assertEqual(len(polygons), len(course.polygons))

    def test_courses_fit_the_gpu_windows(self):
        for config in CONFIGS:
            for seed in range(20):
                terrain.compile_course(terrain.generate_course(config, seed))

    def test_mass_model_matches_official_bodies(self):
        env = self.official(CONFIGS[0], 1)
        self.assertAlmostEqual(env.hull.mass, spec.HULL_MASS, places=5)
        self.assertAlmostEqual(env.legs[0].mass, spec.UPPER_MASS, places=5)
        self.assertAlmostEqual(env.legs[1].mass, spec.LOWER_MASS, places=5)
        self.assertAlmostEqual(env.legs[0].inertia, spec.UPPER_INERTIA, places=5)
        self.assertAlmostEqual(env.legs[1].inertia, spec.LOWER_INERTIA, places=5)

    def test_novelty_encoding_matches_official(self):
        from poet_distributed.novelty import env2array
        config = CONFIGS[4]
        official = self.Env_config(name="t", ground_roughness=config.ground_roughness, pit_gap=list(config.pit_gap),
                                   stump_width=list(config.stump_width), stump_height=list(config.stump_height),
                                   stump_float=list(config.stump_float), stair_height=[], stair_width=[], stair_steps=[])
        np.testing.assert_allclose(config.vector, env2array(official))


if __name__ == "__main__":
    unittest.main()
