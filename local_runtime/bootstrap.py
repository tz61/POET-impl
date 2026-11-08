"""Explicit compatibility hooks; the upstream source tree stays unchanged."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def activate(noise_count=1_000_000, workers=1):
    sys.path.insert(0, str(ROOT / "upstream"))
    # One worker keeps the serial adapter; more use real processes.
    sys.path.insert(0, str(ROOT / "local_runtime" / ("parallel" if workers > 1 else "")))
    import numpy as np
    import Box2D  # fail early, before creating experiment output
    import gym
    from poet_distributed import noise as noise_module

    if not hasattr(np, "product"):
        np.product = np.prod

    def local_noise_init(self):
        # Same seeded Gaussian generator as upstream; no shared-memory server.
        self.noise = np.random.RandomState(42).randn(noise_count).astype(np.float32)

    noise_module.SharedNoiseTable.__init__ = local_noise_init
    from poet_distributed import es
    # Upstream caches by (iteration, optim_id), but candidate optimizers restart
    # iteration at 0, so a worker could evaluate a stale theta or niche seed.
    es.fiber_get_theta = getattr(es.fiber_get_theta, "__wrapped__", es.fiber_get_theta)
    es.fiber_get_niche = getattr(es.fiber_get_niche, "__wrapped__", es.fiber_get_niche)
    from poet_distributed.niches.box2d import bipedal_walker_custom as walker
    from local_runtime import legacy_seeding
    # Preserve Gym 0.17's RNG mapping even when a newer installed Gym is used.
    walker.seeding = legacy_seeding
    es.initialize_master_fiber()
    return {"numpy": np.__version__, "gym": gym.__version__,
            "Box2D": getattr(Box2D, "__version__", "unknown"),
            "python": sys.version, "noise_count": noise_count,
            "backend": "multiprocessing" if workers > 1 else "serial",
            "workers": workers, "seeding": "Gym 0.17.3"}
