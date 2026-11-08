#!/usr/bin/env python3
"""Run a policy trained in official Box2D inside the GPU port, as a fidelity check.

Run from the repository root with the Isaac Lab Python:
    ~/env_isaaclab/bin/python tools/eval_box2d_policy_gpu.py runs/x/x.flat.best.json
Repeats the policy across worlds, which also shows how repeatable a rollout is.
"""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / ".cache"))
os.environ.setdefault("WARP_CACHE_PATH", str(ROOT / ".cache" / "warp"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("policies", type=Path, nargs="+")
    parser.add_argument("--num_envs", type=int, default=64)
    parser.add_argument("--seed", type=int, default=24582922)
    from isaaclab_tasks.utils import add_launcher_args, launch_simulation
    add_launcher_args(parser)
    args = parser.parse_args()
    args.device = args.device or "cuda:0"
    import numpy as np
    import warp as wp
    wp.config.kernel_cache_dir = os.environ["WARP_CACHE_PATH"]
    from isaaclab_port.walker.engine import RolloutEngine
    from isaaclab_port.walker.env import WalkerEnv, make_cfg
    from isaaclab_port.walker.poet import PARAMETER_COUNT
    from isaaclab_port.walker.terrain import EnvConfig

    cfg = make_cfg(num_envs=args.num_envs, device=args.device)
    with launch_simulation(cfg, args):
        env = WalkerEnv(cfg)
        engine = RolloutEngine(env, PARAMETER_COUNT)
        for path in args.policies:
            theta = np.array(json.loads(path.read_text())[0])
            batch = engine.run([(EnvConfig(), args.seed, theta) for _ in range(args.num_envs)])
            print(json.dumps(dict(policy=path.name, mean=float(batch.returns.mean()), std=float(batch.returns.std()),
                                  min=float(batch.returns.min()), max=float(batch.returns.max()),
                                  mean_length=float(batch.lengths.mean()), finished=float(batch.finished.mean()))), flush=True)
        sys.stdout.flush()
        os._exit(0)


if __name__ == "__main__":
    main()
