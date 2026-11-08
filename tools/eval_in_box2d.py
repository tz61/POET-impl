#!/usr/bin/env python3
"""Replay saved policies in the official Box2D environment, as a fidelity check.

Run with poet-reproduction/.venv. Reads train_walker.py checkpoints and reports each
policy's official return, length and whether it reached the finish, on that niche.
Usage: .venv/bin/python tools/eval_in_box2d.py runs/x/full_seed_1/checkpoint.json
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "upstream"))
from local_runtime import legacy_seeding
from poet_distributed.niches.box2d import bipedal_walker_custom as walker
from poet_distributed.niches.box2d.env import Env_config, bipedhard_custom
from poet_distributed.niches.box2d.model import Model, simulate

walker.seeding = legacy_seeding


def official_config(config):
    return Env_config(name="eval", ground_roughness=config.get("ground_roughness", 0),
                      pit_gap=list(config.get("pit_gap", [])), stump_width=list(config.get("stump_width", [])),
                      stump_height=list(config.get("stump_height", [])), stump_float=list(config.get("stump_float", [])),
                      stair_height=[], stair_width=[], stair_steps=[])


def evaluate(config, seed, theta):
    model = Model(bipedhard_custom)
    env_config = official_config(config)
    model.make_env(seed=seed, env_config=env_config)
    model.set_model_params(np.asarray(theta))
    returns, lengths = simulate(model, seed=seed, train_mode=False, num_episode=1, env_config_this_sim=env_config)
    env = model.env
    finish = env.hull.position[0] > (walker.TERRAIN_LENGTH - walker.TERRAIN_GRASS) * walker.TERRAIN_STEP
    return float(returns[0]), int(lengths[0]) + 1, bool(finish)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoints", type=Path, nargs="+")
    parser.add_argument("--use_best", action="store_true", help="Use each pair's best policy, not its current one")
    args = parser.parse_args()
    results = []
    for path in args.checkpoints:
        data = json.loads(path.read_text())
        for pair in data["pairs"]:
            theta = pair["best_theta"] if args.use_best and pair["best_theta"] else pair["theta"]
            score, length, finish = evaluate(pair["config"], pair["seed"], theta)
            results.append(dict(run=path.parent.name, config=pair["config"], seed=pair["seed"],
                                gpu_score=pair["best_score"] if args.use_best else pair["score"],
                                box2d_score=score, box2d_length=length, box2d_finish=finish))
            print(json.dumps(results[-1]), flush=True)
    return results


if __name__ == "__main__":
    main()
