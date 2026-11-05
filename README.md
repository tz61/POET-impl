# POET on Isaac Lab 3

A local port of [Paired Open-Ended Trailblazer (POET)](https://arxiv.org/abs/1901.01753) to a planar biped in Isaac Lab 3, with one modification: an optional screen that limits which donors get a one-step adaptation during periodic transfer. Course project for CS 8803 Deep Reinforcement Learning, Team 33 / Paper 44. Slides and video materials are in a separate repository, POET-slides.

It keeps POET's core mechanism: paired environments and policies, evolution strategies (ES), the minimal criterion, novelty, and direct and proposal transfer. The robot dynamics, terrain representation, and observations all differ, so **this is not a numeric reproduction of the paper's Box2D experiments**, and scores here cannot be compared with the paper's.

## Current status

- **No policy has learned to walk.** Across the 5,880 recorded physics evaluations in `experiments/`, the best return is 11.88 and no policy reaches the goal.
- **Environment evolution has not run on physics.** A parent needs a return of at least 200 to reproduce, and no run has reached that. The physics comparison also ran with reproduction disabled.
- **The recorded failure ends on the torso pitch limit.** The preselected replay terminates at 0.20 s with torso pitch −1.32 rad (limit 1.3) and the hip still 0.88 m high; see `evidence/physics_replay.json`.
- **What has been verified:** the unit tests, the synthetic transfer mechanism test, and the nine-run fixed-task physics comparison with its audit (see [Verification](#verification)).
- **Whether Top-1 preserves transfer quality is open.** The current data cannot show it.

## Installation

Requirements: Linux, Python 3.12, and Isaac Lab 3 in a uv environment at `~/env_isaaclab` (override with `ISAACLAB_PYTHON`). A CUDA GPU is optional; the CPU path uses Featherstone.

The recorded results come from Isaac Lab 3 **Beta 2**, commit `ffff603eafc6b74264a5261cc0183d6a65390d78`, with Newton 1.2.1, Warp 1.13.0, and torch 2.10.0. The internal package version `isaaclab==6.1.14` is not the product version.

`setup_isaaclab3.sh` installs the verified v3.0.0 GA release (2026-10-20, commit `e3f7d9ff1f5d7f8280de3ece9000b7117f66bcf4`). **This upgrade has not been run.** Compatibility with GA was only checked statically, against the [launcher](https://github.com/isaac-sim/IsaacLab/blob/e3f7d9ff1f5d7f8280de3ece9000b7117f66bcf4/source/isaaclab/isaaclab/app/sim_launcher.py) and [visualizer](https://github.com/isaac-sim/IsaacLab/blob/e3f7d9ff1f5d7f8280de3ece9000b7117f66bcf4/source/isaaclab/isaaclab/visualizers/visualizer_cfg.py) sources. The script downloads GA to `vendor/IsaacLab-v3.0.0`, keeps the existing `~/IsaacLab` source, saves package lists from before and after, and syncs `~/env_isaaclab` to the official lock file. The sync upgrades dependencies and removes packages outside the selected environment, and the configuration includes Isaac Sim 6.2 and importers, so the install is large.

```bash
./setup_isaaclab3.sh
```

Check the environment with:

```bash
./run_isaaclab.sh --mode doctor
```

`run_isaaclab.sh` always calls `~/env_isaaclab/bin/python`. The training entry refuses to run on an environment that does not match GA unless you pass `--allow_local_beta`, so Beta results cannot be mistaken for GA results.

## Running

Run all commands from the repository root. Each run needs a new or empty `--output` directory.

Short smoke tests, which use real physics and a very small budget:

```bash
# CPU on the installed Beta
./run_isaaclab.sh --mode smoke --allow_local_beta --device cpu \
  --solver featherstone --num_envs 2 --pairs 4 \
  --iterations 3 --output runs/my_cpu_diagnostic

# GPU
./run_isaaclab.sh --mode smoke --allow_local_beta --device cuda:0 \
  --num_envs 16 --pairs 8 --output runs/my_gpu_smoke
```

Training, resuming, and viewing:

```bash
# Small training run; each ES update uses 2 x pairs perturbation rollouts
./run_isaaclab.sh --mode train --allow_local_beta --device cuda:0 --num_envs 64 --pairs 32 \
  --iterations 1000 --output runs/pilot_seed1

# Resume from the full JSON state; iterations is the cumulative target
./run_isaaclab.sh --mode train --allow_local_beta --device cuda:0 --num_envs 64 \
  --checkpoint runs/pilot_seed1/checkpoint.json \
  --iterations 2000 --output runs/pilot_seed1_continued

# View a saved policy; this needs a graphical display
./run_isaaclab.sh --mode play --allow_local_beta --device cuda:0 --num_envs 1 \
  --checkpoint runs/pilot_seed1/checkpoint.json \
  --viz newton_gl --output runs/pilot_seed1_play

# Top-1 modification
./run_isaaclab.sh --mode train --allow_local_beta --device cuda:0 \
  --transfer_mode topk --transfer_top_k 1 \
  --iterations 1000 --output runs/topk_seed1

# Ablation with no cross-environment transfer; parent inheritance is kept
./run_isaaclab.sh --mode train --allow_local_beta --device cuda:0 --no_transfer \
  --iterations 1000 --output runs/no_transfer_seed1
```

Notes:

- Training keeps a default horizon of 2,000 steps. Control runs at 60 Hz and physics at 120 Hz. Featherstone defaults to 16 integration substeps per physics tick and MJWarp to 1; `--substeps` overrides this, but it changes the physical experiment. Featherstone with 1 substep diverged on contact, and that run is marked as rejected.
- Resuming follows the algorithm settings saved in the checkpoint; only the parallel count and cumulative rounds may change. Task source, assets, horizon, solver, physics version, and CPU/CUDA type must match the adjacent `task_fingerprint.json`, which prevents mixing results from different experiments. Copy that file along with a checkpoint.
- `--max_seconds` stops after a complete iteration and never interrupts an ES update partway.
- Rerun the nine-condition physics comparison with `experiments/reproduce_matrix.sh`; see [experiments/README.md](experiments/README.md).

## Repository layout

- `isaaclab_port/core.py`: POET population, ES, environment mutation, 5-NN novelty, transfer, and JSON checkpoints. Reuses Adam and centered ranks from the official source.
- `isaaclab_port/env.py`: Isaac Lab `DirectRLEnv` with four torques, a 30 m course, and fall, goal, and time-limit handling.
- `isaaclab_port/runtime.py`: one independent policy per parallel world; only the first episode is scored, masking extra reward after automatic reset.
- `isaaclab_port/assets/`: URDF, directly generated USDA, and the code that generates them. USDA loads by default.
- `isaaclab_port/diagnostics/`: real Isaac Lab physics diagnostics and their scripts.
- `train_isaaclab.py`: training and replay entry point.
- `experiments/`, `evidence/`: raw records of the nine physics runs, audit scripts, and frozen summaries.
- `tests/`: unit tests.
- `upstream/`: unmodified official `original_poet` source (Apache-2.0); all 23 files match their Git blobs, recorded in `UPSTREAM_SOURCE.json`.
- `ORIGINAL_BASELINE.md`: differences between the paper and the official Box2D example.
- `run_local.py`, `local_runtime/`, `vendor/box2d-source/`: an earlier Box2D entry point. It was never validated in a real Box2D run and is not the current target.
- `runs/`: the short diagnostic runs cited below. New output in this folder is ignored by git.

## Task and parameters

The robot asset is generated locally: fixed world anchor → passive x, z, and pitch joints → two legs with hip and knee motors. Planar motion comes from these joints, not from forcing positions each frame, and no robot mesh is downloaded. Parallel environments are isolated by Newton's separate world IDs and share the same coordinate origin, which removes floating-point contact differences from translating a long course to different coordinates.

The observation has 24 dimensions: torso pitch and three velocities, four leg joint angles and velocities, two foot heights above terrain, and ten terrain heights ahead. The last twelve replace the paper's contact flags and lidar. They are known terrain information, not physical laser ranging. The policy is a 24→40→40→4 tanh network.

Terrain genes are roughness, stump height, and gap width, in metres. The ground is built from 0.4 m collision tiles, and a gap is a missing tile. The mutation steps are `(0.06, 0.1, 0.4)` and the maxima `(0.6, 0.8, 2.4)`. These are port parameters, not the paper's encoding.

Defaults: at most 4 active environments; 8 candidates per generation with at most 1 admitted; transfer attempted every 25 rounds; environment reproduction attempted every 200 rounds. Reproduction needs a score of 200, and the child minimal criterion is `[50,300]`. Reaching the goal is worth about 300 progress points, minus posture and torque costs; a fall ends the episode with a −100 penalty. Whether these thresholds suit the changed dynamics still needs measurement, and they are not lowered automatically to produce results. If the flat-ground policy does not reach 200, no new environment is generated.

## The Top-K modification

`--transfer_mode full` keeps complete transfer. `--transfer_mode topk --transfer_top_k 1` adapts only the donor with the highest direct score for each target. All direct candidates still compete in the final selection, and child environment initialization is unchanged. `--no_transfer` still disables all cross-environment transfer. Top-1 can only differ from Full with at least 3 active environments.

Old checkpoint JSON without the new fields defaults to full. Resuming keeps the saved mode, and a conflicting explicit argument is an error. The algorithm source is part of the task fingerprint, so older physics records cannot stand in for rollouts of newer code.

The [synthetic mechanism test](evidence/mechanism_comparison.json) and the [nine physics runs with their raw data, audit, and recheck commands](experiments/README.md) are in this repository. The physics comparison covers four fixed tasks, three seeds, and three transfer modes. No policy reached the goal and environment reproduction did not run, so it is not a complete open-ended POET reproduction.

## Output files

A completed experiment directory contains `environment.json`, `arguments.json`, `settings.json`, `task_fingerprint.json`, `progress.jsonl`, `events.jsonl`, `checkpoint.json`, and `summary.json`. A simulation failure records `failure.json` and writes no result.

- `evaluations`: cumulative algorithm evaluations, including those before a checkpoint when resuming.
- `requested_episodes`: policies this process actually requested to evaluate.
- `first_episode_steps`: first-episode steps used for returns in this process.
- `simulated_world_steps`: actual parallel simulation steps in this process, including padded lanes and auto-reset environments. Do not claim equal compute from the previous two numbers alone.
- Replay results report whether the goal was reached, separately from a high score.
- `events.jsonl` separately records candidate mutations, child transfers, admissions, retirements, and periodic transfer attempts and acceptances.

## Verification

Run the unit tests with:

```bash
~/env_isaaclab/bin/python -m unittest discover -s tests -v
```

These include analytic-objective tests and are not locomotion experiments. To recheck the physics comparison without simulation, see [experiments/README.md](experiments/README.md).

Recorded checks on Beta 2:

| Check | Result and evidence |
| --- | --- |
| Upstream source integrity | Git blob SHA-1 of all 23 files matches `UPSTREAM_SOURCE.json`. |
| Unit tests | 42 pass, covering ES, transfer, checkpoints, batched policies, first-episode scoring, non-finite state detection, task compatibility, and periodic Top-K. No physics simulation needed. |
| Zero-torque ground contact | Featherstone with 16 substeps, 20 actions, states finite and motion bounded throughout; `isaaclab_port/diagnostics/biped_zero_featherstone_sub16.json`. This only validates short-term dynamics. |
| Terrain rebuild and parallel consistency | 2 independent worlds, flat → rough/gap → flat, 12 evaluations; the same policy gets the same return in both worlds and after rebuild; collision tiles go 75 → 73 → 75; `isaaclab_port/diagnostics/terrain_switch_spacing0.json`. Crossing obstacles was not validated. |
| ES on CPU | `runs/isaaclab_cpu_stable/summary.json`: 3 rounds, 4 perturbation pairs each, 27 episodes, best return −98.576367. The policy fell early. |
| ES on GPU | `runs/presentation_v2_gpu_probe/summary.json`: MJWarp, 3 rounds, 27 episodes, best return −98.73. A pipeline check only. |
| Checkpoint reload | `runs/isaaclab_cpu_replay/summary.json`: reloaded from a checkpoint and re-evaluated in one environment, matching the saved best return exactly; ended at 13 steps, short of the goal. |
| Physics comparison | Nine conditions, strict audit passing; see [experiments/README.md](experiments/README.md). |

Early `runs/isaaclab_cpu_smoke` used 1 substep, was unstable, and is marked `rejected_unstable_dynamics`, so it is not evidence of training success. `terrain_switch_spacing34.json` keeps the parallel differences from the original scene translation, which led to independent worlds with a shared origin.

GA training and graphical replay remain unverified. A full paper-level conclusion would still need sufficient training, independent seeds, and budget-matched comparisons with direct ES, a direct-path curriculum, and no-transfer POET. This project claims no reproduction of the paper's hard-terrain solve rate, coverage, or advantage.

## License and attribution

`upstream/` is the official POET code under its own Apache-2.0 `LICENSE` and `NOTICE`. `vendor/box2d-source/` keeps its own zlib `LICENSE`.
