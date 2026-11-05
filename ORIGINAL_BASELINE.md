# Original POET: local reproduction and small-scale experiments

> This file keeps the paper and code comparison from before the move to Isaac Lab. The current target, entry points, and measured status are in [README.md](README.md). The Box2D installation and experiment status below are early preparation, and no real Box2D run was completed.

This directory targets the original [POET paper by Wang et al. (arXiv:1901.01753)](https://arxiv.org/abs/1901.01753), not Enhanced POET. The goal was to first run the original environment and algorithm locally, validate the implementation, and then run explicitly reduced-scale experiments. A successful short run does not mean reproducing the paper's hard terrains, coverage, or statistical conclusions.

**Status: the upstream code has been obtained and compared with the paper; the local compatibility layer, offline Box2D dependency, and run validation were still in preparation. This document records no completed training results.** Later measurements should go in the table at the end, with the actual configuration and logs saved alongside.

## Code sources and layout

- `upstream/`: original algorithm source, including the custom Box2D bipedal walker environment, ES, environment mutation, and transfer logic. The exact branch and commit are in `UPSTREAM_SOURCE.json`; the paper's name alone does not make this source identical to every setting of the 2019 experiments.
- `run_local.py`: local entry point, planned to offer `--profile smoke|pilot|paper --output PATH`.
- `local_runtime/`: compatibility code for local runs and legacy dependency interfaces, kept separate from the algorithm source.
- `runs/`: suggested experiment output directory; each run uses a separate subdirectory holding parameters, logs, environment definitions, and policies.

`upstream/README.md` and `upstream/run_poet_local.sh` are upstream's usage notes and example script. That script writes to the home directory and launches a fairly large run, so local experiments start from this directory's entry point. Upstream's license is in `upstream/LICENSE` and `upstream/NOTICE`.

## What the algorithm reproduces

POET starts from one flat environment and a random policy, and maintains multiple environment-policy pairs at once. In each round it optimizes each policy separately, periodically generates mutated child environments from environments that have made progress, and admits children into the population after difficulty filtering and novelty ranking. When an old environment retires, its encoding stays in the novelty archive.

Transfer is an important part of the original algorithm: for a target environment, it tests both the other environments' current policies and proposals from those policies after one ES step in the target environment, accepting only if they beat the incumbent. Environment mutation, policy selection for new environments, and transfer between existing environments are different operations, and logs must record them separately.

The original novelty comes from Euclidean distance between environment encodings, averaged over the 5 nearest neighbors, without Enhanced POET's agent-performance-based encoding. This keeps the original custom terrain; replacing it with the standard Gymnasium `BipedalWalker` and still calling it the same experiment would be wrong.

## The paper's settings and the official example differ

Below, "paper" means page numbers in the v3 PDF that was read, and "official example" means `run_poet_local.sh` in this directory, along with the `master.py` and algorithm defaults it inherits.

| Item | Paper | Official example |
|---|---|---|
| Policy network | 24→40→40→4, tanh; p.9 | Same; NumPy implementation |
| ES perturbation evaluations per update | 512; p.9 | 256 positive/negative pairs, 512 rollouts in all |
| Learning rate | 0.01, times 0.9999, floor 0.001; p.9 | Same |
| ES noise standard deviation | 0.1, times 0.999, floor 0.01; p.9 | Same |
| Return processing | Rank normalization; p.6 | `centered_ranks`, gradient divided by the noise standard deviation |
| Active environment cap | 20; p.14 | 25 |
| Main experiment iterations | Up to 25,200; p.14 | 50,000 |
| Reproduction eligibility | Return ≥200; p.10 | Return ≥200 |
| Child minimal criterion | Return in [50,300]; p.10 | Inherited command-line default [25,340] |
| Children per generation | 512; p.10 | At most 8 tried, at most 1 admitted |
| Transfer interval | Given as a parameter, no value listed | Every 25 iterations |
| Environment mutation interval | Given as a parameter, no value listed | `adjust_interval=8` × transfer interval 25, every 200 iterations |
| Workers and compute | 256 CPU cores per run; p.14 | 10 worker processes |
| Gap width encoding maximum | (10,10); p.10 Table 1 | (8,8) |
| Evaluations | Implementation parameter not fully listed | 5 episodes per policy evaluation |
| Main random seed | Not given as an example script parameter | 24582922 |

The official example enables the `stump pit roughness` obstacle types. `stochastic` defaults to False, so a saved environment uses a fixed seed, and repeated evaluation is not testing generalization on five independent terrains.

Other implementation details to keep or record explicitly:

- Official ES uses paired positive and negative perturbations. Actual perturbation rollouts per step are `2 × batch_size × batches_per_chunk`. Upstream's `episodes_this_step` counts perturbation pairs and cannot be taken directly as actual episodes; transfer proposals and child tests add further rollouts.
- `l2_coeff` defaults to 0.01. Adam defaults to β₁=0.9, β₂=0.999, ε=1e−8. Accepting a transfer resets Adam and the noise standard deviation; the learning rate is also reset by Adam.
- In the example, periodic transfer enables Adam proposals, while `evaluate_transfer` for new children keeps the default SGD proposals. Changing this is a change in algorithm behavior and cannot be classed as a dependency compatibility fix.
- The text on candidate priority on p.10 mentions lower returns, while Algorithm 3 on pp.26–27 explicitly sorts by novelty; the official code sorts by novelty descending.
- Official mutation processes enabled obstacle types in turn; roughness changes by a `Uniform(-0.6,0.6)` increment, rounded to one decimal. Do not describe this as literally executing every mutation detail in the paper.

So `paper` should only be understood as "aligned as far as possible with the scale parameters the paper states". Everything else still follows the saved configuration, source version, and list of compatibility patches. It is not a switch that guarantees exact repetition of the paper's results.

## Isolated local runs

Dependencies should be installed in this project's isolated environment, to avoid changing coursework or the system Python. Offline Box2D and legacy Gym interface compatibility were still being validated. **Install commands were to be added once a dependency approach was tested successfully**, so no unverified install steps are given here.

Once dependencies and the entry point are ready, use that environment's Python from this directory:

```sh
python run_local.py --profile smoke --output runs/smoke
python run_local.py --profile pilot --output runs/pilot
```

These two commands are an entry point convention, not records of completed runs. Each run should use a new output directory. Do not start a long `paper` run before the measured configuration and budget are confirmed.

| Profile | Purpose | Conclusions it can support |
|---|---|---|
| `smoke` | Very few iterations and perturbation samples, checking the environment, ES, and output chain | The program runs; it does not prove the policy learned to walk |
| `pilot` | Reduced population, perturbations, and total budget, observing learning and environment generation | What happened within the reported local budget |
| `paper` | Aligned with the main scale settings the paper states | Paper conclusions can only be discussed after completing runs, evaluation, and baselines |

This machine has an i7-13700K with 16 physical cores and 24 logical CPUs. Measure actual rollout and per-ES-update time first, then choose the parallel count and total budget; a CPU model or thread count cannot replace run measurements. Each main experiment in the paper needed about 256 cores for 10 days, about 61,440 CPU core-hours, with 3 independent main experiments; baselines and ablations need more compute.

If the flat-ground policy does not reach 200 within a short budget, the original algorithm may never generate eligible offspring. Report this as "reproduction condition not yet triggered"; do not quietly lower thresholds, load pretrained policies, or manually add solved terrains to manufacture success. Diagnostic runs that change these conditions must be labeled separately.

## Metrics and baselines

Minimum validation includes: the environment can reset and step; ES positive and negative perturbation evaluation and updates work; environment definitions and seeds can be saved and reloaded; transfer is only accepted on improvement; candidates pass two difficulty filters; when active capacity is full, the oldest retires first and the archive is kept.

Truly evaluating POET's advantage also needs the comparisons below, using the same environments and clear compute budgets:

1. **Direct ES**: train from random policies on target terrains POET generated and solved. The paper runs 5 independent runs per target, up to 16,000 ES updates; pp.10–12.
2. **Direct-path curriculum**: start from flat ground, increase parameters monotonically toward a specified target terrain, and advance to the next environment at 200. The paper calculates a comparable ES budget from the target solution's ancestor path; pp.13–15.
3. **No-transfer POET**: keep environment mutation and parent inheritance, and disable direct and proposal transfer across environments; p.18. Removing only logs or only one kind of transfer does not count as this ablation.

Record at least the results of different random seeds, actual total episodes, environment steps, ES updates, wall-clock time, environment encodings and seeds, parent-child relations, and each transfer's source and returns before and after acceptance. Equal POET iterations do not mean equal compute, because active population size and transfer overhead can differ.

The paper's "solved" requires reaching the finish with a total return ≥230. Difficulty levels check stump upper bound ≥2.4, gap upper bound ≥6.0, and roughness ≥4.5; meeting 1/2/3 is challenging / very challenging / extremely challenging; p.14. Reporting only the top return, showing only success videos, or counting only generated terrains cannot replace comparisons of solved terrains and coverage.

## Measurement log: to be updated after runs

These are explicit positions for pending records, not zero-score results or claims of passing.

| Check or experiment | Status | Configuration, output, and measurements |
|---|---|---|
| Offline dependencies and Box2D import | To verify | Exact Python and dependency versions to be filled in |
| Custom environment reset/step | Not recorded | Test command and output to be filled in |
| `smoke` end-to-end run | Not run | Output directory, seed, and time to be filled in |
| Flat-ground policy learning | Not run | Return curve, episodes, and steps to be filled in |
| Child generation and admission | Not run | Generated, filtered, and admitted counts and environment encodings to be filled in |
| Periodic direct and proposal transfer | Not run | Attempts, acceptances, and return changes to be filled in |
| `pilot` multi-seed experiments | Not run | Budget and per-seed results to be filled in |
| Direct ES, curriculum, and no-transfer baselines | Not run | Budget matching method and comparison results to be filled in |
| Paper scale and coverage reproduction | Not run | Cannot be inferred from smoke or a single pilot |
