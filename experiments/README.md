# Preserved physical comparison data

This directory contains the nine completed conditions used by the current video: three seeds (`24582922`, `24582923`, `24582924`) × `full`, `topk`, and `no_transfer`. These are actual IsaacLab Beta CPU / Featherstone measurements on four fixed training tasks. They are not a complete open-ended POET reproduction: no environment mutation, reproduction, novelty selection, or retirement ran in this control.

`selected_main/` contains ordinary directories, with all original checkpoints, per-checkpoint evaluations, raw evaluator calls, events, progress, settings, and task fingerprints preserved byte for byte. Each condition also contains its actual batch's `source_environment.json`, `source_protocol.json`, completion report, and execution log. The original selection manifest and shared protocol are retained unchanged. Historical absolute paths in those records describe where execution happened; they are not live dependencies.

The primary comparison uses saved checkpoint **5**, the greatest common checkpoint across all nine completed runs. It matches optimization rounds, not simulated work. Terminal runs stopped at different rounds under the soft wall-clock limit. All original terminal records remain available. The return metric is oracle best-of-library coverage over four test tasks, not one universal policy or a learned policy selector. No condition reached the 30 m goal in the primary comparison.

## Integrity and provenance

- `provenance/consolidation_manifest.json` maps 178 copied files to their original locations, sizes, and SHA256 hashes. It records the predeclared execution replacements and what was intentionally omitted.
- `verification/consolidation_validation.json` confirms that all nine raw-data summaries, all checkpoint results, grouped metrics, costs, environment objects, and the video replay checkpoint agree with the frozen current evidence.
- `verification/independent_physics_audit.json` passes the strict nine-condition audit, including initialization pairing, the unchanged pre-transfer prefix, raw cost accounting, proposal selection, and all six restart-prefix comparisons.
- `verification/experiment_tests.log` records eight passing analytical protocol/aggregation tests. These tests do not make physical performance claims.
- `verification/reaggregated_physics_comparison.json` is a fresh aggregation from this local copy. Only location fields and array ordering differ from `../evidence/physics_comparison.json`; scientific values are identical. The frozen evidence was not edited.

The six compact `main_seed_*` / `parallel_*` directories contain only interrupted `evaluations.jsonl` prefixes required by the strict restart audit. They are not additional conditions and must not enter aggregation. Interrupted checkpoints and pilot runs were omitted. The original selection manifest records exit 143 with cause undetermined; failed-attempt cost is not included in the nine completed runs' totals.

`heldout_terrains.json` is the exact frozen original bank, SHA256 `84c13204027e52865be3d54e471d9aa019af30568bdfd3439676cf9b3e61c7d9`. The runner uses its `rough_low`, `stump_low`, `gap_mid`, and `mixed_low` entries. `prepare_experiments.py` preserves the original bank generator; its generated launch plan predates this fixed-task control. Do not run that generator into this preserved directory.

## Recheck without simulation

Run from the repository root. Choose fresh output names when repeating an aggregation.

```bash
PYTHONDONTWRITEBYTECODE=1 python3 experiments/audit_physics_independent.py \
  experiments/selected_main --require-complete

PYTHONDONTWRITEBYTECODE=1 python3 experiments/aggregate_physics_results.py \
  --inputs experiments/selected_main \
  --output /tmp/poet_reaggregation_new.json

PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover \
  -s experiments -p test_physics_comparison.py -v
```

The byte-for-byte executed scripts are in `provenance/executed_scripts/`; they are the files that match `runner_sha256` and `recorder_sha256` in the preserved records. When this directory moved from the presentation project into this repository, the active top-level copies changed only in their path constants, so they run from any clone location. `provenance/split_path_changes.patch` is the complete diff. New runs therefore record a new runner hash and cannot be resumed from or pooled with the preserved records. The strict audit checks source hashes against `isaaclab_port/` in this repository.

## Run a new physical comparison

These commands launch real simulation and create new outputs. They were not run during this consolidation. Use the existing IsaacLab environment, not the voice-generation `.venv`.

```bash
POET_MAX_SECONDS=900 bash experiments/reproduce_matrix.sh \
  /absolute/path/to/a/NEW_physics_output
```

The launcher uses `~/env_isaaclab/bin/python` by default, starts nine processes, requests 20 optimization rounds, and applies the 900-second soft limit at iteration boundaries. Concurrent-host timing and that stopping rule can produce different terminal rounds. The actual selected runs used four worlds, four ES pairs, a 600-control-step horizon, and 16 Featherstone substeps. Preserve the recorded simulator version and do not relabel these Beta CPU measurements as GA or GPU evidence.

For a new replay of the same preselected policy and task used by the video:

```bash
PYTHONDONTWRITEBYTECODE=1 ~/env_isaaclab/bin/python \
  experiments/extract_physics_replay.py \
  --checkpoint experiments/selected_main/full_seed_24582922/checkpoint_005.json \
  --policy-index 0 --heldout-task-index 0 --max-steps 600 \
  --output /tmp/poet_replay_new.json
```

The replay tool verifies source and installed simulator fingerprints before simulation. The existing replay is reconstructed from recorded simulator states; it is not a viewport capture. `build_experiment_report.py` rebuilds the report from the frozen canonical evidence, but is not run by the audit commands above. Its report text was translated to English when this repository was made English-only; its figures and numbers come from the same evidence.
