#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${POET_ISAAC_PYTHON:-$HOME/env_isaaclab/bin/python}"
if [[ $# -ne 1 ]]; then
  printf '%s\n' 'Usage: bash reproduce_matrix.sh /absolute/path/to/NEW_output_directory' >&2
  exit 2
fi
OUTPUT_ROOT="$1"
if [[ -e "$OUTPUT_ROOT" ]]; then
  printf '%s\n' 'Output already exists; choose a fresh directory.' >&2
  exit 2
fi
mkdir -p "$OUTPUT_ROOT"
OUTPUT_ROOT="$(cd "$OUTPUT_ROOT" && pwd)"
PIDS=()
INPUTS=()
for seed in 24582922 24582923 24582924; do
  for mode in full topk no_transfer; do
    result="$OUTPUT_ROOT/${mode}_${seed}"
    INPUTS+=("$result")
    "$PYTHON" -u "$SCRIPT_DIR/run_physics_comparison.py" \
      --seeds "$seed" --modes "$mode" --iterations 20 \
      --max-seconds "${POET_MAX_SECONDS:-0}" --world-step-budget 0 \
      --output "$result" > "$result.log" 2>&1 &
    PIDS+=("$!")
  done
done
FAILED=0
for pid in "${PIDS[@]}"; do wait "$pid" || FAILED=1; done
if [[ "$FAILED" -ne 0 ]]; then
  printf '%s\n' 'At least one simulation failed; inspect the retained logs.' >&2
  exit 1
fi
"$PYTHON" "$SCRIPT_DIR/aggregate_physics_results.py" \
  --inputs "${INPUTS[@]}" --output "$OUTPUT_ROOT/physics_comparison.json"
printf 'Saved comparison: %s\n' "$OUTPUT_ROOT/physics_comparison.json"
