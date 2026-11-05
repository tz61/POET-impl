#!/usr/bin/env bash
# Converge the requested dedicated environment to the verified official release.
# Run from a terminal with network access and write access to ~/env_isaaclab.
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
isaac_env="${ISAACLAB_ENV:-$HOME/env_isaaclab}"
isaac_source="$project_dir/vendor/IsaacLab-v3.0.0"
expected_commit=e3f7d9ff1f5d7f8280de3ece9000b7117f66bcf4
mkdir -p "$project_dir/setup-records" "$project_dir/vendor"
if [[ ! -x "$isaac_env/bin/python" ]]; then
  uv venv --python 3.12 "$isaac_env"
fi
stamp="$(date +%Y%m%d-%H%M%S)"
uv pip freeze --python "$isaac_env/bin/python" > "$project_dir/setup-records/packages-before-$stamp.txt"
if [[ ! -d "$isaac_source" ]]; then
  git clone --depth 1 --branch v3.0.0 https://github.com/isaac-sim/IsaacLab.git "$isaac_source"
fi
actual_commit="$(git -C "$isaac_source" rev-parse HEAD)"
if [[ "$actual_commit" != "$expected_commit" ]]; then
  printf 'Source commit differs from verified v3.0.0: %s\n' "$actual_commit" >&2
  exit 2
fi
if [[ -n "$(git -C "$isaac_source" status --porcelain)" ]]; then
  printf 'Isaac Lab source has local edits; refusing to sync an unverified tree.\n' >&2
  exit 2
fi
# Exact sync follows the official lockfile. It can remove old/unselected packages
# from this dedicated env. Existing ~/IsaacLab source is never changed.
# Include the full simulator to keep its USD/runtime stack consistent with GA.
UV_PROJECT_ENVIRONMENT="$isaac_env" uv sync --project "$isaac_source" --locked --extra isaacsim --extra importers
uv pip freeze --python "$isaac_env/bin/python" > "$project_dir/setup-records/packages-after-$stamp.txt"
cd "$project_dir"
"$isaac_env/bin/python" -m isaaclab_port.doctor | tee "$project_dir/setup-records/environment-after-$stamp.json"
