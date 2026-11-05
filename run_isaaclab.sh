#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
isaac_python="${ISAACLAB_PYTHON:-$HOME/env_isaaclab/bin/python}"
if [[ ! -x "$isaac_python" ]]; then
  printf 'Missing requested environment: %s\n' "$isaac_python" >&2
  exit 2
fi
cd "$project_dir"
export PYTHONNOUSERSITE=1
exec "$isaac_python" train_isaaclab.py "$@"
