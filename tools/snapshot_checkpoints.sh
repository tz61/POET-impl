#!/bin/bash
# Keep a copy of each run's checkpoint every 100 iterations, for equal-compute comparisons.
# train_walker.py overwrites checkpoint.json, so nothing else keeps the history. Safe to restart.
# Arguments: run root, snapshot folder, systemd scope to follow.
root=${1:-runs/walker_main}
snapshots=${2:-runs/walker_main_snapshots}
scope=${3:-walker-main}
cd "$(dirname "$0")/.."
mkdir -p "$snapshots"
while systemctl --user is-active --quiet "$scope.scope"; do
    for checkpoint in "$root"/*/checkpoint.json; do
        run=$(basename "$(dirname "$checkpoint")")
        iteration=$(python3 -c "import json,sys;print(json.load(open(sys.argv[1]))['iteration'])" "$checkpoint" 2>/dev/null) || continue
        target="$snapshots/${run}_$(printf %05d "$iteration").json"
        if [ $((iteration % 100)) -eq 0 ] && [ ! -e "$target" ]; then
            cp "$checkpoint" "$target.tmp" && mv "$target.tmp" "$target"
        fi
    done
    sleep 60
done
