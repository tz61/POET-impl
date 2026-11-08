#!/usr/bin/env python3
"""Collect the iteration-2000 walker results into one JSON file, as plotted in POET-slides.

Usage: python3 tools/export_results.py --output walker_results.json [--runs runs]
Reads the matrix copy runs/walker_main_at_2000, runs/confirm_main, and the evaluations in
runs/final_2000 (bank scores, Box2D replays, control rescoring, noise replay).
"""
import argparse
import json
from pathlib import Path

import numpy as np

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--runs", type=Path, default=Path("runs"))
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
OUT = args.output
RUNS = args.runs
FINAL = RUNS / "final_2000"
MODES = ("no_transfer", "full", "topk", "full_confirm", "topk_confirm")
SEEDS = ("24582922", "24582923", "24582924")


def view(comparison, name):
    """Per mode and seed: courses solved by the library and by its best policy, mean library return, cost."""
    modes = comparison[name]["modes"]
    out = {}
    for mode in MODES:
        metrics = modes[mode]
        out[mode] = {seed: dict(library_solved=metrics["courses_solved_by_library"]["seeds"][seed],
                                best_policy_solved=metrics["courses_solved_by_best_policy"]["seeds"][seed],
                                library_return=round(metrics["mean_library_return"]["seeds"][seed], 2),
                                iteration=int(metrics["iteration"]["seeds"][seed]),
                                steps=int(metrics["steps"]["seeds"][seed]),
                                rollouts=int(metrics["rollouts"]["seeds"][seed])) for seed in SEEDS}
    return out


def bank(path):
    comparison = json.loads(path.read_text())
    return dict(budget_steps=comparison["budget"], equal_iteration=view(comparison, "equal_iteration"),
                equal_compute=view(comparison, "equal_compute"))


def transfer_state():
    """Late acceptance per target and transfer event, and optimizer state at iteration 2000."""
    out = {}
    for mode in MODES:
        root = RUNS / ("confirm_main" if mode.endswith("_confirm") else "walker_main_at_2000")
        out[mode] = {}
        for seed in SEEDS:
            d = root / f"{mode}_seed_{seed}"
            rows = {r["iteration"]: r for r in map(json.loads, (d / "progress.jsonl").read_text().splitlines())}
            events = [json.loads(l) for l in (d / "events.jsonl").read_text().splitlines() if l.strip()]
            accepted = {(e["iteration"], e["target"]) for e in events
                        if e["event"] == "transfer_accepted" and e["iteration"] >= 1400}
            slots = sum(rows[i]["active"] for i in range(1400, 2000, 25) if i in rows)
            pairs = json.loads((d / "checkpoint.json").read_text())["pairs"]
            theta = np.array([p["theta"] for p in pairs], float)
            unit = theta / np.linalg.norm(theta, axis=1, keepdims=True)
            cos = unit @ unit.T
            out[mode][seed] = dict(
                accepted=sum(e["event"] == "transfer_accepted" for e in events),
                rejected_by_confirmation=sum(e["event"] == "transfer_rejected" for e in events),
                late_acceptance_per_target_event=round(len(accepted) / slots, 3) if slots else 0.0,
                adam_steps_median=float(np.median([p["adam"]["t"] for p in pairs])),
                sigma_mean=round(float(np.mean([p["sigma"] for p in pairs])), 4),
                policy_cosine_mean=round(float(cos[np.triu_indices(len(theta), 1)].mean()), 3))
    return out


def noise():
    out = []
    for row in json.loads((FINAL / "transfer_noise.json").read_text()):
        entry = dict(run=row["run"], iteration=row["iteration"], niches=row["niches"], rollouts=row["rollouts"],
                     single_rollout_sd_max=round(max(row["incumbent_sd"]), 1),
                     five_rollout_mean_sd_max=round(max(row["incumbent_five_sd"]), 1))
        for key, label in (("replay_five", "5"), ("replay_twenty", "20")):
            rs = row[key]
            weight = np.array([x["accept_rate"] for x in rs])
            false = np.array([x["false_accept_share"] for x in rs])
            claimed = [x["claimed_margin"] for x in rs if x["claimed_margin"] is not None]
            real = [x["real_margin"] for x in rs if x["real_margin"] is not None]
            entry[f"samples_{label}"] = dict(accept_rate=round(float(weight.mean()), 3),
                                             false_accept_share=round(float((weight * false).sum() / weight.sum()), 3),
                                             claimed_margin=round(float(np.mean(claimed)), 1),
                                             real_margin=round(float(np.mean(real)), 1))
        out.append(entry)
    return out


def controls():
    rows = json.loads((FINAL / "controls_recheck.json").read_text())
    targets = {t["name"]: t for t in json.loads((FINAL / "targets.json").read_text())}
    out = {}
    for name, target in targets.items():
        key = name.split("env_")[1]
        out[name] = dict(config=target["config"], budget=target["budget"], level=target["level"], runs={})
        for mode in ("direct", "curriculum"):
            group = [r for r in rows if key in r["run"] and f"_{mode}_" in r["run"]]
            marked = []
            for r in group:
                progress = (FINAL / "controls" / r["run"] / "progress.jsonl").read_text().splitlines()
                niches = json.loads(progress[-1])["niches"]
                marked.append(any(v["solved_at"] >= 0 for v in niches.values()))
            out[name]["runs"][mode] = dict(
                marked_solved=int(sum(marked)), rescored_solved=int(sum(r["solved"] for r in group)), seeds=len(group),
                mean_return=round(float(np.mean([r["mean"] for r in group])), 1),
                finish_rate=round(float(np.mean([r["finish"] for r in group])), 3),
                mean_steps=int(np.mean([r["steps"] for r in group])))
    for r in json.loads((FINAL / "poet_recheck.json").read_text()):
        out[r["target"]]["poet_policy"] = dict(rescored_solved=r["solved"], mean_return=round(r["mean"], 1),
                                               finish_rate=r["finish"], sd=round(r["sd"], 1),
                                               marked_solved_at=r["solved_at"], evaluated_at=r["iteration"])
    return out


def sim_gap():
    points = []
    for line in (FINAL / "simgap.jsonl").read_text().splitlines():
        r = json.loads(line)
        mode, seed = r["run"].rsplit("_seed_", 1)
        points.append(dict(mode=mode, seed=seed, gpu=round(r["gpu_score"], 1), box2d=round(r["box2d_score"], 1),
                           box2d_finish=r["box2d_finish"]))
    return points


def official():
    row = json.loads((FINAL / "box2d_bank_official.json").read_text())["results"][0]
    return {k: row[k] for k in ("iteration", "steps", "rollouts", "policies", "courses_solved_by_library",
                                "courses_solved_by_best_policy", "mean_library_return")}


data = dict(
    note="Exported from poet-reproduction runs; see sources. GPU bank: 20 rollouts per course, solved needs a "
         "mean return >= 230 and a majority of finishes. Box2D bank: one deterministic episode per course.",
    sources=dict(gpu_bank="runs/final_2000/comparison20.json", box2d_bank="runs/final_2000/comparison_box2d.json",
                 official_box2d_library="runs/final_2000/box2d_bank_official.json",
                 sim_gap="runs/final_2000/simgap.jsonl", controls="runs/final_2000/controls_recheck.json",
                 poet_on_targets="runs/final_2000/poet_recheck.json", noise="runs/final_2000/transfer_noise.json",
                 transfer_state="runs/walker_main_at_2000, runs/confirm_main"),
    settings=dict(iterations=2000, paper_iterations=25200, worlds=8192, seeds=list(SEEDS), courses=128,
                  confirm_rollouts=40, eval_rollouts=5),
    gpu_bank=bank(FINAL / "comparison20.json"),
    box2d_bank=bank(FINAL / "comparison_box2d.json"),
    official_box2d_library=official(),
    sim_gap=sim_gap(),
    controls=controls(),
    noise=noise(),
    transfer_state=transfer_state(),
)
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(data, indent=1) + "\n")
print(OUT, OUT.stat().st_size)
