#!/usr/bin/env python3
"""Replay one POET transfer event on analytic functions, with real ES proposals.

This is a synthetic mechanism check. It does not create a simulator, robot,
locomotion episode, or training result. Every counted policy evaluation is an
actual call through the production POET.evaluate/gradient/proposal code.
Run: ~/env_isaaclab/bin/python presentation/evidence/run_mechanism_comparison.py
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2] / "poet-reproduction"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from isaaclab_port.core import POET, Settings, Terrain


def quadratic(terrain, parameters):
    return 10.0 - np.square(parameters[:, 0] - (0.5 + terrain.roughness))


def two_peaks(terrain, parameters):
    # Identical analytic tasks deliberately isolate the ranking failure: x=0
    # has a better direct score than x=2.6, but x=2.6 is near a higher peak.
    x = parameters[:, 0]
    return np.exp(-np.square(x / 0.5)) + 2.0 * np.exp(-np.square((x - 3.0) / 0.4))


class CountedAnalyticEvaluator:
    def __init__(self, function):
        self.function = function
        self.batch_sizes = []

    def __call__(self, terrain, parameters):
        self.batch_sizes.append(len(parameters))
        return self.function(terrain, parameters)


class PairedNoiseReplay(POET):
    """Single-event test harness; it changes no production training behavior.

    A deterministic stream keyed by seed, target and donor lets both modes use
    the same ES perturbations for their shared proposals. Production POET uses
    a sequential RNG and is NOT modified to use this replay coupling.
    """
    def proposal(self, target, theta, use_adam=True):
        key = json.dumps([self.settings.seed, target.terrain.name, theta.tolist()],
                         separators=(",", ":")).encode()
        seed = int.from_bytes(hashlib.sha256(key).digest()[:8], "little")
        original_rng = self.rng
        self.rng = np.random.default_rng(seed)
        try:
            return super().proposal(target, theta, use_adam=use_adam)
        finally:
            self.rng = original_rng


def run_event(function, seed, mode):
    evaluator = CountedAnalyticEvaluator(function)
    events = []
    settings = Settings(seed=seed, pairs=32, learning_rate=0.05, noise=0.1, l2=0.0,
                        transfers=mode != "no_transfer",
                        periodic_transfer_mode="topk" if mode == "topk" else "full",
                        transfer_top_k=1)
    poet = PairedNoiseReplay(settings, evaluator,
                            lambda name, **data: events.append({"event": name, **data}),
                            parameter_count=1)
    poet.active = [poet.new_pair(Terrain(roughness=0.1 * i, seed=i), np.array([x]), 0)
                   for i, x in enumerate((-2.0, 0.0, 2.6, -1.0))]
    poet.archive = [pair.terrain for pair in poet.active]
    # Setup is accounted separately and does not inflate transfer-event work.
    before = []
    for pair in poet.active:
        pair.score = float(poet.evaluate(pair.terrain, pair.theta)[0])
        poet.update_best(pair)
        before.append(pair.score)
    setup_calls = len(evaluator.batch_sizes)
    setup_evaluations = poet.evaluations
    initial_policies = [pair.theta.tolist() for pair in poet.active]
    poet.transfer()
    sizes = evaluator.batch_sizes[setup_calls:]
    attempts = [item for item in events if item["event"] == "transfer_attempt"]
    result = {
        "mode": mode,
        "settings": asdict(settings),
        "target_names": [pair.terrain.name for pair in poet.active],
        "initial_policies": initial_policies,
        "before_scores": before,
        "after_scores": [pair.score for pair in poet.active],
        "after_policies": [pair.theta.tolist() for pair in poet.active],
        "setup_policy_evaluations": setup_evaluations,
        "transfer_policy_evaluations": poet.evaluations - setup_evaluations,
        "transfer_evaluator_calls": len(sizes),
        "transfer_evaluator_batch_sizes": sizes,
        "direct_evaluations": len(attempts),
        "adaptation_proposals": sum(item["proposal_selected"] for item in attempts),
        "accepted_targets": sum(item["event"] == "transfer_accepted" for item in events),
        "events": events,
    }
    assert result["transfer_policy_evaluations"] == sum(sizes)
    assert result["transfer_policy_evaluations"] == {"full": 792, "topk": 272, "no_transfer": 0}[mode]
    return result


def build_comparison():
    scenarios = []
    for name, function, formula in (
        ("smooth_quadratic", quadratic, "10 - (x - (0.5 + task_index * 0.1))^2"),
        ("two_peak_ranking_counterexample", two_peaks,
         "exp(-(x/0.5)^2) + 2*exp(-((x-3)/0.4)^2)"),
    ):
        for seed in (7, 81, 123):
            modes = {mode: run_event(function, seed, mode)
                     for mode in ("full", "topk", "no_transfer")}
            full = np.array(modes["full"]["after_scores"])
            topk = np.array(modes["topk"]["after_scores"])
            assert np.all(full >= topk - 1e-12), "Matched-noise full candidates must contain top-K candidates"
            assert modes["no_transfer"]["before_scores"] == modes["no_transfer"]["after_scores"]
            scenarios.append({
                "scenario": name, "formula": formula, "seed": seed, "modes": modes,
                "full_minus_topk_after_scores": (full - topk).tolist(),
                "screening_lost_return_on_target_zero": float(full[0] - topk[0]),
            })
    counterexamples = [case for case in scenarios if case["scenario"] == "two_peak_ranking_counterexample"]
    assert all(case["screening_lost_return_on_target_zero"] > 0.0 for case in counterexamples)
    source = ROOT / "isaaclab_port" / "core.py"
    return {
        "schema_version": 1,
        "evidence_kind": "synthetic_analytic_single_transfer_event",
        "is_physics_or_locomotion_result": False,
        "description": "Production POET transfer and ES proposals on one-dimensional analytic objectives; no simulation.",
        "scope": {
            "active_tasks": 4, "parameter_dimensions": 1, "antithetic_pairs": 32,
            "top_k": 1, "transfer_events_per_case": 1, "seed_count": 3,
            "proposal_optimizer_state": "fresh state: production proposal uses its SGD branch",
            "noise_pairing": "Replay harness keys ES perturbations by seed, target name and donor parameters; production RNG unchanged.",
            "initial_states": "Hand-constructed policies and analytic tasks; not learned populations.",
        },
        "provenance": {
            "core_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "python_version": sys.version.split()[0], "numpy_version": np.__version__,
            "reproduction_command": "~/env_isaaclab/bin/python presentation/evidence/run_mechanism_comparison.py",
        },
        "summary": {
            "policy_evaluations_per_event": {"full": 792, "topk": 272, "no_transfer": 0},
            "evaluator_calls_per_event": {"full": 36, "topk": 20, "no_transfer": 0},
            "full_to_topk_event_evaluation_reduction_fraction": (792 - 272) / 792,
            "counterexample_target_zero_return_gaps": [case["screening_lost_return_on_target_zero"] for case in counterexamples],
            "claims_supported": [
                "Top-K reduces the number of production ES proposals in this controlled transfer event.",
                "All direct donor evaluations are retained.",
                "Direct-score screening can miss a donor whose one-step adaptation is better.",
            ],
            "claims_not_supported": [
                "Robot locomotion quality, success rate, physical robustness or GA/GPU validation.",
                "End-to-end POET training speedup, learning superiority, or multi-seed physical performance.",
                "A fair fixed-budget training comparison: this script replays only one transfer event.",
            ],
        },
        "scenarios": scenarios,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("mechanism_comparison.json"))
    args = parser.parse_args()
    result = build_comparison()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"output": str(args.output), "evidence_kind": result["evidence_kind"],
                      "summary": result["summary"]}, indent=2))


if __name__ == "__main__":
    main()
