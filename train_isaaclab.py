#!/usr/bin/env python3
"""Train or replay POET's Isaac Lab 3 port using ~/env_isaaclab."""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / ".cache"))
os.environ.setdefault("WARP_CACHE_PATH", str(ROOT / ".cache" / "warp"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "train", "play", "doctor"), default="smoke")
    parser.add_argument("--output", type=Path, default=ROOT / "runs" / "isaaclab_smoke")
    parser.add_argument("--iterations", type=int)
    parser.add_argument("--num_envs", type=int, default=32)
    parser.add_argument("--pairs", type=int, default=32, help="Antithetic pairs; twice as many ES rollouts")
    parser.add_argument("--seed", type=int, default=24582922)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--no_transfer", action="store_true")
    parser.add_argument("--transfer_mode", choices=("full", "topk"), default=None,
                        help="Periodic transfer only; defaults to full for a new run, checkpoint value on resume")
    parser.add_argument("--transfer_top_k", type=int, default=None,
                        help="Donors adapted per target in topk mode (default 1); all direct donors remain candidates")
    parser.add_argument("--episode_steps", type=int, default=2000)
    parser.add_argument("--allow_local_beta", action="store_true", help="Explicit development check; not a GA validation")
    parser.add_argument("--solver", choices=("mjwarp", "featherstone"), default="mjwarp")
    parser.add_argument("--substeps", type=int, help="Solver steps per 1/120 s physics tick; defaults to 16 for Featherstone, 1 for MJWarp")
    parser.add_argument("--max_seconds", type=float, default=0)
    # Prefer GA's public launcher; the fallback is only for explicit local beta checks.
    try:
        from isaaclab.app import add_launcher_args, launch_simulation
    except ImportError:
        from isaaclab_tasks.utils import add_launcher_args, launch_simulation
    add_launcher_args(parser)
    args = parser.parse_args()

    from isaaclab_port.doctor import inspect
    report = inspect()
    if args.mode == "doctor":
        print(json.dumps(report, indent=2))
        return 0
    if not report["ga_environment_confirmed"] and not args.allow_local_beta:
        parser.error("Installed source/dependencies do not match verified v3.0.0 GA. Run setup_isaaclab3.sh, or use --allow_local_beta for an explicitly labeled development test.")
    device = args.device or "cuda:0"
    args.device = device
    if device.startswith("cuda") and not report["cuda_available"]:
        parser.error("CUDA is unavailable in this process. Use a GPU-enabled terminal, or explicitly test --device cpu --solver featherstone.")
    if args.num_envs < 1 or args.pairs < 1 or args.episode_steps < 1:
        parser.error("num_envs, pairs and episode_steps must be positive")
    if args.transfer_top_k is not None and args.transfer_top_k < 1:
        parser.error("transfer_top_k must be positive")
    args.substeps = args.substeps if args.substeps is not None else (16 if args.solver == "featherstone" else 1)
    if args.substeps < 1:
        parser.error("substeps must be positive")
    if args.max_seconds < 0:
        parser.error("max_seconds must be nonnegative (0 means unlimited)")
    if device == "cpu" and args.solver == "mjwarp":
        parser.error("The supported CPU path is --solver featherstone. MJWarp requires CUDA in this runner.")
    if args.mode == "play" and args.checkpoint is None:
        parser.error("--mode play requires --checkpoint")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Output directory already contains files; choose a new output path")
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "environment.json").write_text(json.dumps(report, indent=2))
    (args.output / "arguments.json").write_text(json.dumps(vars(args), indent=2, default=str))

    from isaaclab_newton.physics import NewtonCfg, MJWarpSolverCfg, FeatherstoneSolverCfg
    from isaaclab_port.core import POET, Settings, Terrain
    from isaaclab_port.env import make_env_cfg
    from isaaclab_port.runtime import Evaluator
    from isaaclab_port.task_fingerprint import (
        build_task_fingerprint, validate_task_fingerprint, write_task_fingerprint,
    )
    import warp as wp
    wp.config.kernel_cache_dir = str(ROOT / ".cache" / "warp")

    def cfg_factory(terrain):
        solver = FeatherstoneSolverCfg() if args.solver == "featherstone" else MJWarpSolverCfg()
        physics = NewtonCfg(solver_cfg=solver, use_cuda_graph=device.startswith("cuda"),
                            num_substeps=args.substeps)
        return make_env_cfg(asdict(terrain), seed=terrain.seed, num_envs=args.num_envs,
                            device=device, physics=physics, episode_steps=args.episode_steps)

    evaluator = Evaluator(args.num_envs, cfg_factory, render=args.mode == "play")
    events = (args.output / "events.jsonl").open("w", buffering=1)

    def event(kind, **data):
        events.write(json.dumps(dict(event=kind, **data)) + "\n")

    settings = Settings(seed=args.seed, pairs=args.pairs, transfers=not args.no_transfer,
                        periodic_transfer_mode=args.transfer_mode or "full",
                        transfer_top_k=args.transfer_top_k if args.transfer_top_k is not None else 1)
    poet = POET.load(args.checkpoint, evaluator, event) if args.checkpoint else POET(settings, evaluator, event)
    if args.checkpoint:
        if args.transfer_mode is not None and args.transfer_mode != poet.settings.periodic_transfer_mode:
            parser.error("transfer_mode differs from checkpoint; resume preserves the saved algorithm")
        if args.transfer_top_k is not None and args.transfer_top_k != poet.settings.transfer_top_k:
            parser.error("transfer_top_k differs from checkpoint; resume preserves the saved algorithm")
        if args.no_transfer and poet.settings.transfers:
            parser.error("no_transfer differs from checkpoint; resume preserves the saved algorithm")
    launch_cfg = cfg_factory(poet.active[0].terrain)
    fingerprint = build_task_fingerprint(
        episode_steps=args.episode_steps, solver=args.solver, substeps=args.substeps,
        simulation_versions={"source_commit": report.get("source_commit"), **report["packages"]},
        task_config={"physics_dt": launch_cfg.sim.dt, "decimation": launch_cfg.decimation,
                     "torque_scale": launch_cfg.torque_scale,
                     "course_length": launch_cfg.course_length,
                     "device_family": device.split(":")[0],
                     "use_cuda_graph": device.startswith("cuda")},
    )
    if args.checkpoint:
        validate_task_fingerprint(args.checkpoint.parent / "task_fingerprint.json", fingerprint)
    write_task_fingerprint(args.output / "task_fingerprint.json", fingerprint)
    iterations = args.iterations if args.iterations is not None else (2 if args.mode == "smoke" else 100)
    if args.mode != "play" and iterations <= poet.iteration:
        parser.error("iterations is the total target and must exceed completed checkpoint iterations")
    (args.output / "settings.json").write_text(json.dumps(asdict(poet.settings), indent=2))
    start = time.monotonic()
    progress = (args.output / "progress.jsonl").open("w", buffering=1)
    try:
        with launch_simulation(launch_cfg, args):
            try:
                if args.mode == "play":
                    best = max(poet.active, key=lambda p: p.best_score)
                    score = float(evaluator(best.terrain, best.best_theta[None])[0])
                    result = dict(mode="play", terrain=asdict(best.terrain), return_=score,
                                  episode_lengths=evaluator.last_episode_lengths,
                                  reached_goal=evaluator.last_episode_successes)
                else:
                    result = {}
                    while poet.iteration < iterations:
                        result = poet.step()
                        result.update(evaluator.metrics(), wall_seconds=time.monotonic() - start)
                        result["best_scores"] = {p.terrain.name: p.best_score for p in poet.active}
                        progress.write(json.dumps(result) + "\n")
                        print(json.dumps(result), flush=True)
                        poet.save(args.output / "checkpoint.json")
                        if args.max_seconds and time.monotonic() - start >= args.max_seconds:
                            result["stop_reason"] = "wall_time_budget"
                            break
                result.update(environment=report, wall_seconds=time.monotonic() - start,
                              interpretation="Isaac Lab port; changed dynamics, terrain and observations. Not the Box2D paper benchmark.")
                (args.output / "summary.json").write_text(json.dumps(result, indent=2))
            finally:
                evaluator.close()
    except Exception as exc:
        (args.output / "failure.json").write_text(json.dumps(dict(error=repr(exc), environment=report), indent=2))
        raise
    finally:
        progress.close()
        events.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
