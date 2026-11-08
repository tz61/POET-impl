#!/usr/bin/env python3
"""Run several POET or direct-ES runs at once on one Isaac Lab GPU simulation.

Run from the repository root with the Isaac Lab Python, for example:
    ~/env_isaaclab/bin/python train_walker.py --output runs/x --num_envs 8192 \
        --iterations 3000 --seeds 1 2 3 --modes full topk no_transfer
Each mode x seed is a thread submitting rollouts to a shared engine, so idle worlds in
one run serve another. Rerun with the same arguments and --resume to continue.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import sys
import threading
import time
import traceback

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / ".cache"))
os.environ.setdefault("WARP_CACHE_PATH", str(ROOT / ".cache" / "warp"))
SOURCES = ("isaaclab_port/walker/env.py", "isaaclab_port/walker/terrain.py", "isaaclab_port/walker/spec.py",
           "isaaclab_port/walker/engine.py", "isaaclab_port/walker/poet.py", "isaaclab_port/assets/box2d_walker.usda")


def source_hashes():
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in SOURCES}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num_envs", type=int, default=8192)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--seeds", type=int, nargs="+", default=[24582922])
    parser.add_argument("--modes", nargs="+", default=["full"],
                        choices=("full", "topk", "no_transfer", "direct", "curriculum", "full_confirm", "topk_confirm"))
    parser.add_argument("--pairs", type=int, default=256)
    parser.add_argument("--max_active", type=int, default=20)
    parser.add_argument("--children", type=int, default=512)
    parser.add_argument("--checkpoint_every", type=int, default=25)
    parser.add_argument("--targets", type=Path,
                        help="JSON list of {name, config, seed, budget?}: one course per run; budget overrides iterations")
    parser.add_argument("--resume", action="store_true")
    from isaaclab_tasks.utils import add_launcher_args
    add_launcher_args(parser)
    return parser.parse_args()


def trim_log(path, iteration):
    """Drop rows at or past iteration, and any partial last line, so a resumed run repeats nothing.

    Returns the last row kept.
    """
    if not path.exists():
        return None
    kept = []
    for line in path.read_text().splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row["iteration"] < iteration:
            kept.append((line, row))
    path.write_text("".join(line + "\n" for line, _ in kept))
    return kept[-1][1] if kept else None


def run_thread(name, folder, make_poet, engine, iterations, checkpoint_every, resume, failures, finished):
    try:
        folder.mkdir(parents=True, exist_ok=True)
        checkpoint = folder / "checkpoint.json"
        resuming = resume and checkpoint.exists()
        # Work after the last checkpoint is lost when a run stops, so its log rows go too.
        start_iteration = json.loads(checkpoint.read_text())["iteration"] if resuming else 0
        trim_log(folder / "events.jsonl", start_iteration)
        last_row = trim_log(folder / "progress.jsonl", start_iteration)
        events = (folder / "events.jsonl").open("a", buffering=1)
        progress = (folder / "progress.jsonl").open("a", buffering=1)

        def event(kind, **data):
            events.write(json.dumps(dict(event=kind, **data)) + "\n")

        def evaluate(tasks):
            batch = engine.submit(tasks)
            batch.done.wait()
            return batch

        poet = make_poet(evaluate, event, checkpoint if resuming else None)
        (folder / "settings.json").write_text(json.dumps(asdict(poet.settings), indent=2))
        start = time.monotonic() - (last_row["wall_seconds"] if last_row else 0.0)
        while poet.iteration < iterations and not poet.done and not failures:
            result = poet.step()
            result["wall_seconds"] = round(time.monotonic() - start, 1)
            progress.write(json.dumps(result) + "\n")
            if poet.iteration % checkpoint_every == 0:
                poet.save(checkpoint)
        poet.save(checkpoint)
        (folder / "summary.json").write_text(json.dumps(dict(
            status="completed" if not failures else "stopped", iteration=poet.iteration, rollouts=poet.rollouts,
            steps=poet.steps, wall_seconds=time.monotonic() - start, archive=len(poet.archive),
            active=[dict(niche=p.niche.name, config=asdict(p.niche.config), seed=p.niche.seed, score=p.score,
                         finished=p.finished, best_score=p.best_score, solved_at=p.solved_at, born=p.born,
                         parent=p.parent) for p in poet.active],
        ), indent=2))
    except Exception:
        failures.append((name, traceback.format_exc()))
    finally:
        finished.append(name)


def main():
    args = parse_args()
    args.device = args.device or "cuda:0"
    from isaaclab_tasks.utils import launch_simulation
    from isaaclab_port.doctor import inspect
    report = inspect()
    if args.resume and "curriculum" in args.modes:
        raise SystemExit("Curriculum runs do not resume; rerun them from flat ground")
    if args.output.exists() and any(args.output.iterdir()) and not args.resume:
        raise SystemExit("Output directory already contains files; choose a new path or pass --resume")
    args.output.mkdir(parents=True, exist_ok=True)
    hashes = source_hashes()
    hash_file = args.output / "source_hashes.json"
    if args.resume and hash_file.exists() and json.loads(hash_file.read_text()) != hashes:
        raise SystemExit("Source changed since these runs started; start a new output directory")
    hash_file.write_text(json.dumps(hashes, indent=2))
    if args.resume:
        with (args.output / "resumes.jsonl").open("a") as log:
            log.write(json.dumps(dict(time=time.time(), arguments=vars(args)), default=str) + "\n")
    else:
        (args.output / "environment.json").write_text(json.dumps(report, indent=2))
        (args.output / "arguments.json").write_text(json.dumps(vars(args), indent=2, default=str))

    import warp as wp
    wp.config.kernel_cache_dir = os.environ["WARP_CACHE_PATH"]
    from isaaclab_port.walker.engine import RolloutEngine
    from isaaclab_port.walker.env import WalkerEnv, make_cfg
    from isaaclab_port.walker.poet import PARAMETER_COUNT, CurriculumRun, Niche, Settings, WalkerPOET
    from isaaclab_port.walker.terrain import EnvConfig

    def maker(mode, seed, target=None):
        kwargs = dict(seed=seed, pairs=args.pairs, max_active=args.max_active, children=args.children)
        if mode.startswith("topk"):
            kwargs.update(transfer_mode="topk", transfer_top_k=1)
        elif mode == "no_transfer":
            kwargs.update(transfers=False)
        elif mode in ("direct", "curriculum"):
            kwargs.update(transfers=False, reproduction=False)
        if mode.endswith("_confirm"):
            kwargs.update(transfer_confirm_rollouts=40)
        settings = Settings(**kwargs)
        start = None
        if target:
            fields = {k: tuple(v) if isinstance(v, list) else v for k, v in target["config"].items()}
            start = Niche(EnvConfig(**fields), target["seed"])

        def make(evaluate, event, checkpoint):
            if mode == "curriculum":
                return CurriculumRun(settings, evaluate, event, start)
            if checkpoint:
                return WalkerPOET.load(checkpoint, evaluate, event)
            return WalkerPOET(settings, evaluate, event, start=start)
        return make

    cfg = make_cfg(num_envs=args.num_envs, device=args.device, terrain_capacity=8192)
    failures, finished, threads = [], [], []
    with launch_simulation(cfg, args):
        env = WalkerEnv(cfg)
        engine = RolloutEngine(env, PARAMETER_COUNT)
        targets = json.loads(args.targets.read_text()) if args.targets else [None]
        for target in targets:
            for mode in args.modes:
                for seed in args.seeds:
                    prefix = f"{target['name']}_" if target else ""
                    name = f"{prefix}{mode}_seed_{seed}"
                    thread = threading.Thread(target=run_thread, name=name, daemon=True, args=(
                        name, args.output / name, maker(mode, seed, target), engine,
                        target.get("budget", args.iterations) if target else args.iterations,
                        args.checkpoint_every, args.resume, failures, finished))
                    threads.append(thread)
                    thread.start()
        total = len(threads)
        try:
            engine.serve(lambda: len(finished) >= total or bool(failures))
        except Exception:
            failures.append(("engine", traceback.format_exc()))
        (args.output / "engine.json").write_text(json.dumps(dict(
            simulated_world_steps=engine.simulated_world_steps, first_episode_steps=engine.first_episode_steps)))
        if failures:
            (args.output / "failure.json").write_text(json.dumps(failures, indent=2))
            for name, text in failures:
                print(name, text, file=sys.stderr, flush=True)
            os._exit(1)
        os._exit(0)


if __name__ == "__main__":
    raise SystemExit(main())
