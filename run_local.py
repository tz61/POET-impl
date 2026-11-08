#!/usr/bin/env python3
"""Run the original POET implementation with an explicit local resource budget."""
import argparse
import csv
import json
import logging
import os
from pathlib import Path
import pickle
import random
import sys
import time
from types import SimpleNamespace

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
ROOT = Path(__file__).resolve().parent

PROFILES = {
    "smoke": dict(n_iterations=3, batches_per_chunk=2, batch_size=4,
                  eval_batches_per_step=1, max_num_envs=4, max_children=8,
                  steps_before_transfer=25, adjust_interval=8, noise_count=1_000_000),
    "pilot": dict(n_iterations=100, batches_per_chunk=4, batch_size=8,
                  eval_batches_per_step=1, max_num_envs=4, max_children=8,
                  steps_before_transfer=25, adjust_interval=8, noise_count=1_000_000),
    "paper": dict(n_iterations=25200, batches_per_chunk=256, batch_size=1,
                  eval_batches_per_step=5, max_num_envs=20, max_children=512,
                  steps_before_transfer=25, adjust_interval=8, noise_count=250_000_000),
}


def write_json(path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def trim_logs(output, next_iteration):
    """Drop rows past the checkpoint, and any partial last line, so a resumed run repeats nothing."""
    path = output / "metrics.csv"
    if path.exists():
        lines = path.read_text().splitlines()
        path.write_text("".join(line + "\n" for line in lines[:1] + [
            line for line in lines[1:] if int(line.split(",", 1)[0]) < next_iteration]))
    path = output / "events.jsonl"
    if path.exists():
        kept = []
        for line in path.read_text().splitlines():
            try:
                if json.loads(line)["iteration"] < next_iteration:
                    kept.append(line)
            except json.JSONDecodeError:
                pass
        path.write_text("".join(line + "\n" for line in kept))


def reject_unbuildable(evaluate, report):
    """Score a course Box2D cannot build as failing the minimal criterion.

    Once a stump height range starts at 0, the official terrain code can draw a stump
    so thin that Box2D's float32 area check raises, which would end the whole run.
    """
    def wrapped(self, theta):
        try:
            return evaluate(self, theta)
        except AssertionError as error:
            if "b2_epsilon" not in str(error):
                raise
            report(self.optim_id)
            return float("-inf")
    return wrapped


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--profile", choices=PROFILES, default="smoke")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--iterations", type=int, help="Total iteration count (also when resuming)")
    p.add_argument("--seed", type=int, default=24582922)
    p.add_argument("--workers", type=int, default=1, help="Rollout processes; 1 uses the serial adapter")
    p.add_argument("--max-envs", type=int, help="Override the active environment cap")
    p.add_argument("--max-children", type=int, help="Override mutation candidates per reproduction event")
    p.add_argument("--resume", type=Path, help="Only load checkpoints you trust (Python pickle)")
    p.add_argument("--no-transfer", action="store_true", help="Disable alternate-donor child initialization and periodic transfers")
    p.add_argument("--max-seconds", type=float, default=0, help="Stop cleanly at an iteration boundary; 0 disables limit")
    p.add_argument("--checkpoint-every", type=int, default=10)
    p.add_argument("--dry-run", action="store_true", help="Print exact configuration without imports or training")
    return p.parse_args()


def config_for(cli):
    d = dict(PROFILES[cli.profile])
    d.update(init="random", learning_rate=0.01, lr_decay=0.9999, lr_limit=0.001,
             noise_std=0.1, noise_decay=0.999, noise_limit=0.01, l2_coeff=0.01,
             eval_batch_size=1, num_workers=cli.workers, master_seed=cli.seed,
             mc_lower=50, mc_upper=300, repro_threshold=200,
             normalize_grads_by_noise_std=True, propose_with_adam=True,
             checkpointing=False, returns_normalization="centered_ranks",
             stochastic=False, envs=["stump", "pit", "roughness"], start_from=None,
             log_file=str(cli.output.resolve()), no_transfer=cli.no_transfer,
             profile=cli.profile, episode_limit=2000)
    if cli.iterations is not None:
        d["n_iterations"] = cli.iterations
    if cli.max_envs is not None:
        d["max_num_envs"] = cli.max_envs
    if cli.max_children is not None:
        d["max_children"] = cli.max_children
    if d["n_iterations"] < 1 or cli.checkpoint_every < 1:
        raise ValueError("iterations and checkpoint-every must be positive")
    return d


def main():
    cli = parse_args()
    config = config_for(cli)
    if cli.dry_run:
        print(json.dumps(config, indent=2))
        return 0
    output = cli.output.resolve()
    if not cli.resume and output.exists() and any(output.iterdir()):
        raise ValueError(f"Output is not empty: {output}; choose a new directory or --resume")

    from local_runtime.bootstrap import activate
    try:
        versions = activate(config["noise_count"], config["num_workers"])
    except ImportError as exc:
        print(f"Dependency unavailable: {exc}\nRun ./setup.sh from {ROOT} in a network-enabled terminal.", file=sys.stderr)
        return 2
    checkpoint = None
    if cli.resume:
        # Unpickling needs poet_distributed, which activate() puts on sys.path.
        with cli.resume.open("rb") as f:
            checkpoint = pickle.load(f)
        old = checkpoint["config"]
        for k in old:
            if k not in ("n_iterations", "log_file") and old[k] != config[k]:
                raise ValueError(f"Resume configuration differs: {k}")
        if config["n_iterations"] <= checkpoint["next_iteration"]:
            raise ValueError("--iterations must exceed completed checkpoint iterations")
        trim_logs(output, checkpoint["next_iteration"])
    import numpy as np
    from poet_distributed import es
    from poet_distributed.poet_algo import MultiESOptimizer
    from poet_distributed.niches.box2d import box2d as niche_module
    from poet_distributed.niches.box2d import bipedal_walker_custom as walker

    output.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        handlers=[logging.FileHandler(output / "run.log"), logging.StreamHandler()])
    logging.getLogger("poet_distributed.poet_algo").setLevel(logging.INFO)
    np.random.seed(config["master_seed"])
    random.seed(config["master_seed"])
    write_json(output / "config.json", config)
    write_json(output / "environment.json", versions)

    clock_start = time.monotonic()
    cumulative_time = checkpoint["wall_seconds"] if checkpoint else 0.0
    counters = checkpoint["counters"] if checkpoint else dict(episodes=0, environment_steps=0,
                direct_transfer_attempts=0, proposal_transfer_attempts=0, accepted_transfers=0,
                mutation_candidates=0, admitted_children=0, retired_environments=0)
    phase = {"iteration": -1}
    events_file = (output / "events.jsonl").open("a", buffering=1)

    def event(kind, **data):
        events_file.write(json.dumps(dict(iteration=phase["iteration"], event=kind, **data)) + "\n")

    original_get_chunk = es.ESOptimizer.get_chunk

    def counted_get_chunk(self, tasks):
        # Rollouts run in workers, so count them where results reach the master.
        results = original_get_chunk(self, tasks)
        for result in results:
            counters["episodes"] += int(result.lengths.size)
            counters["environment_steps"] += int(result.lengths.sum()) + int(result.lengths.size)
        return results

    es.ESOptimizer.get_chunk = counted_get_chunk
    es.ESOptimizer.evaluate_theta = reject_unbuildable(
        es.ESOptimizer.evaluate_theta, lambda name: event("candidate_unbuildable", environment=name))
    original_update = es.ESOptimizer.update_dicts_after_transfer
    original_pick = es.ESOptimizer.pick_proposal

    def recorded_update(self, source_optim_id, source_optim_theta, stats, keyword):
        counters[("direct" if keyword == "theta" else "proposal") + "_transfer_attempts"] += 1
        event("transfer_evaluated", target=self.optim_id, donor=source_optim_id,
              transfer_kind=keyword, score=float(stats.eval_returns_mean), incumbent=float(self.self_evals))
        return original_update(self, source_optim_id, source_optim_theta, stats, keyword)

    def recorded_pick(self, *a, **kw):
        previous = self.theta.copy()
        donor, before = self.proposal_source, self.self_evals
        result = original_pick(self, *a, **kw)
        if not np.array_equal(previous, self.theta):
            counters["accepted_transfers"] += 1
            event("transfer_accepted", target=self.optim_id, donor=donor,
                  before=float(before), after=float(self.self_evals))
        return result

    es.ESOptimizer.update_dicts_after_transfer = recorded_update
    es.ESOptimizer.pick_proposal = recorded_pick

    class LocalPOET(MultiESOptimizer):
        def get_new_env(self, parents):
            env, seed, parent = super().get_new_env(parents)
            counters["mutation_candidates"] += 1
            self.parent_names[env.name] = parent
            event("mutation", parent=parent, child=env.name, seed=int(seed), config=env._asdict())
            return env, seed, parent

        def add_optimizer(self, env, seed, created_at=0, model_params=None):
            result = super().add_optimizer(env, seed, created_at, model_params)
            if env.name != "flat" and not getattr(self, "restoring", False):
                counters["admitted_children"] += 1
                event("admitted", name=env.name, parent=self.parent_names.get(env.name), seed=int(seed))
            return result

        def delete_optimizer(self, optim_id):
            result = super().delete_optimizer(optim_id)
            if not getattr(self, "restoring", False):
                counters["retired_environments"] += 1
                event("retired", name=optim_id)
            return result

        def pass_dedup(self, env):
            # Avoid upstream's later assertion when a retired name is rediscovered.
            return env.name not in self.env_archive

        def create_optimizer(self, env, seed, created_at=0, model_params=None, is_candidate=False):
            o = super().create_optimizer(env, seed, created_at, model_params, is_candidate)
            if config["no_transfer"] and is_candidate:
                def inherited_only(optimizers, propose_with_adam=False):
                    donor = optimizers[self.parent_names[env.name]]
                    theta = donor.theta.copy()
                    return o.evaluate_theta(theta), theta
                o.evaluate_transfer = inherited_only
            return o

    LocalPOET.parent_names = {}
    zoo = LocalPOET(SimpleNamespace(**config))
    start_iteration = 0
    if checkpoint:
        zoo.restoring = True
        for name in list(zoo.optimizers):
            zoo.delete_optimizer(name)
        zoo.env_archive.clear()
        for item in checkpoint["optimizers"]:
            zoo.add_optimizer(item["env"], item["seed"], model_params=item["state"]["theta"])
            o = zoo.optimizers[item["env"].name]
            o.__dict__.update(item["state"])
            # Keep paths/loggers created for this output, but retain all optimizer state.
            o.filename_best = str(output / (output.name + "." + o.optim_id + ".best.json"))
        zoo.env_archive = checkpoint["archive"]
        zoo.parent_names = checkpoint["parent_names"]
        zoo.env_reproducer.rs.set_state(checkpoint["reproducer_rng"])
        np.random.set_state(checkpoint["numpy_rng"])
        random.setstate(checkpoint["python_rng"])
        # Real workers reseed from the niche for each rollout, so only serial state is saved.
        if checkpoint["worker_numpy_rng"] is not None:
            zoo.fiber_pool._numpy_state = checkpoint["worker_numpy_rng"]
            zoo.fiber_pool._python_state = checkpoint["worker_python_rng"]
        zoo.restoring = False
        start_iteration = checkpoint["next_iteration"]
        event("resumed", next_iteration=start_iteration)

    def save_checkpoint(next_iteration):
        items = []
        for name, o in zoo.optimizers.items():
            state = {k: v for k, v in o.__dict__.items()
                     if k not in ("fiber_pool", "fiber_shared", "data_logger", "filename_best", "t_start")}
            items.append(dict(env=zoo.env_registry[name], seed=zoo.fiber_shared["niches"][name].seed, state=state))
        data = dict(config=config, next_iteration=next_iteration, optimizers=items,
                    archive=zoo.env_archive, parent_names=zoo.parent_names,
                    reproducer_rng=zoo.env_reproducer.rs.get_state(), numpy_rng=np.random.get_state(),
                    python_rng=random.getstate(), worker_numpy_rng=getattr(zoo.fiber_pool, "_numpy_state", None),
                    worker_python_rng=getattr(zoo.fiber_pool, "_python_state", None), counters=counters.copy(),
                    wall_seconds=cumulative_time + time.monotonic() - clock_start)
        tmp = output / "checkpoint.pkl.tmp"
        tmp.write_bytes(pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL))
        tmp.replace(output / "checkpoint.pkl")

    fields = ["iteration", "environment", "score", "best_score", "active_environments",
              "episodes", "environment_steps", "wall_seconds"]
    metrics_path = output / "metrics.csv"
    empty_metrics = not metrics_path.exists()
    metrics_file = metrics_path.open("a", newline="", buffering=1)
    writer = csv.DictWriter(metrics_file, fieldnames=fields)
    if empty_metrics:
        writer.writeheader()
    completed = start_iteration
    status = "completed"
    try:
        for i in range(start_iteration, config["n_iterations"]):
            phase["iteration"] = i
            zoo.adjust_envs_niches(i, config["adjust_interval"] * config["steps_before_transfer"],
                                  max_num_envs=config["max_num_envs"], max_children=config["max_children"], max_admitted=1)
            for o in zoo.optimizers.values():
                o.clean_dicts_before_iter()
            zoo.ind_es_step(i)
            if not config["no_transfer"] and len(zoo.optimizers) > 1 and i % config["steps_before_transfer"] == 0:
                zoo.transfer(propose_with_adam=True, checkpointing=False, reset_optimizer=True)
            wall = cumulative_time + time.monotonic() - clock_start
            for name, o in zoo.optimizers.items():
                if not np.isfinite(o.theta).all() or not np.isfinite(o.self_evals):
                    raise FloatingPointError(f"Non-finite policy or score in {name}")
                o.save_to_logger(i)
                writer.writerow(dict(iteration=i, environment=name, score=float(o.self_evals),
                                     best_score=float(o.best_score), active_environments=len(zoo.optimizers),
                                     episodes=counters["episodes"], environment_steps=counters["environment_steps"],
                                     wall_seconds=wall))
            metrics_file.flush()
            completed = i + 1
            if completed % cli.checkpoint_every == 0:
                save_checkpoint(completed)
            if cli.max_seconds and time.monotonic() - clock_start >= cli.max_seconds:
                status = "time_budget_reached"
                break
    finally:
        # On failure retain the previous fully completed checkpoint, not partial RNG/state.
        metrics_file.close()
        events_file.close()
    save_checkpoint(completed)
    summary = dict(status=status, completed_iterations=completed, requested_iterations=config["n_iterations"],
                   counters=counters, active_environments=list(zoo.optimizers),
                   archive_size=len(zoo.env_archive),
                   current_scores={n: float(o.self_evals) for n, o in zoo.optimizers.items()},
                   best_scores={n: float(o.best_score) for n, o in zoo.optimizers.items()},
                   wall_seconds=cumulative_time + time.monotonic() - clock_start,
                   interpretation="Local execution only; no claim of reproducing the paper's performance or coverage.")
    write_json(output / "summary.json", summary)
    zoo.fiber_pool.close()
    zoo.fiber_pool.join()
    zoo.manager.shutdown()
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
