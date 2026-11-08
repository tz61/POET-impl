"""POET's loop for the walker task, evaluating whole iterations as one GPU batch.

Algorithm settings default to the paper, and mutation and novelty follow the official
code. Differences from the official driver, none of which change what is computed:
all active pairs' perturbations run in one batch rather than pair by pair, candidate
children are scored together, and a policy is scored in the same batch as its
perturbations, with a fresh score taken before any decision. A policy's score is the
mean of the official five rollouts, which here differ: GPU worlds are not perfectly
repeatable, while Box2D on a fixed seed is. Periodic transfer can screen adaptation
to the top K donors by direct score (transfer_mode="topk"), the project's modification.
"""
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "upstream"))
from poet_distributed.optimizers import Adam
from poet_distributed.stats import compute_centered_ranks

from .terrain import EnvConfig

PARAMETER_COUNT = 24 * 40 + 40 + 40 * 40 + 40 + 40 * 4 + 4


@dataclass(frozen=True)
class Niche:
    config: EnvConfig = EnvConfig()
    seed: int = 0

    @property
    def name(self):
        return self.config.name


@dataclass
class Settings:
    seed: int = 24582922
    pairs: int = 256
    learning_rate: float = 0.01
    lr_decay: float = 0.9999
    lr_limit: float = 0.001
    noise: float = 0.1
    noise_decay: float = 0.999
    noise_limit: float = 0.01
    l2: float = 0.01
    transfer_interval: int = 25
    mutation_interval: int = 200
    max_active: int = 20
    children: int = 512
    max_admitted: int = 1
    mc_lower: float = 50.0
    mc_upper: float = 300.0
    reproduce_score: float = 200.0
    solved_score: float = 230.0
    eval_rollouts: int = 5
    transfers: bool = True
    transfer_mode: str = "full"
    transfer_top_k: int = 1
    reproduction: bool = True
    transfer_confirm_rollouts: int = 0


@dataclass
class Pair:
    niche: Niche
    theta: np.ndarray
    born: int
    optimizer: Adam
    sigma: float
    parent: str = ""
    score: float = float("-inf")
    finished: bool = False
    best_score: float = float("-inf")
    best_theta: np.ndarray = None
    solved_at: int = -1


class Reproducer:
    """The official mutation operator, from poet_distributed/reproduce_ops.py."""

    def __init__(self, seed):
        self.rs = np.random.RandomState(seed)

    def _populate(self, values, default, increment=0.0, enforce=False, maximum=()):
        values = list(values)
        if len(values) == 0 or enforce:
            return list(default)
        if len(maximum) == 2:
            choices = []
            for change0 in (increment, 0.0, -increment):
                low = np.round(values[0] + change0, 1)
                if low > maximum[0] or low < default[0]:
                    continue
                for change1 in (increment, 0.0, -increment):
                    high = np.round(values[1] + change1, 1)
                    if high > maximum[1] or high < default[1]:
                        continue
                    if (change0 == 0.0 and change1 == 0.0) or low > high:
                        continue
                    choices.append([low, high])
            if choices:
                return choices[self.rs.randint(len(choices))]
        return values

    def mutate(self, parent):
        roughness = float(np.clip(np.round(parent.ground_roughness + self.rs.uniform(-0.6, 0.6), 1), 0.0, 10.0))
        pit = self._populate(parent.pit_gap, (0, 0.8), increment=0.4, maximum=(8.0, 8.0))
        enforce = len(parent.stump_width) == 0
        width = self._populate(parent.stump_width, (1, 2), enforce=enforce) if enforce else list(parent.stump_width)
        height = self._populate(parent.stump_height, (0, 0.4), increment=0.2, enforce=enforce, maximum=(5.0, 5.0))
        return EnvConfig(ground_roughness=roughness, pit_gap=tuple(float(v) for v in pit),
                         stump_width=tuple(int(v) for v in width), stump_height=tuple(float(v) for v in height),
                         stump_float=(0, 1))


class WalkerPOET:
    """evaluate(tasks) takes (config, seed, theta) triples and returns a finished Batch."""

    def __init__(self, settings, evaluate, event=None, start=None, parameter_count=PARAMETER_COUNT):
        if settings.transfer_mode not in ("full", "topk"):
            raise ValueError("transfer_mode must be 'full' or 'topk'")
        if settings.transfer_top_k < 1:
            raise ValueError("transfer_top_k must be positive")
        self.settings = settings
        self.evaluate_fn = evaluate
        self.event = event or (lambda *a, **k: None)
        self.rng = np.random.default_rng(settings.seed)
        self.reproducer = Reproducer(settings.seed)
        self.parameter_count = parameter_count
        self.iteration = 0
        self.rollouts = 0
        self.steps = 0
        niche = start or Niche(EnvConfig(), settings.seed)
        self.active = [self.new_pair(niche, self.rng.normal(0, 0.1, parameter_count), 0)]
        self.archive = {niche.config.key: niche.config}
        self.retired = []

    def new_pair(self, niche, theta, born, parent=""):
        return Pair(niche, np.array(theta), born, Adam(theta, self.settings.learning_rate), self.settings.noise, parent)

    def evaluate(self, tasks):
        batch = self.evaluate_fn(tasks)
        self.rollouts += len(tasks)
        self.steps += int(batch.lengths.sum())
        if not np.isfinite(batch.returns).all():
            raise FloatingPointError("Non-finite return")
        return batch

    def score_tasks(self, items):
        """Repeat each (niche, theta) for scoring; read_scores() averages the results."""
        return [(niche.config, niche.seed, theta) for niche, theta in items
                for _ in range(self.settings.eval_rollouts)]

    def read_scores(self, batch, count, offset=0):
        repeats = self.settings.eval_rollouts
        end = offset + count * repeats
        returns = batch.returns[offset:end].reshape(count, repeats).mean(axis=1)
        # A policy "reaches the finish" if most of its rollouts do.
        finished = batch.finished[offset:end].reshape(count, repeats).mean(axis=1) > 0.5
        return returns, finished

    def perturbations(self, niche, theta, sigma):
        """The 2P antithetic rollouts for one gradient: their noise and tasks."""
        noise = self.rng.standard_normal((self.settings.pairs, len(theta)), dtype=np.float32)
        candidates = np.concatenate((theta + sigma * noise, theta - sigma * noise))
        return noise, [(niche.config, niche.seed, c) for c in candidates]

    def gradient(self, noise, returns, sigma):
        pairs = self.settings.pairs
        ranks = compute_centered_ranks(np.stack((returns[:pairs], returns[pairs:]), axis=1))
        return noise.T @ (ranks[:, 0] - ranks[:, 1]) / (pairs * sigma)

    @staticmethod
    def record(pair, score, finished, iteration, settings):
        pair.score, pair.finished = float(score), bool(finished)
        if pair.score > pair.best_score:
            pair.best_score, pair.best_theta = pair.score, pair.theta.copy()
        if pair.solved_at < 0 and pair.finished and pair.score >= settings.solved_score:
            pair.solved_at = iteration

    def optimize_all(self):
        settings = self.settings
        requests = [self.perturbations(p.niche, p.theta, p.sigma) for p in self.active]
        # Waiting for a second batch would double each iteration's latency.
        scored = self.score_tasks([(p.niche, p.theta) for p in self.active])
        batch = self.evaluate(scored + [task for _, tasks in requests for task in tasks])
        for pair, score, finished in zip(self.active, *self.read_scores(batch, len(self.active))):
            self.record(pair, score, finished, self.iteration, settings)
        offset, count = len(scored), 2 * settings.pairs
        for index, (pair, (noise, _)) in enumerate(zip(self.active, requests)):
            start = offset + index * count
            grad = -self.gradient(noise, batch.returns[start:start + count], pair.sigma) + settings.l2 * pair.theta
            _, pair.theta = pair.optimizer.update(pair.theta, grad)
            pair.optimizer.stepsize = max(settings.lr_limit, pair.optimizer.stepsize * settings.lr_decay)
            pair.sigma = max(settings.noise_limit, pair.sigma * settings.noise_decay)

    def refresh_scores(self):
        """Score the current policies, for decisions that compare them."""
        batch = self.evaluate(self.score_tasks([(p.niche, p.theta) for p in self.active]))
        for pair, score, finished in zip(self.active, *self.read_scores(batch, len(self.active))):
            self.record(pair, score, finished, self.iteration, self.settings)

    def propose(self, requests):
        """One ES step for each (target pair or fresh niche, theta, use_adam, sigma)."""
        settings = self.settings
        prepared = [self.perturbations(niche, theta, sigma) for niche, theta, _, sigma, _ in requests]
        batch = self.evaluate([task for _, tasks in prepared for task in tasks])
        count = 2 * settings.pairs
        proposals = []
        for index, ((_, theta, use_adam, sigma, optimizer), (noise, _)) in enumerate(zip(requests, prepared)):
            grad = -self.gradient(noise, batch.returns[index * count:(index + 1) * count], sigma) + settings.l2 * theta
            if use_adam and optimizer is not None and optimizer.t > 0:
                proposals.append(optimizer.propose(theta, grad)[1])
            else:
                # Upstream's child transfer has no Adam history, so it uses plain SGD.
                proposals.append(theta - settings.learning_rate * grad)
        return proposals

    def confirm_transfers(self, replacements):
        """Rescore each winner and its incumbent on fresh rollouts before a replacement.

        The best of many noisy scores overstates itself. Box2D on a fixed seed has no such
        noise, so this restores what its comparison means on GPU worlds that do not repeat.
        """
        repeats = self.settings.transfer_confirm_rollouts
        if not repeats or not replacements:
            return replacements
        batch = self.evaluate([(target.niche.config, target.niche.seed, candidate)
                               for target, _, theta, _, _ in replacements
                               for candidate in (target.theta, theta) for _ in range(repeats)])
        returns = batch.returns.reshape(len(replacements), 2, repeats).mean(axis=2)
        finished = batch.finished.reshape(len(replacements), 2, repeats).mean(axis=2) > 0.5
        confirmed = []
        for (target, claimed, theta, source, _), (incumbent, rescored), (_, done) in zip(replacements, returns, finished):
            if rescored > incumbent:
                confirmed.append((target, float(rescored), theta, source, bool(done)))
            else:
                self.event("transfer_rejected", iteration=self.iteration, target=target.niche.name, donor=source,
                           claimed=float(claimed), incumbent=float(incumbent), rescored=float(rescored))
        return confirmed

    def transfer(self):
        """Direct scores for every donor, then adaptation for all or the top K of them."""
        settings = self.settings
        donors = [(p.niche.name, p.theta.copy()) for p in self.active]
        links = [(target, name, theta) for target in self.active for name, theta in donors if name != target.niche.name]
        batch = self.evaluate(self.score_tasks([(t.niche, theta) for t, _, theta in links]))
        direct = {(id(t), name): (score, finished)
                  for (t, name, _), score, finished in zip(links, *self.read_scores(batch, len(links)))}
        selected = []
        ranks = {}
        for target in self.active:
            ranked = sorted(((name, direct[(id(target), name)][0]) for name, _ in donors if name != target.niche.name),
                            key=lambda item: (-item[1], item[0]))
            ranks.update({(id(target), name): rank for rank, (name, _) in enumerate(ranked, start=1)})
            chosen = ranked[:settings.transfer_top_k] if settings.transfer_mode == "topk" else ranked
            selected.extend((target, name) for name, _ in chosen)
        thetas = dict(donors)
        proposals = self.propose([(t.niche, thetas[name], True, t.sigma, t.optimizer) for t, name in selected])
        batch = self.evaluate(self.score_tasks([(t.niche, proposal) for (t, _), proposal in zip(selected, proposals)]))
        scores, flags = self.read_scores(batch, len(selected))
        adapted = {(id(t), name): (score, proposal, finished)
                   for (t, name), score, proposal, finished in zip(selected, scores, proposals, flags)}
        replacements = []
        for target in self.active:
            best_score, best_theta, source, best_finished = target.score, None, "", False
            for name, theta in donors:
                if name == target.niche.name:
                    continue
                key = (id(target), name)
                direct_score, direct_finished = direct[key]
                score, proposal, finished = adapted.get(key, (None, None, False))
                self.event("transfer_attempt", iteration=self.iteration, target=target.niche.name, donor=name,
                           direct_score=float(direct_score), proposal_score=None if score is None else float(score),
                           proposal_selected=key in adapted, direct_rank=ranks[key], mode=settings.transfer_mode)
                if direct_score > best_score:
                    best_score, best_theta, source, best_finished = float(direct_score), theta, name, direct_finished
                if score is not None and score > best_score:
                    best_score, best_theta, source, best_finished = float(score), proposal, name + ":proposal", finished
            if best_theta is not None:
                replacements.append((target, best_score, best_theta.copy(), source, best_finished))
        for target, score, theta, source, finished in self.confirm_transfers(replacements):
            self.event("transfer_accepted", iteration=self.iteration, target=target.niche.name, donor=source,
                       before=target.score, after=score)
            target.theta = theta
            target.optimizer.reset()
            target.sigma = settings.noise
            self.record(target, score, finished, self.iteration, settings)

    def novelty(self, config):
        distances = sorted(float(np.linalg.norm(config.vector - other.vector)) for other in self.archive.values())
        return float(np.mean(distances[:5]))

    def reproduce(self):
        settings = self.settings
        parents = [p for p in self.active if p.score >= settings.reproduce_score]
        if not parents:
            self.event("reproduction_ineligible", iteration=self.iteration,
                       best=max(p.score for p in self.active))
            return
        candidates, seen = [], set()
        for _ in range(settings.children):
            parent = parents[self.reproducer.rs.randint(len(parents))]
            config = self.reproducer.mutate(parent.niche.config)
            if config.key in self.archive or config.key in seen:
                continue
            seen.add(config.key)
            candidates.append((Niche(config, int(self.rng.integers(1000000))), parent))
        batch = self.evaluate(self.score_tasks([(n, p.theta) for n, p in candidates]))
        passing = []
        for (niche, parent), score in zip(candidates, self.read_scores(batch, len(candidates))[0]):
            if settings.mc_lower <= score <= settings.mc_upper:
                passing.append((self.novelty(niche.config), niche, parent, float(score)))
        self.event("mutation", iteration=self.iteration, candidates=len(candidates), passed_mc=len(passing))
        passing.sort(key=lambda item: item[0], reverse=True)
        admitted = 0
        for novelty, niche, parent, inherited in passing:
            donors = self.active if settings.transfers else [parent]
            thetas = [d.theta.copy() for d in donors]
            if settings.transfers:
                thetas += self.propose([(niche, d.theta, False, settings.noise, None) for d in donors])
            batch = self.evaluate(self.score_tasks([(niche, theta) for theta in thetas]))
            scores, flags = self.read_scores(batch, len(thetas))
            best = int(np.argmax(scores))
            score = float(scores[best])
            if not settings.mc_lower <= score <= settings.mc_upper:
                continue
            pair = self.new_pair(niche, thetas[best], self.iteration, parent.niche.name)
            self.record(pair, score, flags[best], self.iteration, settings)
            self.active.append(pair)
            self.archive[niche.config.key] = niche.config
            self.event("admitted", iteration=self.iteration, niche=niche.name, config=asdict(niche.config),
                       seed=niche.seed, parent=parent.niche.name, novelty=novelty, inherited=inherited, score=score)
            while len(self.active) > settings.max_active:
                retired = self.active.pop(0)
                self.retired.append(self.pair_record(retired))
                self.event("retired", iteration=self.iteration, niche=retired.niche.name)
            admitted += 1
            if admitted >= settings.max_admitted:
                break

    def step(self):
        settings = self.settings
        i = self.iteration
        if settings.reproduction and i > 0 and i % settings.mutation_interval == 0:
            self.refresh_scores()
            self.reproduce()
        self.optimize_all()
        if settings.transfers and len(self.active) > 1 and i % settings.transfer_interval == 0:
            self.refresh_scores()
            self.transfer()
        self.iteration += 1
        return dict(iteration=i, active=len(self.active), archive=len(self.archive), rollouts=self.rollouts,
                    steps=self.steps, niches={p.niche.name: dict(score=round(p.score, 3), finished=p.finished,
                                                                  solved_at=p.solved_at) for p in self.active})

    @property
    def done(self):
        return False

    @staticmethod
    def pair_record(p):
        return dict(config=asdict(p.niche.config), seed=p.niche.seed, theta=p.theta.tolist(), born=p.born,
                    parent=p.parent, sigma=p.sigma, score=p.score, finished=p.finished,
                    best_score=p.best_score, solved_at=p.solved_at,
                    best_theta=None if p.best_theta is None else p.best_theta.tolist(),
                    adam={k: getattr(p.optimizer, k).tolist() if isinstance(getattr(p.optimizer, k), np.ndarray)
                          else getattr(p.optimizer, k) for k in ("t", "m", "v", "stepsize")})

    def snapshot(self):
        pairs = [self.pair_record(p) for p in self.active]
        return dict(format=1, implementation="WalkerPOET", settings=asdict(self.settings), iteration=self.iteration,
                    rollouts=self.rollouts, steps=self.steps, rng=self.rng.bit_generator.state,
                    reproducer_rng=[self.reproducer.rs.get_state()[0], self.reproducer.rs.get_state()[1].tolist(),
                                    *self.reproducer.rs.get_state()[2:]],
                    archive=[asdict(c) for c in self.archive.values()], pairs=pairs, retired=self.retired)

    def save(self, path):
        path = Path(path)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(self.snapshot(), allow_nan=True))
        temporary.replace(path)

    @classmethod
    def load(cls, path, evaluate, event=None):
        data = json.loads(Path(path).read_text())
        if data["format"] != 1:
            raise ValueError("Unsupported checkpoint version")
        obj = cls(Settings(**data["settings"]), evaluate, event)
        obj.iteration, obj.rollouts, obj.steps = data["iteration"], data["rollouts"], data["steps"]
        obj.rng.bit_generator.state = data["rng"]
        state = data["reproducer_rng"]
        obj.reproducer.rs.set_state((state[0], np.array(state[1], dtype=np.uint32), *state[2:]))

        def config(item):
            return EnvConfig(**{k: tuple(v) if isinstance(v, list) else v for k, v in item.items()})

        obj.archive = {config(c).key: config(c) for c in data["archive"]}
        obj.retired = data.get("retired", [])
        obj.active = []
        for item in data["pairs"]:
            pair = obj.new_pair(Niche(config(item["config"]), item["seed"]), np.array(item["theta"]), item["born"], item["parent"])
            pair.sigma, pair.score, pair.finished = item["sigma"], item["score"], item["finished"]
            pair.best_score, pair.solved_at = item["best_score"], item["solved_at"]
            pair.best_theta = None if item["best_theta"] is None else np.array(item["best_theta"])
            for key, value in item["adam"].items():
                setattr(pair.optimizer, key, np.array(value) if key in ("m", "v") else value)
            obj.active.append(pair)
        return obj


class CurriculumRun(WalkerPOET):
    """The paper's direct-path control: raise obstacles monotonically toward one target.

    Starts on flat ground with one policy. Once the score reaches the reproduction
    threshold, each encoding value holds or takes one mutation step toward the target
    with equal chance. Stops when the target is reached and solved, or at the budget.
    """
    # Largest single official mutation: roughness, pit range, stump height range.
    STEPS = np.array([0.6, 0.4, 0.4, 0.2, 0.2])
    # Scale of the official novelty code's normalize option.
    NORMALIZE = np.array([8.0, 8.0, 8.0, 3.0, 3.0])

    def __init__(self, settings, evaluate, event, target, parameter_count=PARAMETER_COUNT):
        super().__init__(settings, evaluate, event, start=Niche(EnvConfig(), target.seed),
                         parameter_count=parameter_count)
        self.target = target
        self.closest_solved = None
        self.solved_target = False
        self.rs = np.random.RandomState(settings.seed)

    @property
    def done(self):
        return self.solved_target

    def distance_to_target(self, config):
        return float(np.linalg.norm((self.target.config.vector - config.vector) / self.NORMALIZE))

    def next_config(self, config):
        current, goal = config.vector, self.target.config.vector
        while True:
            moved = np.where(self.rs.rand(5) < 0.5, np.minimum(current + self.STEPS, goal), current)
            if not np.array_equal(moved, current):
                break
        target = self.target.config
        has_stump = moved[4] > 0 or bool(target.stump_width)
        return EnvConfig(ground_roughness=float(np.round(moved[0], 3)),
                         pit_gap=(float(moved[1]), float(moved[2])) if moved[2] > 0 else (),
                         stump_width=tuple(target.stump_width) if has_stump else (),
                         stump_height=(float(moved[3]), float(moved[4])) if has_stump else (),
                         stump_float=tuple(target.stump_float) if has_stump else ())

    def step(self):
        settings = self.settings
        pair = self.active[0]
        self.optimize_all()
        config = pair.niche.config
        distance = self.distance_to_target(config)
        if pair.finished and pair.score >= settings.solved_score:
            if self.closest_solved is None or distance < self.closest_solved:
                self.closest_solved = distance
                self.event("curriculum_solved", iteration=self.iteration, config=asdict(config), distance=distance)
            if distance == 0.0:
                self.solved_target = True
        if not self.solved_target and distance > 0.0 and pair.score >= settings.reproduce_score:
            pair.niche = Niche(self.next_config(config), self.target.seed)
            pair.finished, pair.score = False, float("-inf")
            self.event("curriculum_advance", iteration=self.iteration, config=asdict(pair.niche.config),
                       distance=self.distance_to_target(pair.niche.config))
        self.iteration += 1
        return dict(iteration=self.iteration - 1, active=1, archive=1, rollouts=self.rollouts, steps=self.steps,
                    closest_solved=self.closest_solved, distance=self.distance_to_target(pair.niche.config),
                    solved_target=self.solved_target,
                    niches={pair.niche.name: dict(score=round(pair.score, 3), finished=pair.finished, solved_at=pair.solved_at)})
