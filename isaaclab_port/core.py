"""Simulator-independent POET loop, using the upstream ES optimizer and ranks.

This port changes terrain representation and observations, not the defining loop:
paired optimization, minimal-criterion mutation, novelty, and direct/proposal transfer.
"""
from dataclasses import asdict, dataclass, field
import hashlib
import json
from pathlib import Path
import sys
from typing import Callable

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "upstream"))
from poet_distributed.optimizers import Adam
from poet_distributed.stats import compute_centered_ranks

PARAMETER_COUNT = 24 * 40 + 40 + 40 * 40 + 40 + 40 * 4 + 4


@dataclass(frozen=True)
class Terrain:
    """Terrain genotype. All three amplitudes/widths use metres."""
    roughness: float = 0.0
    stump_height: float = 0.0
    gap_width: float = 0.0
    seed: int = 0

    @property
    def vector(self):
        return np.array([self.roughness, self.stump_height, self.gap_width], dtype=float)

    @property
    def key(self):
        # Deduplicate genotypes, not random realizations of identical geometry.
        return tuple(float(x) for x in self.vector)

    @property
    def name(self):
        digest = hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()[:10]
        return "terrain_" + digest


@dataclass
class Settings:
    seed: int = 24582922
    pairs: int = 32
    learning_rate: float = 0.01
    lr_decay: float = 0.9999
    lr_limit: float = 0.001
    noise: float = 0.1
    noise_decay: float = 0.999
    noise_limit: float = 0.01
    l2: float = 0.01
    transfer_interval: int = 25
    mutation_interval: int = 200
    max_active: int = 4
    children: int = 8
    mc_lower: float = 50.0
    mc_upper: float = 300.0
    reproduce_score: float = 200.0
    transfers: bool = True
    # Only periodic transfer is screened; child initialization remains unchanged.
    periodic_transfer_mode: str = "full"
    transfer_top_k: int = 1


@dataclass
class Pair:
    terrain: Terrain
    theta: np.ndarray
    born: int
    optimizer: Adam
    sigma: float
    score: float = float("-inf")
    best_score: float = float("-inf")
    best_theta: np.ndarray | None = None


class POET:
    """Use evaluator(terrain, parameter_matrix) -> one score per policy.

    Evaluations of a terrain must reset to the same initial state, independently
    of batching. The runner records actual simulator work separately.
    """

    def __init__(self, settings: Settings, evaluator: Callable, event: Callable | None = None,
                 parameter_count: int = PARAMETER_COUNT):
        if settings.pairs < 1 or settings.max_active < 1 or settings.children < 1:
            raise ValueError("pairs, max_active and children must be positive")
        if min(settings.transfer_interval, settings.mutation_interval) < 1:
            raise ValueError("intervals must be positive")
        if settings.mc_lower > settings.mc_upper:
            raise ValueError("invalid minimal-criterion interval")
        if settings.periodic_transfer_mode not in ("full", "topk"):
            raise ValueError("periodic_transfer_mode must be 'full' or 'topk'")
        if (isinstance(settings.transfer_top_k, bool)
                or not isinstance(settings.transfer_top_k, int) or settings.transfer_top_k < 1):
            raise ValueError("transfer_top_k must be a positive integer")
        self.settings = settings
        self.evaluate_fn = evaluator
        self.event = event or (lambda *a, **k: None)
        self.rng = np.random.default_rng(settings.seed)
        self.parameter_count = parameter_count
        self.iteration = 0
        self.evaluations = 0
        terrain = Terrain(seed=settings.seed)
        theta = self.rng.normal(0, 0.1, parameter_count)
        self.active = [self.new_pair(terrain, theta, 0)]
        self.archive = [terrain]

    def new_pair(self, terrain, theta, born):
        return Pair(terrain, theta.copy(), born,
                    Adam(theta, self.settings.learning_rate), self.settings.noise)

    def evaluate(self, terrain, theta):
        matrix = np.atleast_2d(theta)
        values = np.asarray(self.evaluate_fn(terrain, matrix), dtype=float)
        if values.shape != (len(matrix),) or not np.isfinite(values).all():
            raise ValueError("Evaluator must return a finite score for every policy")
        self.evaluations += len(matrix)
        return values

    def gradient(self, terrain, theta, sigma):
        noise = self.rng.normal(size=(self.settings.pairs, len(theta)))
        # Interleaved + / - pairs match the upstream rank ordering.
        candidates = (theta[None, None, :] + sigma * noise[:, None, :] * np.array([1., -1.])[None, :, None])
        returns = self.evaluate(terrain, candidates.reshape(-1, len(theta))).reshape(-1, 2)
        ranks = compute_centered_ranks(returns)
        return noise.T @ (ranks[:, 0] - ranks[:, 1]) / (self.settings.pairs * sigma)

    def proposal(self, target, theta, use_adam=True):
        grad = -self.gradient(target.terrain, theta, target.sigma) + self.settings.l2 * theta
        if use_adam and target.optimizer.t > 0:
            return target.optimizer.propose(theta, grad)[1]
        # Newly created targets have no Adam history: upstream child transfer uses SGD.
        return theta - self.settings.learning_rate * grad

    def optimize(self, pair):
        grad = -self.gradient(pair.terrain, pair.theta, pair.sigma) + self.settings.l2 * pair.theta
        _, pair.theta = pair.optimizer.update(pair.theta, grad)
        pair.optimizer.stepsize = max(self.settings.lr_limit, pair.optimizer.stepsize * self.settings.lr_decay)
        pair.sigma = max(self.settings.noise_limit, pair.sigma * self.settings.noise_decay)
        pair.score = float(self.evaluate(pair.terrain, pair.theta)[0])
        self.update_best(pair)

    @staticmethod
    def update_best(pair):
        if pair.score > pair.best_score:
            pair.best_score = pair.score
            pair.best_theta = pair.theta.copy()

    def transfer(self):
        """Compare every direct donor, adapting all or the top K direct donors.

        Screening affects only this periodic event, never child initialization.
        A stable donor-name tiebreak determines the top-K set. Adaptation retains
        snapshot order so full mode keeps the original proposal RNG sequence.
        """
        if not self.settings.transfers:
            return
        # Freeze all donors first: acceptance must not cascade within this event.
        donors = [(p.terrain.name, p.theta.copy()) for p in self.active]
        replacements = []
        for target in self.active:
            best_score, best_theta, source = target.score, target.theta, target.terrain.name
            direct_candidates = [
                (name, theta, float(self.evaluate(target.terrain, theta)[0]))
                for name, theta in donors if name != target.terrain.name
            ]
            ranked = sorted(direct_candidates, key=lambda candidate: (-candidate[2], candidate[0]))
            ranks = {name: rank for rank, (name, _, _) in enumerate(ranked, start=1)}
            selected = {name for name, _, _ in (
                ranked[:self.settings.transfer_top_k]
                if self.settings.periodic_transfer_mode == "topk" else ranked
            )}
            for name, theta, direct in direct_candidates:
                proposal, adapted = None, None
                if name in selected:
                    proposal = self.proposal(target, theta)
                    adapted = float(self.evaluate(target.terrain, proposal)[0])
                self.event("transfer_attempt", iteration=self.iteration, target=target.terrain.name,
                           donor=name, direct_score=direct, proposal_score=adapted,
                           proposal_selected=name in selected, direct_rank=ranks[name],
                           mode=self.settings.periodic_transfer_mode)
                if direct > best_score:
                    best_score, best_theta, source = direct, theta, name
                if adapted is not None and adapted > best_score:
                    best_score, best_theta, source = adapted, proposal, name + ":proposal"
            replacements.append((target, best_score, best_theta.copy(), source))
        for target, score, theta, source in replacements:
            if score > target.score:
                self.event("transfer_accepted", iteration=self.iteration, target=target.terrain.name,
                           donor=source, before=target.score, after=score)
                target.theta, target.score = theta, score
                target.optimizer.reset()
                target.sigma = self.settings.noise
                self.update_best(target)

    def mutate(self, parent):
        v = parent.vector.copy()
        j = int(self.rng.integers(3))
        steps, maxima = np.array([0.06, 0.1, 0.4]), np.array([0.6, 0.8, 2.4])
        v[j] = np.clip(v[j] + self.rng.choice([-1., 1.]) * steps[j], 0, maxima[j])
        v = np.round(v, 3)
        return Terrain(*v, seed=int(self.rng.integers(2**31)))

    def mc(self, score):
        return self.settings.mc_lower <= score <= self.settings.mc_upper

    def reproduce(self):
        parents = [p for p in self.active if p.score >= self.settings.reproduce_score]
        if not parents:
            self.event("reproduction_ineligible", iteration=self.iteration)
            return
        known = {t.key for t in self.archive}
        candidates = []
        for _ in range(self.settings.children):
            parent = parents[int(self.rng.integers(len(parents)))]
            child = self.mutate(parent.terrain)
            if child.key in known:
                continue
            known.add(child.key)
            score = float(self.evaluate(child, parent.theta)[0])
            self.event("mutation", iteration=self.iteration, parent=parent.terrain.name,
                       child=asdict(child), inherited_score=score, passed=self.mc(score))
            if self.mc(score):
                distances = sorted(np.linalg.norm(child.vector - t.vector) for t in self.archive)
                novelty = float(np.mean(distances[:5]))
                candidates.append((novelty, child, parent))
        candidates.sort(key=lambda x: x[0], reverse=True)
        for novelty, child, parent in candidates:
            target = self.new_pair(child, parent.theta, self.iteration)
            donors = self.active if self.settings.transfers else [parent]
            best_score, best_theta = float("-inf"), None
            for donor in donors:
                direct = float(self.evaluate(child, donor.theta)[0])
                if direct > best_score:
                    best_score, best_theta = direct, donor.theta.copy()
                if self.settings.transfers:
                    proposal = self.proposal(target, donor.theta, use_adam=False)
                    score = float(self.evaluate(child, proposal)[0])
                    self.event("child_transfer_attempt", iteration=self.iteration, target=child.name,
                               donor=donor.terrain.name, direct_score=direct, proposal_score=score)
                    if score > best_score:
                        best_score, best_theta = score, proposal
            if not self.mc(best_score):
                continue
            target.theta, target.score = best_theta, best_score
            self.update_best(target)
            self.active.append(target)
            self.archive.append(child)
            self.event("admitted", iteration=self.iteration, terrain=asdict(child),
                       parent=parent.terrain.name, novelty=novelty, score=best_score)
            while len(self.active) > self.settings.max_active:
                retired = self.active.pop(0)
                self.event("retired", iteration=self.iteration, terrain=retired.terrain.name)
            break  # one admission per mutation event

    def step(self):
        i = self.iteration
        if i > 0 and i % self.settings.mutation_interval == 0:
            self.reproduce()
        for pair in self.active:
            self.optimize(pair)
        if self.settings.transfers and len(self.active) > 1 and i % self.settings.transfer_interval == 0:
            self.transfer()
        self.iteration += 1
        return dict(iteration=i, evaluations=self.evaluations, active=len(self.active),
                    archive=len(self.archive), scores={p.terrain.name: p.score for p in self.active})

    def save(self, path):
        pairs = []
        for p in self.active:
            pairs.append(dict(terrain=asdict(p.terrain), theta=p.theta.tolist(), born=p.born, sigma=p.sigma,
                              score=p.score if np.isfinite(p.score) else None,
                              best_score=p.best_score if np.isfinite(p.best_score) else None,
                              best_theta=None if p.best_theta is None else p.best_theta.tolist(),
                              adam={k: getattr(p.optimizer, k).tolist() if isinstance(getattr(p.optimizer, k), np.ndarray)
                                    else getattr(p.optimizer, k) for k in ("t", "m", "v", "stepsize")}))
        data = dict(format=1, implementation="POET Isaac Lab port", settings=asdict(self.settings),
                    parameter_count=self.parameter_count, iteration=self.iteration, evaluations=self.evaluations,
                    rng=self.rng.bit_generator.state, archive=[asdict(t) for t in self.archive], pairs=pairs)
        path = Path(path)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, allow_nan=False))
        tmp.replace(path)

    @classmethod
    def load(cls, path, evaluator, event=None):
        data = json.loads(Path(path).read_text())
        if data["format"] != 1:
            raise ValueError("Unsupported checkpoint version")
        obj = cls(Settings(**data["settings"]), evaluator, event, data["parameter_count"])
        obj.iteration, obj.evaluations = data["iteration"], data["evaluations"]
        obj.rng.bit_generator.state = data["rng"]
        obj.archive = [Terrain(**t) for t in data["archive"]]
        obj.active = []
        for item in data["pairs"]:
            p = obj.new_pair(Terrain(**item["terrain"]), np.array(item["theta"]), item["born"])
            p.score = float("-inf") if item["score"] is None else item["score"]
            p.best_score = float("-inf") if item["best_score"] is None else item["best_score"]
            p.sigma = item["sigma"]
            p.best_theta = None if item["best_theta"] is None else np.array(item["best_theta"])
            for k, v in item["adam"].items():
                setattr(p.optimizer, k, np.array(v) if k in ("m", "v") else v)
            obj.active.append(p)
        return obj
