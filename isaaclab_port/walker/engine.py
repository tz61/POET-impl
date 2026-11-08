"""Run many policies on many courses across the GPU's worlds.

Finished worlds immediately take the next queued rollout, so a long rollout never
holds up the rest of a batch. Several POET runs may share one engine: each submits
from its own thread, and the thread owning the simulator calls pump().
"""
from collections import Counter, deque
import threading

import numpy as np
import torch

from isaaclab_port.runtime import policy_actions

from . import terrain


class Batch:
    def __init__(self, count):
        self.returns = np.zeros(count)
        self.lengths = np.zeros(count, dtype=np.int64)
        self.finished = np.zeros(count, dtype=bool)
        self.remaining = count
        self.done = threading.Event()
        if count == 0:
            self.done.set()


class RolloutEngine:
    def __init__(self, env, parameter_count):
        self.env = env
        self.device = env.device
        self.thetas = torch.zeros(env.num_envs, parameter_count, device=self.device)
        self.owner = [None] * env.num_envs
        self.busy = torch.zeros(env.num_envs, dtype=torch.bool, device=self.device)
        self.scores = torch.zeros(env.num_envs, dtype=torch.float64, device=self.device)
        self.steps = torch.zeros(env.num_envs, dtype=torch.int64, device=self.device)
        self.queue = deque()
        self.queued_slots = Counter()
        self.lock = threading.Lock()
        self.simulated_world_steps = 0
        self.first_episode_steps = 0
        env.assigner = self._assign
        self.obs = env._get_observations()["policy"]

    def submit(self, tasks):
        """tasks: (config, seed, theta) triples. Returns a Batch to wait on."""
        batch = Batch(len(tasks))
        bank = self.env.bank
        # Compiling takes milliseconds per course, so keep it out of the lock.
        compiled = {}
        for config, seed, _ in tasks:
            key = (config.key, int(seed))
            if key not in bank and key not in compiled:
                compiled[key] = terrain.compile_course(terrain.generate_course(config, seed))
        thetas = [torch.as_tensor(theta, dtype=torch.float32) for _, _, theta in tasks]
        with self.lock:
            # A queued course must survive until a world picks it up.
            in_use = set(self.env.world_terrain.tolist()) | set(self.queued_slots)
            for index, (config, seed, _) in enumerate(tasks):
                slot = bank.add(config, seed, in_use, compiled.get((config.key, int(seed))))
                in_use.add(slot)
                self.queued_slots[slot] += 1
                self.queue.append((batch, index, slot, thetas[index]))
        return batch

    def _assign(self, env_ids):
        """Called by the env inside a reset: give each world work, or leave it idle."""
        worlds = env_ids.tolist()
        with self.lock:
            taken = [self.queue.popleft() for _ in range(min(len(worlds), len(self.queue)))]
            for item in taken:
                self.queued_slots[item[2]] -= 1
                if self.queued_slots[item[2]] == 0:
                    del self.queued_slots[item[2]]
        slots = torch.zeros(len(worlds), dtype=torch.int64, device=self.device)
        self.thetas[env_ids] = 0.0
        self.busy[env_ids] = False
        for position, world in enumerate(worlds):
            self.owner[world] = taken[position][:2] if position < len(taken) else None
        if taken:
            working = env_ids[:len(taken)]
            self.thetas[working] = torch.stack([item[3] for item in taken]).to(self.device)
            self.busy[working] = True
            slots[:len(taken)] = torch.tensor([item[2] for item in taken], device=self.device)
        return slots

    def pump(self):
        """Advance the simulation one control step. Returns False when nothing is running."""
        idle = (~self.busy).nonzero().flatten()
        if len(idle) and self.queue:
            self.env.reset_worlds(idle[:len(self.queue)])
            self.obs = self.env._get_observations()["policy"]
        if not self.busy.any():
            return False
        was_busy = self.busy.clone()
        # A reset inside step() reassigns finished worlds, so note who owns them now.
        self.previous_owner = list(self.owner)
        actions = policy_actions(self.obs, self.thetas)
        with torch.no_grad():
            obs, reward, terminated, truncated, _ = self.env.step(actions)
        self.obs = obs["policy"]
        if self.env._numerical_failure[was_busy].any():
            raise FloatingPointError("Non-finite simulator state in an evaluated episode")
        self.scores += torch.where(was_busy, reward.to(torch.float64), 0.0)
        self.steps += was_busy
        self.simulated_world_steps += self.env.num_envs
        ended = ((terminated | truncated) & was_busy).nonzero().flatten()
        if len(ended):
            self._harvest(ended)
        return True

    def _harvest(self, worlds):
        scores = self.scores[worlds].cpu().numpy()
        steps = self.steps[worlds].cpu().numpy()
        finished = self.env.finished[worlds].cpu().numpy()
        self.first_episode_steps += int(steps.sum())
        for world, score, length, finish in zip(worlds.tolist(), scores, steps, finished):
            batch, index = self.previous_owner[world]
            batch.returns[index], batch.lengths[index], batch.finished[index] = score, length, finish
            batch.remaining -= 1
            if batch.remaining == 0:
                batch.done.set()
        self.scores[worlds] = 0.0
        self.steps[worlds] = 0

    def run(self, tasks):
        """Blocking evaluation for a single caller that also owns the simulator."""
        batch = self.submit(tasks)
        while not batch.done.is_set():
            if not self.pump():
                raise RuntimeError("Engine stopped with work outstanding")
        return batch

    def serve(self, should_stop):
        """Owner-thread loop for several submitting threads."""
        while not should_stop():
            if not self.pump():
                threading.Event().wait(0.001)
