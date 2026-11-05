"""Batched policy evaluation in an actual Isaac Lab simulator."""
from dataclasses import asdict

import numpy as np
import torch


def policy_actions(observation: torch.Tensor, parameters: torch.Tensor) -> torch.Tensor:
    """Evaluate one independent 24-40-40-4 tanh network per parallel world."""
    h = observation
    offset = 0
    for inputs, outputs in ((24, 40), (40, 40), (40, 4)):
        count = inputs * outputs
        weights = parameters[:, offset:offset + count].reshape(-1, inputs, outputs)
        offset += count
        bias = parameters[:, offset:offset + outputs]
        offset += outputs
        h = torch.tanh(torch.bmm(h.unsqueeze(1), weights).squeeze(1) + bias)
    if offset != parameters.shape[1]:
        raise ValueError("Incorrect policy parameter count")
    return h


class Evaluator:
    """Collect first-episode returns; ignore auto-reset episodes in vector worlds."""
    def __init__(self, num_envs, cfg_factory, *, render=False):
        self.num_envs = num_envs
        self.cfg_factory = cfg_factory
        self.render = render
        self.env = None
        self.terrain = None
        self.requested_episodes = 0
        self.first_episode_steps = 0
        self.simulated_world_steps = 0
        self.last_episode_lengths = []
        self.last_episode_successes = []

    def close(self):
        if self.env is not None:
            self.env.close()
            self.env = None

    def __call__(self, terrain, matrix):
        from isaaclab_port.env import PoetBipedEnv
        if self.env is None or self.terrain != terrain:
            self.close()
            self.env = PoetBipedEnv(self.cfg_factory(terrain))
            self.terrain = terrain
        env = self.env
        results = []
        all_lengths = []
        all_successes = []
        for start in range(0, len(matrix), self.num_envs):
            chunk = matrix[start:start + self.num_envs]
            n = len(chunk)
            padded = np.concatenate((chunk, np.repeat(chunk[-1:], self.num_envs - n, axis=0)))
            theta = torch.as_tensor(padded, dtype=torch.float32, device=env.device)
            obs, _ = env.reset(seed=terrain.seed)
            active = torch.ones(n, dtype=torch.bool, device=env.device)
            scores = torch.zeros(n, dtype=torch.float64, device=env.device)
            lengths = torch.zeros(n, dtype=torch.int64, device=env.device)
            successes = torch.zeros(n, dtype=torch.bool, device=env.device)
            # Environment state persists between calls and reset() updates it
            # outside this block. no_grad avoids immutable inference tensors.
            with torch.no_grad():
                for _ in range(env.max_episode_length):
                    actions = policy_actions(obs["policy"], theta)
                    obs, reward, terminated, truncated, info = env.step(actions)
                    numerical_failure = info.get("numerical_failure")
                    if numerical_failure is not None and (numerical_failure[:n] & active).any():
                        raise FloatingPointError("Non-finite simulator state in an evaluated episode")
                    scores += torch.where(active, reward[:n].to(torch.float64), 0.0)
                    lengths += active
                    successes |= info.get("terminal_success", torch.zeros_like(terminated))[:n] & active
                    active &= ~(terminated[:n] | truncated[:n])
                    self.simulated_world_steps += self.num_envs
                    if self.render:
                        env.sim.render()
                    if not active.any():
                        break
            results.extend(scores.cpu().tolist())
            all_lengths.extend(lengths.cpu().tolist())
            all_successes.extend(successes.cpu().tolist())
            self.requested_episodes += n
            self.first_episode_steps += int(lengths.sum().item())
        self.last_episode_lengths = all_lengths
        self.last_episode_successes = all_successes
        return np.asarray(results)

    def metrics(self):
        return dict(requested_episodes=self.requested_episodes,
                    first_episode_steps=self.first_episode_steps,
                    simulated_world_steps=self.simulated_world_steps)
