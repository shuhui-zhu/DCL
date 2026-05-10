"""Shared helpers for decentralized DCL trainers."""
import itertools

import numpy as np
import torch
import wandb

ACTION_LABEL = ["C", "D"]


class DecentralizedTrainerBase:
    """Mixin-style base: subclasses set ``env``, ``mega_step`` (repeated games), ``dtype``."""

    dtype = torch.float32

    def onehot_to_int(self, onehot):
        return torch.argmax(onehot[0]).item()

    def int_to_onehot(self, integer):
        onehot = torch.zeros(1, self.env.num_actions, dtype=self.dtype)
        onehot[0][integer] = 1
        return onehot

    def mega_one_hot_to_ints(self, mega_one_hot_vector):
        one_hot = mega_one_hot_vector[0].detach().numpy()
        if not np.isclose(one_hot.sum(), 1.0):
            raise ValueError("Input vector is not a valid one-hot vector")
        index = int(np.argmax(one_hot))
        shape = (self.env.NUM_ACTIONS,) * self.mega_step
        return np.array(np.unravel_index(index, shape))

    def ints_to_mega_one_hot(self, ints):
        shape = (self.env.NUM_ACTIONS,) * self.mega_step
        index = int(np.ravel_multi_index(tuple(ints), shape))
        size = int(np.prod(shape))
        onehot = torch.zeros(1, size, dtype=self.dtype)
        onehot[0][index] = 1
        return onehot

    def log_episode_returns(self, episode_returns):
        mean_returns = np.mean(episode_returns, axis=0)
        wandb.log({
            "social welfare": mean_returns.sum(),
            "average return of agent 0": mean_returns[0],
            "average return of agent 1": mean_returns[1],
        })

    def log_repeated_diagnostics_mega1(self, agent, agent_idx, states):
        first = states[0]
        wandb.log({
            f"policy prob of cooperation for agent {agent_idx}":
                agent.unconstrained_actor(first).squeeze()[0].detach(),
            f"proposal prob of cooperation for agent {agent_idx}":
                agent.proposing_actor(first).squeeze()[0].detach(),
        })
        for ii, jj in itertools.product(range(self.env.NUM_ACTIONS), repeat=2):
            ps = self.int_to_onehot(ii)
            pc = self.int_to_onehot(jj)
            joint = torch.cat((first.unsqueeze(0), ps, pc), dim=1)
            wandb.log({
                f"commitment prob for agent {agent_idx} given [{ACTION_LABEL[ii]},{ACTION_LABEL[jj]}]":
                    agent.commit_actor(joint).detach().squeeze()[1].numpy(),
                f"critic value for agent{agent_idx} of [{ii},{jj}]:":
                    agent.critic(joint).detach(),
            })

    def log_repeated_diagnostics_mega2(self, agent, agent_idx, states):
        first = states[0]
        labels_l = ["(C,C)", "(C,D)", "(D,C)", "(D,D)"]
        for k, lbl in enumerate(labels_l):
            wandb.log({
                f"policy prob of {lbl} for agent {agent_idx}":
                    agent.unconstrained_actor(first).squeeze()[k].detach(),
                f"proposal prob of {lbl} for agent {agent_idx}":
                    agent.proposing_actor(first).squeeze()[k].detach(),
            })
        rng = range(self.env.NUM_ACTIONS)
        for i0s, i1s, i0c, i1c in itertools.product(rng, rng, rng, rng):
            ps = self.ints_to_mega_one_hot(np.array([i0s, i1s]))
            pc = self.ints_to_mega_one_hot(np.array([i0c, i1c]))
            joint = torch.cat((first.unsqueeze(0), ps, pc), dim=1)
            key = (
                f"commitment prob for agent {agent_idx} given "
                f"[{ACTION_LABEL[i0s]}{ACTION_LABEL[i1s]},"
                f"{ACTION_LABEL[i0c]}{ACTION_LABEL[i1c]}]"
            )
            wandb.log({key: agent.commit_actor(joint).detach().squeeze()[1].numpy()})
