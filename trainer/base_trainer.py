"""Shared training loop for DCL trainers.

Game-specific subclasses construct the env + agents and override
``_log_policy_diagnostics``; everything else lives here.
"""
import numpy as np
import torch
import wandb

from utility.buffer_class import ReplayBuffer

ACTION_LABEL = ["C", "D"]


class BaseDCLTrainer:
    def __init__(
        self,
        *,
        env,
        agents,
        N_episodes,
        buffer_length,
        max_steps,
        batch_size,
        num_iter_per_batch,
        gamma,
        entropy_coeff,
        entropy_coeff_decay,
        mega_step=1,
        explore=True,
    ):
        self.env = env
        self.agents = agents
        self.N_agents = len(agents)
        self.discount_factor = gamma
        self.mega_step = mega_step
        self.N_episodes = N_episodes
        self.batch_size = batch_size
        self.max_steps = max_steps
        self.num_iter_per_batch = num_iter_per_batch
        self.entropy_coeff = entropy_coeff
        self.entropy_coeff_decay = entropy_coeff_decay
        self.explore = explore
        self.dtype = torch.float32
        self.episode_returns = []

        proposal_dim = self.env.NUM_ACTIONS * mega_step
        self.replaybuffer = ReplayBuffer(
            max_length=buffer_length,
            gamma=gamma,
            state_dim=self.env.state_dim,
            proposal_dim=proposal_dim,
            commitment_dim=2,
            action_dim=proposal_dim,
            num_agents=self.N_agents,
        )

        for i in range(self.N_agents):
            self.agents[i].build_connection(self.agents[1 - i])

    # -------------------------------------------------------------- conversions
    def onehot_to_int(self, onehot):
        return torch.argmax(onehot[0]).item()

    def int_to_onehot(self, integer):
        onehot = torch.zeros(1, self.env.NUM_ACTIONS, dtype=self.dtype)
        onehot[0][integer] = 1
        return onehot

    def mega_one_hot_to_ints(self, mega_one_hot_vector):
        """One-hot over ``NUM_ACTIONS ** mega_step`` -> integer action sequence."""
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

    # ------------------------------------------------------------------ rollout
    def _decode_actions(self, m, a, is_mutual_commitment):
        """Pick proposed-or-unconstrained one-hots and return env-ready int actions."""
        chosen = m if is_mutual_commitment else a
        if self.mega_step == 1:
            return [self.onehot_to_int(x) for x in chosen], None
        # mega_step > 1: each one-hot encodes a sequence of mega_step ints.
        return None, np.array([self.mega_one_hot_to_ints(x) for x in chosen])

    def run_an_episode(self):
        state = torch.tensor(self.env.reset(), dtype=self.dtype)
        done = False
        accumulated_rewards = np.array([0.0, 0.0])

        while not done:
            m = [agent.get_proposal(state, explore=self.explore).detach() for agent in self.agents]
            c = [agent.get_commitment(state, m[i], m[1 - i], explore=self.explore).detach()
                 for i, agent in enumerate(self.agents)]
            a = [agent.get_unconstrained_action(state, explore=self.explore).detach() for agent in self.agents]
            is_mutual_commitment = self.agents[0].is_commit(c[0]) * self.agents[1].is_commit(c[1])

            actions_step, actions_mega = self._decode_actions(m, a, is_mutual_commitment)

            if self.mega_step == 1:
                next_state, step_rewards, done = self.env.step(actions_step)
            else:
                step_rewards = np.array([0.0, 0.0])
                for k_step in range(self.mega_step):
                    next_state, rewards, done = self.env.step(actions_mega[:, k_step])
                    step_rewards = step_rewards * self.discount_factor + rewards

            accumulated_rewards = accumulated_rewards * self.discount_factor + step_rewards

            self.replaybuffer.push(
                state, m[0], m[1], c[0], c[1], is_mutual_commitment, a[0], a[1],
                torch.tensor(step_rewards[0], dtype=self.dtype),
                torch.tensor(step_rewards[1], dtype=self.dtype),
                torch.tensor(next_state, dtype=self.dtype),
                torch.tensor(done, dtype=self.dtype),
            )
            state = torch.tensor(next_state, dtype=self.dtype)

        self.replaybuffer.finish_an_episode()
        self.episode_returns.append(accumulated_rewards)

    # ---------------------------------------------------------------- training
    def _ready_to_update(self, epi_idx):
        return (epi_idx + 1) * self.max_steps // self.mega_step % self.batch_size == 0

    def _update_agents(self, batch):
        states, proposals, commitments, _is_mutual, actions, _rewards, _next, _dones, returns = batch
        for i, agent in enumerate(self.agents):
            for _ in range(self.num_iter_per_batch):
                agent.update_critic(states, proposals[i], proposals[1 - i],
                                    actions[i], actions[1 - i],
                                    _is_mutual, returns[i])
                agent.update_unconstrained_policy(states, commitments[i], commitments[1 - i],
                                                  actions[i], actions[1 - i], self.entropy_coeff)
                agent.update_commitment_policy(states, proposals[i], proposals[1 - i],
                                                commitments[i], commitments[1 - i],
                                                actions[i], actions[1 - i], self.entropy_coeff)
                agent.update_proposal_policy(states, self.entropy_coeff)
            self.entropy_coeff = float(np.maximum(0.1, self.entropy_coeff - self.entropy_coeff_decay))
            self._log_policy_diagnostics(agent, i, states)

    def _log_episode_returns(self):
        mean_returns = np.mean(self.episode_returns, axis=0)
        wandb.log({
            "social welfare": mean_returns.sum(),
            "average return of agent 0": mean_returns[0],
            "average return of agent 1": mean_returns[1],
        })
        self.episode_returns = []

    def _log_policy_diagnostics(self, agent, agent_idx, states):
        """Game-specific wandb logging. Subclasses override."""

    def train(self):
        for epi_idx in range(self.N_episodes):
            self.run_an_episode()
            if self._ready_to_update(epi_idx):
                batch = self.replaybuffer.sample(self.batch_size)
                self._log_episode_returns()
                self._update_agents(batch)
