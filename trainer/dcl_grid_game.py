"""Decentralized DCL trainer for the grid social-dilemma game."""
import numpy as np
import torch
import wandb

from utility.buffer_class import ReplayBuffer

from trainer.base_trainer import ACTION_LABEL, DecentralizedTrainerBase


class Grid_Game_Trainer(DecentralizedTrainerBase):
    def __init__(
        self, env, agent_class, N_agents, N_episodes, with_constraints, buffer_length, max_steps,
        batch_size, temperature, num_iter_per_batch, hidden_dim, lr_critic, lr_actor, gamma,
        is_entropy, entropy_coeff, entropy_coeff_decay, temperature_decay, grid_size,
        epsilon, perturb, epsilon_decay, explore=True,
    ):
        self.env = env(max_steps=max_steps, grid_size=grid_size)
        self.mega_step = 1  # for base-class encoding helpers if used
        self.N_agents = N_agents
        self.discount_factor = gamma
        self.agents = [
            agent_class(
                perturb=perturb, with_constraints=with_constraints, gamma=gamma,
                grid_size=grid_size, num_agents=N_agents, action_dim=self.env.NUM_ACTIONS,
                temperature=temperature, hidden_dim=hidden_dim, lr_critic=lr_critic,
                lr_actor=lr_actor, is_entropy=is_entropy, temperature_decay=temperature_decay,
            )
            for _ in range(N_agents)
        ]
        self.replaybuffer = ReplayBuffer(
            max_length=buffer_length, gamma=gamma, state_dim=grid_size * N_agents,
            proposal_dim=self.env.NUM_ACTIONS, commitment_dim=2,
            action_dim=self.env.NUM_ACTIONS, num_agents=N_agents,
        )
        self.N_episodes = N_episodes
        self.batch_size = batch_size
        self.max_steps = max_steps
        self.explore = explore
        self.num_iter_per_batch = num_iter_per_batch
        self.dtype = torch.float32
        self.entropy_coeff = entropy_coeff
        self.entropy_coeff_decay = entropy_coeff_decay
        self.returns = []
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay

    def run_an_episode(self):
        state = torch.tensor(self.env.reset(), dtype=self.dtype)
        done = False
        accumulated_rewards = np.array([0.0, 0.0])
        while not done:
            m = [agent.get_proposal(state, explore=self.explore).detach() for agent in self.agents]
            c = [agent.get_commitment(state, m[i], m[1 - i], explore=self.explore).detach()
                 for i, agent in enumerate(self.agents)]
            a = [agent.get_unconstrained_action(state, explore=self.explore).detach()
                 for agent in self.agents]

            is_mutual_commitment = self.agents[0].is_commit(c[0]) * self.agents[1].is_commit(c[1])
            if is_mutual_commitment:
                actions = [self.onehot_to_int(m_i) for m_i in m]
            else:
                actions = [self.onehot_to_int(a_i) for a_i in a]

            next_state, rewards, done = self.env.step(actions)
            accumulated_rewards = accumulated_rewards * self.discount_factor + rewards

            self.replaybuffer.push(
                state, m[0], m[1], c[0], c[1], is_mutual_commitment, a[0], a[1],
                torch.tensor(rewards[0], dtype=self.dtype),
                torch.tensor(rewards[1], dtype=self.dtype),
                torch.tensor(next_state, dtype=self.dtype),
                torch.tensor(done, dtype=self.dtype),
            )
            state = torch.tensor(next_state, dtype=self.dtype)

        self.replaybuffer.finish_an_episode()
        self.returns.append(accumulated_rewards)

    def train(self):
        for epi_idx in range(self.N_episodes):
            self.run_an_episode()
            if (epi_idx + 1) * self.max_steps % self.batch_size == 0:
                states, proposals, commitments, is_mutual_commitments, actions, rewards, next_states, dones, returns = (
                    self.replaybuffer.sample(self.batch_size)
                )
                self.log_episode_returns(self.returns)
                self.returns = []
                for i, agent in enumerate(self.agents):
                    for _ in range(self.num_iter_per_batch):
                        agent.update_critic(
                            states, proposals[i], proposals[1 - i], actions[i], actions[1 - i],
                            is_mutual_commitments, returns[i], returns[1 - i],
                        )
                        agent.update_unconstrained_policy(
                            states, commitments[i], commitments[1 - i], actions[i], actions[1 - i],
                            self.entropy_coeff,
                        )
                        agent.update_commitment_policy(
                            states, proposals[i], proposals[1 - i], commitments[i], commitments[1 - i],
                            actions[i], actions[1 - i], self.entropy_coeff,
                        )
                        agent.update_proposal_policy(states, self.entropy_coeff)

                    wandb.log({"entropy_coeff": self.entropy_coeff})
                    self.entropy_coeff = np.maximum(0.0, self.entropy_coeff - self.entropy_coeff_decay)
                    wandb.log({"epsilon": self.epsilon})
                    if (epi_idx + 1) * self.max_steps / self.batch_size % 100 == 0:
                        self.epsilon = np.maximum(0.0, self.epsilon - self.epsilon_decay)

                    wandb.log({
                        f"policy prob of cooperation for agent {i}":
                            agent.unconstrained_actor(states[0]).squeeze()[i].detach(),
                    })
                    for ii in range(2):
                        for jj in range(2):
                            proposal_self_onehot = self.int_to_onehot(ii)
                            proposal_coplayer_onehot = self.int_to_onehot(jj)
                            joint = torch.cat((states[0].unsqueeze(0), proposal_self_onehot, proposal_coplayer_onehot), dim=1)
                            if i == 0:
                                key = f"commitment prob for agent {i} given [{ACTION_LABEL[ii]},{ACTION_LABEL[1 - jj]}]"
                            else:
                                key = f"commitment prob for agent {i} given [{ACTION_LABEL[1 - ii]},{ACTION_LABEL[jj]}]"
                            wandb.log({key: agent.commit_actor(joint).detach().squeeze()[1].numpy()})
                    wandb.log({
                        f"proposal prob of cooperation for agent {i}":
                            agent.proposing_actor(states[0]).squeeze()[i].detach(),
                    })
