import numpy as np
import torch
import wandb

from utility.buffer_class import ReplayBuffer

from trainer.base_trainer import DecentralizedTrainerBase


class Repeated_Game_Trainer(DecentralizedTrainerBase):
    def __init__(
        self, env, agent_class, N_agents, N_episodes, with_constraints, buffer_length, max_steps,
        batch_size, temperature, num_iter_per_batch, hidden_dim, lr_critic, lr_actor, gamma,
        is_entropy, entropy_coeff, entropy_coeff_decay, temperature_decay, mega_step,
        epsilon, perturb, epsilon_decay, explore=True,
    ):
        self.env = env(max_steps=max_steps)
        self.N_agents = N_agents
        self.discount_factor = gamma
        self.mega_step = mega_step
        self.agents = [
            agent_class(
                perturb=perturb, with_constraints=with_constraints, gamma=gamma,
                num_agents=N_agents, state_dim=self.env.state_dim, action_dim=self.env.NUM_ACTIONS,
                mega_step=mega_step, temperature=temperature, hidden_dim=hidden_dim,
                lr_critic=lr_critic, lr_actor=lr_actor, is_entropy=is_entropy,
                temperature_decay=temperature_decay,
            )
            for _ in range(N_agents)
        ]
        self.replaybuffer = ReplayBuffer(
            max_length=buffer_length, gamma=gamma, state_dim=self.env.state_dim,
            proposal_dim=self.env.NUM_ACTIONS * mega_step, commitment_dim=2,
            action_dim=self.env.NUM_ACTIONS * mega_step, num_agents=N_agents,
        )
        self.N_episodes = N_episodes
        self.batch_size = batch_size
        self.max_steps = max_steps
        self.explore = explore
        self.num_iter_per_batch = num_iter_per_batch
        self.dtype = torch.float32
        self.entropy_coeff = entropy_coeff
        self.entropy_coeff_decay = entropy_coeff_decay
        self.mega_returns = []
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay

    def run_an_episode(self):
        state = torch.tensor(self.env.reset(), dtype=self.dtype)
        done = False
        accumulated_rewards = np.array([0.0, 0.0])
        while not done:
            if self.mega_step == 2:
                mega_rewards = np.array([0.0, 0.0])
            m = [agent.get_proposal(state, explore=self.explore, epsilon=self.epsilon).detach()
                 for agent in self.agents]
            c = [agent.get_commitment(state, m[i], m[1 - i], explore=self.explore, epsilon=self.epsilon).detach()
                 for i, agent in enumerate(self.agents)]
            a = [agent.get_unconstrained_action(state, explore=self.explore, epsilon=self.epsilon).detach()
                 for agent in self.agents]
            is_mutual_commitment = self.agents[0].is_commit(c[0]) * self.agents[1].is_commit(c[1])

            if is_mutual_commitment:
                if self.mega_step == 2:
                    actions_for_mega_step = np.array([self.mega_one_hot_to_ints(m_i) for m_i in m])
                else:
                    actions = [self.onehot_to_int(m_i) for m_i in m]
            else:
                if self.mega_step == 2:
                    actions_for_mega_step = np.array([self.mega_one_hot_to_ints(a_i) for a_i in a])
                else:
                    actions = [self.onehot_to_int(a_i) for a_i in a]

            if self.mega_step == 2:
                for k_step in range(self.mega_step):
                    next_state, rewards, done = self.env.step(actions_for_mega_step[:, k_step])
                    mega_rewards = mega_rewards * self.discount_factor + rewards
            else:
                next_state, mega_rewards, done = self.env.step(actions)

            accumulated_rewards = accumulated_rewards * self.discount_factor + mega_rewards

            self.replaybuffer.push(
                state, m[0], m[1], c[0], c[1], is_mutual_commitment, a[0], a[1],
                torch.tensor(mega_rewards[0], dtype=self.dtype),
                torch.tensor(mega_rewards[1], dtype=self.dtype),
                torch.tensor(next_state, dtype=self.dtype),
                torch.tensor(done, dtype=self.dtype),
            )
            state = torch.tensor(next_state, dtype=self.dtype)

        self.replaybuffer.finish_an_episode()
        self.mega_returns.append(accumulated_rewards)

    def train(self):
        for epi_idx in range(self.N_episodes):
            self.run_an_episode()
            if (epi_idx + 1) * self.max_steps / self.mega_step % self.batch_size == 0:
                states, proposals, commitments, is_mutual_commitments, actions, rewards, next_states, dones, returns = (
                    self.replaybuffer.sample(self.batch_size)
                )
                self.log_episode_returns(self.mega_returns)
                self.mega_returns = []
                for i, agent in enumerate(self.agents):
                    for _ in range(self.num_iter_per_batch):
                        agent.update_critic(
                            states, proposals[i], proposals[1 - i], actions[i], actions[1 - i],
                            is_mutual_commitments, returns[i],
                        )
                        agent.update_unconstrained_policy(
                            states, commitments[i], commitments[1 - i], actions[i], actions[1 - i],
                            self.entropy_coeff,
                        )
                        agent.update_commitment_policy(
                            states, proposals[i], proposals[1 - i], commitments[i], commitments[1 - i],
                            actions[i], actions[1 - i], self.entropy_coeff, self.epsilon,
                        )
                        agent.update_proposal_policy(states, self.entropy_coeff, self.epsilon)

                        agent.update_coplayer_critic(
                            states, proposals[i], proposals[1 - i], actions[i], actions[1 - i],
                            is_mutual_commitments, returns[1 - i],
                        )
                        agent.update_coplayer_unconstrained_policy(
                            states, commitments[i], commitments[1 - i], actions[i], actions[1 - i],
                            self.entropy_coeff,
                        )
                        agent.update_coplayer_commitment_policy(
                            states, proposals[i], proposals[1 - i], commitments[i], commitments[1 - i],
                            actions[i], actions[1 - i], self.entropy_coeff, self.epsilon,
                        )
                        agent.update_coplayer_proposal_policy(states, self.entropy_coeff, self.epsilon)

                    wandb.log({"entropy_coeff": self.entropy_coeff})
                    self.entropy_coeff = np.maximum(0.0, self.entropy_coeff - self.entropy_coeff_decay)
                    wandb.log({"epsilon": self.epsilon})
                    if (epi_idx + 1) * self.max_steps / self.mega_step / self.batch_size % 100 == 0:
                        self.epsilon = np.maximum(0.0, self.epsilon - self.epsilon_decay)

                    if self.mega_step == 1:
                        self.log_repeated_diagnostics_mega1(agent, i, states)
                    else:
                        self.log_repeated_diagnostics_mega2(agent, i, states)
