"""Trainer for repeated matrix games (IPD, IPC). Supports mega_step >= 1."""
import itertools

import numpy as np
import torch
import wandb

from trainer.base_trainer import ACTION_LABEL, BaseDCLTrainer


class Repeated_Game_Trainer(BaseDCLTrainer):
    def __init__(self, env, agent_class, N_agents, N_episodes, with_constraints,
                 buffer_length, max_steps, batch_size, temperature, num_iter_per_batch,
                 hidden_dim, lr_critic, lr_actor, gamma, is_entropy, entropy_coeff,
                 entropy_coeff_decay, temperature_decay, mega_step, explore=True):
        env_inst = env(max_steps=max_steps)
        agents = [
            agent_class(
                with_constraints=with_constraints, gamma=gamma, num_agents=N_agents,
                state_dim=env_inst.state_dim, action_dim=env_inst.NUM_ACTIONS,
                mega_step=mega_step, temperature=temperature, hidden_dim=hidden_dim,
                lr_critic=lr_critic, lr_actor=lr_actor, is_entropy=is_entropy,
                temperature_decay=temperature_decay,
            )
            for _ in range(N_agents)
        ]
        super().__init__(
            env=env_inst, agents=agents, N_episodes=N_episodes,
            buffer_length=buffer_length, max_steps=max_steps, batch_size=batch_size,
            num_iter_per_batch=num_iter_per_batch, gamma=gamma,
            entropy_coeff=entropy_coeff, entropy_coeff_decay=entropy_coeff_decay,
            mega_step=mega_step, explore=explore,
        )

    # ---- diagnostics ---------------------------------------------------------
    def _log_policy_diagnostics(self, agent, agent_idx, states):
        if self.mega_step == 1:
            self._log_step1_diagnostics(agent, agent_idx, states)
        else:
            self._log_mega_diagnostics(agent, agent_idx, states)

    def _log_step1_diagnostics(self, agent, agent_idx, states):
        first_state = states[0]
        wandb.log({
            f"policy prob of cooperation for agent {agent_idx}":
                agent.unconstrained_actor(first_state).squeeze()[0].detach(),
            f"proposal prob of cooperation for agent {agent_idx}":
                agent.proposing_actor(first_state).squeeze()[0].detach(),
        })
        for ii, jj in itertools.product(range(self.env.NUM_ACTIONS), repeat=2):
            ps = self.int_to_onehot(ii)
            pc = self.int_to_onehot(jj)
            joint = torch.cat((first_state.unsqueeze(0), ps, pc), dim=1)
            wandb.log({
                f"commitment prob for agent {agent_idx} given [{ACTION_LABEL[ii]},{ACTION_LABEL[jj]}]":
                    agent.commit_actor(joint).detach().squeeze()[1].numpy(),
                f"critic value for agent{agent_idx} of [{ii},{jj}]:":
                    agent.critic(joint).detach(),
            })

    def _log_mega_diagnostics(self, agent, agent_idx, states):
        first_state = states[0]
        unconstrained_probs = agent.unconstrained_actor(first_state).squeeze().detach()
        proposal_probs = agent.proposing_actor(first_state).squeeze().detach()
        labels_2step = ["(C,C)", "(C,D)", "(D,C)", "(D,D)"]
        for k, lbl in enumerate(labels_2step):
            wandb.log({
                f"policy prob of {lbl} for agent {agent_idx}": unconstrained_probs[k],
                f"proposal prob of {lbl} for agent {agent_idx}": proposal_probs[k],
            })

        rng = range(self.env.NUM_ACTIONS)
        for i0s, i1s, i0c, i1c in itertools.product(rng, rng, rng, rng):
            ps = self.ints_to_mega_one_hot(np.array([i0s, i1s]))
            pc = self.ints_to_mega_one_hot(np.array([i0c, i1c]))
            joint = torch.cat((first_state.unsqueeze(0), ps, pc), dim=1)
            key = (
                f"commitment prob for agent {agent_idx} given "
                f"[{ACTION_LABEL[i0s]}{ACTION_LABEL[i1s]},"
                f"{ACTION_LABEL[i0c]}{ACTION_LABEL[i1c]}]"
            )
            wandb.log({key: agent.commit_actor(joint).detach().squeeze()[1].numpy()})
