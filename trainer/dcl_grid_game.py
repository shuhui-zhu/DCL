import itertools

import torch
import wandb

from trainer.base_trainer import ACTION_LABEL, BaseDCLTrainer


class Grid_Game_Trainer(BaseDCLTrainer):
    def __init__(self, env, agent_class, N_agents, N_episodes, with_constraints,
                 buffer_length, max_steps, batch_size, temperature, num_iter_per_batch,
                 hidden_dim, lr_critic, lr_actor, gamma, is_entropy, entropy_coeff,
                 entropy_coeff_decay, temperature_decay, grid_size, explore=True):
        env_inst = env(max_steps=max_steps, grid_size=grid_size)
        agents = [
            agent_class(
                with_constraints=with_constraints, gamma=gamma, grid_size=grid_size,
                num_agents=N_agents, action_dim=env_inst.NUM_ACTIONS,
                temperature=temperature, hidden_dim=hidden_dim,
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
            mega_step=1, explore=explore,
        )

    # ---- diagnostics ---------------------------------------------------------
    def _log_policy_diagnostics(self, agent, agent_idx, states):
        first_state = states[0]
        wandb.log({
            f"policy prob of cooperation for agent {agent_idx}":
                agent.unconstrained_actor(first_state).squeeze()[agent_idx].detach(),
            f"proposal prob of cooperation for agent {agent_idx}":
                agent.proposing_actor(first_state).squeeze()[agent_idx].detach(),
        })
        for ii, jj in itertools.product(range(self.env.NUM_ACTIONS), repeat=2):
            ps = self.int_to_onehot(ii)
            pc = self.int_to_onehot(jj)
            joint = torch.cat((first_state.unsqueeze(0), ps, pc), dim=1)
            # The agent-1 view is mirrored: its "self" index is the co-player's column for agent 0.
            if agent_idx == 0:
                key = f"commitment prob for agent {agent_idx} given [{ACTION_LABEL[ii]},{ACTION_LABEL[1 - jj]}]"
            else:
                key = f"commitment prob for agent {agent_idx} given [{ACTION_LABEL[1 - ii]},{ACTION_LABEL[jj]}]"
            wandb.log({key: agent.commit_actor(joint).detach().squeeze()[1].numpy()})
