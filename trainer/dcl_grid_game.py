import numpy as np
import torch
from utility.buffer_class import ReplayBuffer
import wandb

class Grid_Game_Trainer:
    def __init__(self, env, agent_class, N_agents, N_episodes, with_constraints, buffer_length, max_steps, batch_size,temperature,num_iter_per_batch,
                 hidden_dim, lr_critic, lr_actor, gamma,is_entropy, entropy_coeff, entropy_coeff_decay,temperature_decay,grid_size,
                 explore=True):
        self.env = env(max_steps=max_steps, grid_size=grid_size)
        self.N_agents = N_agents
        self.discount_factor = gamma
        self.agents = [agent_class(with_constraints=with_constraints, gamma = gamma, grid_size=grid_size, num_agents=N_agents,action_dim=self.env.NUM_ACTIONS, temperature=temperature,hidden_dim=hidden_dim, lr_critic=lr_critic, lr_actor=lr_actor, is_entropy=is_entropy, temperature_decay=temperature_decay) for _ in range(N_agents)] # initialize agents classes
        [self.agents[i].build_connection(self.agents[1-i]) for i in range(N_agents)] # build connection between agents
        self.replaybuffer = ReplayBuffer(max_length=buffer_length,gamma=gamma,state_dim=self.env.state_dim,proposal_dim=self.env.NUM_ACTIONS,commitment_dim=2,action_dim=self.env.NUM_ACTIONS,num_agents=N_agents)
        self.N_episodes = N_episodes
        self.batch_size = batch_size
        self.max_steps = max_steps
        self.explore = explore
        self.num_iter_per_batch = num_iter_per_batch
        self.dtype = torch.float32
        self.entropy_coeff = entropy_coeff
        self.entropy_coeff_decay = entropy_coeff_decay
        self.returns = []

    def onehot_to_int(self, onehot):
        """
        Convert one-hot action to integer action
        """
        int = torch.argmax(onehot[0]).item()
        return int
    
    def int_to_onehot(self, integer):
        """
        Convert integer proposal to one-hot proposal
        """
        onehot = torch.zeros(1,self.env.num_actions, dtype=torch.float32)
        onehot[0][integer] = 1
        return onehot

    def run_an_episode(self):
        """
        Each episode includes 1 step.
        """
        state = torch.tensor(self.env.reset(),dtype=self.dtype) # values from environment is np.array
        done = False
        accumulated_rewards = np.array([0,0])
        while not done:
            m = [agent.get_proposal(state, explore=self.explore).detach() for agent in self.agents] # one-hot value
            c = [agent.get_commitment(state, m[i],m[1-i], explore=self.explore).detach() for i, agent in enumerate(self.agents)] # one-hot value, [[0,1]] is commitment, [[1,0]] is no commitment
            a = [agent.get_unconstrained_action(state, explore=self.explore).detach() for agent in self.agents] # one-hot value
            is_mutual_commitment = self.agents[0].is_commit(c[0])*self.agents[1].is_commit(c[1]) # agent 0 and agent 1 both commit

            if is_mutual_commitment:
                actions = [self.onehot_to_int(m_i) for m_i in m]
            else:
                actions = [self.onehot_to_int(a_i) for a_i in a]
            next_state, rewards, done = self.env.step(actions)
            accumulated_rewards = accumulated_rewards * self.discount_factor + rewards
            
            self.replaybuffer.push(state, m[0], m[1], c[0], c[1], is_mutual_commitment, a[0], a[1],\
                            torch.tensor(rewards[0],dtype=self.dtype), torch.tensor(rewards[1],dtype=self.dtype), torch.tensor(next_state,dtype=self.dtype),
                            torch.tensor(done, dtype=self.dtype))
            state = torch.tensor(next_state,dtype=self.dtype)
        self.replaybuffer.finish_an_episode()
        self.returns.append(accumulated_rewards)

    def train(self):
        """
        Train the agents
        """
        action_labels = ["C","D"]
        for epi_idx in range(self.N_episodes):
            self.run_an_episode()
            if (epi_idx+1)*self.max_steps % self.batch_size == 0:
                states, proposals, commitments, is_mutual_commitments, actions, rewards, next_states, dones, returns = self.replaybuffer.sample(self.batch_size)
                mean_returns = np.mean(self.returns, axis=0)
                wandb.log({"social welfare": mean_returns.sum()})
                wandb.log({"average return of agent 0": mean_returns[0]})
                wandb.log({"average return of agent 1": mean_returns[1]})
                self.returns = []
                for i, agent in enumerate(self.agents):
                    for _ in range(self.num_iter_per_batch): 
                        agent.update_critic(states, proposals[i], proposals[1-i], actions[i], actions[1-i], is_mutual_commitments, returns[i]) # update critic first (according to MADDPG, SAC, etc.)
                        agent.update_unconstrained_policy(states, commitments[i], commitments[1-i], actions[i], actions[1-i], self.entropy_coeff)        
                        agent.update_commitment_policy(states, proposals[i], proposals[1-i], commitments[i], commitments[1-i], actions[i], actions[1-i], self.entropy_coeff)
                        agent.update_proposal_policy(states, self.entropy_coeff)
                    self.entropy_coeff = np.maximum(0.1, self.entropy_coeff - self.entropy_coeff_decay)
                    wandb.log({"policy prob of cooperation for agent {}".format(i): agent.unconstrained_actor(states[0]).squeeze()[i].detach()})
                    for ii in range(2):
                        for jj in range(2):
                            proposal_self_onehot = self.int_to_onehot(ii)
                            proposal_coplayer_onehot = self.int_to_onehot(jj)
                            if i==0:
                                wandb.log({"commitment prob for agent {} given [{},{}]".format(i,action_labels[ii],action_labels[1-jj]): agent.commit_actor(torch.cat((states[0].unsqueeze(dim=0), proposal_self_onehot, proposal_coplayer_onehot),dim=1)).detach().squeeze()[1].numpy()})
                            else:
                                wandb.log({"commitment prob for agent {} given [{},{}]".format(i,action_labels[1-ii],action_labels[jj]): agent.commit_actor(torch.cat((states[0].unsqueeze(dim=0), proposal_self_onehot, proposal_coplayer_onehot),dim=1)).detach().squeeze()[1].numpy()})
                    wandb.log({"proposal prob of cooperation for agent {}".format(i): agent.proposing_actor(states[0]).squeeze()[i].detach()})
                # print("finish an update")
        return