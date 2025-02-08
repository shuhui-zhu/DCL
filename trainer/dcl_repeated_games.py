import numpy as np
import torch
from utility.buffer_class import ReplayBuffer
import wandb

class Repeated_Game_Trainer(object):
    def __init__(self, env, agent_class, N_agents, N_episodes, with_constraints, buffer_length, max_steps, batch_size, temperature, num_iter_per_batch,
                 hidden_dim, lr_critic, lr_actor, gamma,is_entropy, entropy_coeff, entropy_coeff_decay, temperature_decay, mega_step, epsilon,perturb, epsilon_decay,
                 explore=True):
        self.env = env(max_steps=max_steps)
        self.N_agents = N_agents
        self.discount_factor = gamma
        self.mega_step = mega_step
        self.agents = [agent_class(perturb=perturb,with_constraints=with_constraints, gamma = gamma, num_agents=N_agents ,state_dim=2 ,action_dim=self.env.NUM_ACTIONS, mega_step=mega_step, temperature=temperature,hidden_dim=hidden_dim, lr_critic=lr_critic, lr_actor=lr_actor, is_entropy=is_entropy, temperature_decay=temperature_decay) for _ in range(N_agents)] # initialize agents classes
        self.replaybuffer = ReplayBuffer(max_length=buffer_length,gamma=gamma,state_dim=self.env.state_dim,proposal_dim=self.env.NUM_ACTIONS*mega_step,commitment_dim=2,action_dim=self.env.NUM_ACTIONS*mega_step,num_agents=N_agents)
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
    
    def onehot_to_int(self, onehot):
        """
        Convert one-hot tensor to integer
        """
        int = torch.argmax(onehot[0]).item()
        return int
    
    def int_to_onehot(self, integer):
        """
        Convert integer to one-hot tensor
        """
        onehot = torch.zeros(1,self.env.num_actions)
        onehot[0][integer] = 1
        return onehot
    
    def mega_one_hot_to_ints(self, mega_one_hot_vector):
        #  Convert a mega one-hot vector to a list of integers (np.array)
        mega_one_hot_vector = mega_one_hot_vector[0].detach().numpy()
        # Ensure one_hot_vector is valid
        if np.sum(mega_one_hot_vector) != 1:
            raise ValueError("Input vector is not a valid one-hot vector")
        
        # Find the index of the '1' in the original one-hot vector
        index = np.argmax(mega_one_hot_vector)
        
        if self.mega_step == 2:
            # Define the mappings based on the specific example cases
            if index == 0:
                return np.array([0, 0])
            elif index == 1:
                return np.array([0, 1])
            elif index == 2:
                return np.array([1, 0])
            elif index == 3:
                return np.array([1, 1])
            else:
                raise ValueError("Index out of range for the given one-hot vector")
        elif self.mega_step == 1:
            if index == 0:
                return np.array([0])
            elif index == 1:
                return np.array([1])
            else:
                raise ValueError("Index out of range for the given one-hot vector")

    def ints_to_mega_one_hot(self, ints):
        if self.mega_step == 2:
            # Map integers to one-hot vectors tensors
            if np.array_equal(ints, np.array([0, 0])):
                return torch.tensor(np.array([[1, 0, 0, 0]]), dtype=self.dtype)
            elif np.array_equal(ints, np.array([0, 1])):
                return torch.tensor(np.array([[0, 1, 0, 0]]), dtype=self.dtype)
            elif np.array_equal(ints, np.array([1, 0])):
                return torch.tensor(np.array([[0, 0, 1, 0]]), dtype=self.dtype)
            elif np.array_equal(ints, np.array([1, 1])):
                return torch.tensor(np.array([[0, 0, 0, 1]]), dtype=self.dtype)
            else:
                raise ValueError("Input integers are not valid for the given one-hot vector")
        elif self.mega_step == 1:
            if np.array_equal(ints, np.array([0])):
                return torch.tensor(np.array([[1, 0]]), dtype=self.dtype)
            elif np.array_equal(ints, np.array([1])):
                return torch.tensor(np.array([[0, 1]]), dtype=self.dtype)
            else:
                raise ValueError("Input integers are not valid for the given one-hot vector")
        
    def run_an_episode(self):
        """
        Each episode includes 1 step.
        """
        state = torch.tensor(self.env.reset(),dtype=self.dtype) # values from environment is np.array
        done = False
        accumulated_rewards = np.array([0,0])
        while not done:
            if self.mega_step == 2:
                mega_rewards=np.array([0.0,0.0])
            m = [agent.get_proposal(state, explore=self.explore, epsilon=self.epsilon).detach() for agent in self.agents] # one-hot value
            c = [agent.get_commitment(state, m[i], m[1-i], explore=self.explore, epsilon=self.epsilon).detach() for i, agent in enumerate(self.agents)] # one-hot value, [[0,1]] is commitment, [[1,0]] is no commitment
            a = [agent.get_unconstrained_action(state, explore=self.explore, epsilon=self.epsilon).detach() for agent in self.agents] # one-hot value
            is_mutual_commitment = self.agents[0].is_commit(c[0])*self.agents[1].is_commit(c[1]) # agent 0 and agent 1 both commit
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
                    next_state, rewards, done = self.env.step(actions_for_mega_step[:,k_step])
                    mega_rewards = mega_rewards*self.discount_factor + rewards
            else:
                next_state, mega_rewards, done = self.env.step(actions)
            accumulated_rewards = accumulated_rewards*self.discount_factor + mega_rewards
            if self.mega_step == 2:
                self.replaybuffer.push(state, m[0], m[1], c[0], c[1], is_mutual_commitment, a[0], a[1],\
                                torch.tensor(mega_rewards[0],dtype=self.dtype), torch.tensor(mega_rewards[1],dtype=self.dtype), torch.tensor(next_state,dtype=self.dtype),
                                torch.tensor(done, dtype=self.dtype)) # save tensors to buffer
            elif self.mega_step == 1:
                self.replaybuffer.push(state, m[0], m[1], c[0], c[1], is_mutual_commitment, a[0], a[1],\
                                torch.tensor(mega_rewards[0],dtype=self.dtype), torch.tensor(mega_rewards[1],dtype=self.dtype), torch.tensor(next_state,dtype=self.dtype),
                                torch.tensor(done, dtype=self.dtype))
            state = torch.tensor(next_state,dtype=self.dtype)
        self.replaybuffer.finish_an_episode()
        self.mega_returns.append(accumulated_rewards)

    def train(self):
        """
        Train the agents
        """
        action_label = ["C","D"]
        with torch.autograd.set_detect_anomaly(True):
            for epi_idx in range(self.N_episodes):
                self.run_an_episode()
                if (epi_idx+1)*self.max_steps/self.mega_step % self.batch_size == 0:
                    states, proposals, commitments, is_mutual_commitments, actions, rewards, next_states, dones, returns = self.replaybuffer.sample(self.batch_size)
                    mean_returns = np.mean(self.mega_returns, axis=0)
                    wandb.log({"social welfare": mean_returns.sum()})
                    wandb.log({"average return of agent 0": mean_returns[0]})
                    wandb.log({"average return of agent 1": mean_returns[1]})
                    self.mega_returns = []
                    for i, agent in enumerate(self.agents):
                        for _ in range(self.num_iter_per_batch): 
                            agent.update_critic(states, proposals[i], proposals[1-i], actions[i], actions[1-i], is_mutual_commitments, returns[i])
                            agent.update_unconstrained_policy(states, commitments[i], commitments[1-i], actions[i], actions[1-i], self.entropy_coeff)        
                            agent.update_commitment_policy(states, proposals[i], proposals[1-i], commitments[i], commitments[1-i], actions[i], actions[1-i], self.entropy_coeff, self.epsilon)
                            agent.update_proposal_policy(states, self.entropy_coeff, self.epsilon)

                            agent.update_coplayer_critic(states, proposals[i], proposals[1-i], actions[i], actions[1-i], is_mutual_commitments, returns[1-i])
                            agent.update_coplayer_unconstrained_policy(states, commitments[i], commitments[1-i], actions[i], actions[1-i], self.entropy_coeff)        
                            agent.update_coplayer_commitment_policy(states, proposals[i], proposals[1-i], commitments[i], commitments[1-i], actions[i], actions[1-i], self.entropy_coeff, self.epsilon)
                            agent.update_coplayer_proposal_policy(states, self.entropy_coeff, self.epsilon)
                        
                        wandb.log({"entropy_coeff":self.entropy_coeff})
                        self.entropy_coeff = np.maximum(0.0, self.entropy_coeff - self.entropy_coeff_decay)
                        wandb.log({"epsilon":self.epsilon})
                        if (epi_idx+1)*self.max_steps/self.mega_step /self.batch_size % 100 == 0:
                            self.epsilon = np.maximum(0.0, self.epsilon - self.epsilon_decay)
                        if self.mega_step == 1:
                            wandb.log({"policy prob of cooperation for agent {}".format(i): agent.unconstrained_actor(states[0]).squeeze()[0].detach()})
                            for ii in range(2):
                                for jj in range(2):
                                    proposal_self_onehot = self.int_to_onehot(ii)
                                    proposal_coplayer_onehot = self.int_to_onehot(jj)
                                    wandb.log({"commitment prob for agent {} given [{},{}]".format(i,action_label[ii],action_label[jj]): agent.commit_actor(torch.cat((states[0].unsqueeze(dim=0), proposal_self_onehot, proposal_coplayer_onehot),dim=1)).detach().squeeze()[1].numpy()})
                                    wandb.log({"critic value for agent{} of [{},{}]:".format(i, ii,jj): agent.critic(torch.cat((states[0].unsqueeze(0), proposal_self_onehot, proposal_coplayer_onehot),dim=1)).detach()})
                            wandb.log({"proposal prob of cooperation for agent {}".format(i): agent.proposing_actor(states[0]).squeeze()[0].detach()})    
                        elif self.mega_step == 2:
                            wandb.log({"policy prob of (C,C) for agent {}".format(i): agent.unconstrained_actor(states[0]).squeeze()[0].detach()})
                            wandb.log({"policy prob of (C,D) for agent {}".format(i): agent.unconstrained_actor(states[0]).squeeze()[1].detach()})
                            wandb.log({"policy prob of (D,C) for agent {}".format(i): agent.unconstrained_actor(states[0]).squeeze()[2].detach()})
                            wandb.log({"policy prob of (D,D) for agent {}".format(i): agent.unconstrained_actor(states[0]).squeeze()[3].detach()})
                            for int0_self in range(self.env.NUM_ACTIONS):
                                for int1_self in range(self.env.NUM_ACTIONS):
                                    for int0_coplayer in range(self.env.NUM_ACTIONS):
                                        for int1_coplayer in range(self.env.NUM_ACTIONS):
                                            proposal_self_mega_onehot = self.ints_to_mega_one_hot(np.array([int0_self, int1_self]))
                                            proposal_coplayer_mega_onehot = self.ints_to_mega_one_hot(np.array([int0_coplayer, int1_coplayer]))
                                            wandb.log({"commitment prob for agent {} given [{}{},{}{}]".format(i,action_label[int0_self],action_label[int1_self],action_label[int0_coplayer],action_label[int1_coplayer]): agent.commit_actor(torch.cat((states[0].unsqueeze(dim=0), proposal_self_mega_onehot, proposal_coplayer_mega_onehot),dim=1)).detach().squeeze()[1].numpy()})
                            wandb.log({"proposal prob of (C,C) for agent {}".format(i): agent.proposing_actor(states[0]).squeeze()[0].detach()})
                            wandb.log({"proposal prob of (C,D) for agent {}".format(i): agent.proposing_actor(states[0]).squeeze()[1].detach()})
                            wandb.log({"proposal prob of (D,C) for agent {}".format(i): agent.proposing_actor(states[0]).squeeze()[2].detach()})
                            wandb.log({"proposal prob of (D,D) for agent {}".format(i): agent.proposing_actor(states[0]).squeeze()[3].detach()})