import numpy as np
import torch
import scipy

class ReplayBuffer(object):
    """
    Replay Buffer for multi-agent RL with parallel rollouts
    """
    def __init__(self, max_length, gamma, state_dim, proposal_dim, commitment_dim, action_dim, num_agents=2):
        """
        Inputs:
            max_steps (int): Maximum number of timepoints to store in buffer
            num_agents (int): Number of agents in environment
            state_dims (list of ints): number of obervation dimensions for each
                                     agent
            proposal_dims (list of ints): number of proposal dimensions for each agent
            commitment_dims (list of ints): number of commitment dimensions for each agent
            action_dims (list of ints): number of action dimensions for each agent
        """
        self.max_length = max_length
        self.num_agents = num_agents
        self.gamma = gamma

        self.state_buffs = torch.zeros((max_length,state_dim),dtype=torch.float32)
        self.proposal_buffs = [torch.zeros((max_length,proposal_dim),dtype=torch.float32) for _ in range(num_agents)]
        self.commitment_buffs = [torch.zeros((max_length,commitment_dim),dtype=torch.float32) for _ in range(num_agents)]
        self.is_mutual_commitment_buffs = torch.zeros((max_length,1),dtype=torch.float32)
        self.action_buffs = [torch.zeros((max_length,action_dim),dtype=torch.float32) for _ in range(num_agents)]
        self.reward_buffs = [torch.zeros((max_length,1),dtype=torch.float32) for _ in range(num_agents)]
        self.next_state_buffs = torch.zeros((max_length,state_dim),dtype=torch.float32)
        self.done_buffs = torch.zeros((max_length,1),dtype=torch.float32)
        self.return_buffs = [torch.zeros((max_length,1),dtype=torch.float32) for _ in range(num_agents)]
        self.ptr, self.path_start_idx = 0, 0

    def __len__(self):
        return len(self.state_buffs)
    
    def discount_cumsum(self, reward_list):
        """
        magic from rllab for computing discounted cumulative sums of vectors.

        input: 
            vector x, 
            [x0, 
            x1, 
            x2]

        output:
            [x0 + discount * x1 + discount^2 * x2,  
            x1 + discount * x2,
            x2]
        """
        reward_list = np.array(list(reward_list))
        cum_return = scipy.signal.lfilter([1], [1, float(-self.gamma)], reward_list[::-1], axis=0)[::-1]
        return cum_return.copy()

    def push(self, state, proposal_0, proposal_1, commitment_0, commitment_1, is_mutual_commitment, action_0, action_1, reward_0, reward_1, next_state, done):
        """
        Add new experience to buffer
        """
        assert self.ptr < self.max_length
        self.state_buffs[self.ptr] = state
        self.proposal_buffs[0][self.ptr] = proposal_0
        self.proposal_buffs[1][self.ptr] = proposal_1
        self.commitment_buffs[0][self.ptr] = commitment_0
        self.commitment_buffs[1][self.ptr] = commitment_1
        self.is_mutual_commitment_buffs[self.ptr] = is_mutual_commitment
        self.action_buffs[0][self.ptr] = action_0
        self.action_buffs[1][self.ptr] = action_1
        self.reward_buffs[0][self.ptr] = reward_0
        self.reward_buffs[1][self.ptr] = reward_1
        self.next_state_buffs[self.ptr] = next_state
        self.done_buffs[self.ptr] = done
        self.ptr += 1

    def finish_an_episode(self):
        path_slice = slice(self.path_start_idx, self.ptr)
        # At the end of episode, compute return for each agent
        self.return_buffs[0][path_slice] = torch.tensor(self.discount_cumsum(self.reward_buffs[0][path_slice]),dtype=torch.float32) # This is correct, we don't need to exclude the last state by [:-1]
        self.return_buffs[1][path_slice] = torch.tensor(self.discount_cumsum(self.reward_buffs[1][path_slice]), dtype=torch.float32)
        self.path_start_idx = self.ptr

    def sample(self, sample_size):
        assert self.ptr == self.max_length
        self.ptr, self.path_start_idx = 0, 0
        return (
            self.state_buffs,
            self.proposal_buffs,
            self.commitment_buffs,
            self.is_mutual_commitment_buffs,
            self.action_buffs,
            self.reward_buffs,
            self.next_state_buffs,
            self.done_buffs,
            self.return_buffs
        )