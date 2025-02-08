import numpy as np

class GridSocialDilemmaEnv: 
    NAME = 'GridGame'
    NUM_AGENTS = 2
    NUM_ACTIONS = 2
    
    def __init__(self, max_steps=16, grid_size=4, k=2, num_agents = 2, num_actions = 2):
        super(GridSocialDilemmaEnv, self).__init__()
        self.max_steps = max_steps
        self.grid_size = grid_size
        self.num_agents = num_agents
        self.num_actions = num_actions
        self.state_dim = grid_size*num_agents
        self.conflict_level = k

        self.initial_position_agent_0 = np.identity(self.grid_size)[0]
        self.initial_position_agent_1 = np.identity(self.grid_size)[-1]
        self.initial_state = np.concatenate([self.initial_position_agent_0, self.initial_position_agent_1],axis=0).reshape(1,-1)
        self.agent_positions = [0, self.grid_size - 1]

    def reset(self):
        self.current_step = 0
        self.agent_positions = [0, self.grid_size - 1]
        state = self.initial_state
        return state

    def step(self, actions):
        self.current_step += 1
        
        for i in range(self.num_agents):
            if actions[i]==0: # stay
                if self.agent_positions[i] > 0:
                    self.agent_positions[i] -= 1
                elif self.agent_positions[i]==0:
                    self.agent_positions[i] = 0
                else:
                    AssertionError("Negative position")
            elif actions[i]==1:
                if self.agent_positions[i] < self.grid_size - 1:
                    self.agent_positions[i] += 1
                elif self.agent_positions[i]==self.grid_size-1:
                    self.agent_positions[i] = self.grid_size - 1
                else:
                    AssertionError("Over the grid size")
        
        # reveal agents' positions
        state = np.concatenate([np.identity(self.grid_size)[self.agent_positions[0]], np.identity(self.grid_size)[self.agent_positions[1]]],axis=0).reshape(1,-1)            
        immediate_rewards = self.calculate_rewards()
        
        done = (self.current_step == self.max_steps)
        
        return state, immediate_rewards, done

    def calculate_rewards(self):
        pos_0, pos_1 = self.agent_positions
        reward_0 = pos_0 - self.conflict_level*(self.grid_size-1 - pos_1)
        reward_1 = self.grid_size-1 - pos_1 - self.conflict_level*pos_0
        rewards = np.array([reward_0, reward_1])
        return rewards

    def render(self):
        grid = [' ' for _ in range(self.grid_size)]
        if self.agent_positions[0] == self.agent_positions[1]:
            grid[self.agent_positions[0]] = 'AB'
        else:
            grid[self.agent_positions[0]] = 'A'
            grid[self.agent_positions[1]] = 'B'
        print('|'.join(grid))
        print(f"Agent A Position: {self.agent_positions[0]}, Agent B Position: {self.agent_positions[1]}")