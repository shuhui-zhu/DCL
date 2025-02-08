import numpy as np

class IteratedPrisonersDilemma:
    """
    A two-agent vectorized environment for the Prisoner's Dilemma game.
    Possible actions for each agent are (C)ooperate and (D)efect.
    """
    NAME = 'IPD'
    NUM_AGENTS = 2
    NUM_ACTIONS = 2
    NUM_STATES = 1 

    def __init__(self, max_steps, state_dim=2, num_actions=2):
        self.max_steps = max_steps
        self.state_dim = state_dim
        self.payout_mat = np.array([[-1., 0.], [-3., -2.]])
        self.dummy_state = np.array([[1., 0.]])

    def reset(self):
        self.step_count = 0
        return self.dummy_state

    def step(self, actions):
        ac0, ac1 = actions
        self.step_count += 1

        rewards = np.array([self.payout_mat[ac1][ac0], self.payout_mat[ac0][ac1]])

        done = (self.step_count == self.max_steps)

        return self.dummy_state, rewards, done

class IteratedPureConflict:
    """
    A two-agent vectorized environment for a pure-conflicting interests game.
    Possible actions for each agent are (C)ooperate and (D)efect.
    """
    NAME = 'IPC'
    NUM_AGENTS = 2
    NUM_ACTIONS = 2
    NUM_STATES = 1 

    def __init__(self, max_steps, state_dim=2, num_actions=2):
        self.max_steps = max_steps
        self.state_dim = state_dim
        self.payout_mat = np.array([[0., 2.], [-1., 0.]])
        self.dummy_state = np.array([[1., 0.]])

    def reset(self):
        self.step_count = 0
        return self.dummy_state

    def step(self, actions):
        ac0, ac1 = actions
        self.step_count += 1

        rewards = np.array([self.payout_mat[ac1][ac0], self.payout_mat[ac0][ac1]])

        done = (self.step_count == self.max_steps)

        return self.dummy_state, rewards, done