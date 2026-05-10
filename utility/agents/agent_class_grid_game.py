"""DCL agent for the grid social-dilemma game.

Same algorithm as the repeated-game agent (mega_step=1) but the joint observation
is ``grid_size * num_agents``-dimensional, so we derive ``state_dim`` here.
"""
from utility.agents.base_agent import DCL_Agent


class DCL_Agent_Grid_Game(DCL_Agent):
    def __init__(self, *, grid_size, num_agents, **kwargs):
        kwargs.pop("mega_step", None)  # grid game is always single-step
        super().__init__(
            state_dim=grid_size * num_agents,
            num_agents=num_agents,
            mega_step=1,
            **kwargs,
        )
        self.grid_size = grid_size
