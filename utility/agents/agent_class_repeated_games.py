"""DCL agent for repeated matrix games (IPD, IPC).

The full implementation lives in :class:`utility.agents.base_agent.DCL_Agent`;
this module exists for backwards-compatible imports.
"""
from utility.agents.base_agent import DCL_Agent


class DCL_Agent_Repeated_Games(DCL_Agent):
    """Repeated-game agent. Identical to :class:`DCL_Agent`."""
    pass
