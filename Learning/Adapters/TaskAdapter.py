# Learning/Adapters/TaskAdapter.py
from abc import ABC, abstractmethod

class TaskAdapter(ABC):
    """
    Environment–specific logic:
      - how to build the state for the brain
      - which actions are valid
      - how an action maps to a movement (x,y)
      - when an episode is terminal
      - reward function (for learning brains)
    """

    # subclasses override if they want a static list
    ACTIONS = []

    @abstractmethod
    def build_state(self, agent, obs, env):
        """Return a hashable state representation for the brain."""
        pass

    @abstractmethod
    def valid_actions(self, agent, env, obs=None):
        """Return list of action labels that are currently legal."""
        pass

    @abstractmethod
    def action_to_move(self, agent, action):
        """Map an action label to a (nx, ny) movement."""
        pass

    @abstractmethod
    def is_terminal(self, agent, obs, env):
        """Return True if episode should be considered finished."""
        pass

    def reward(self, agent, prev_state, action, new_state, obs, step, max_steps):
        """
        Default reward (0). RL brains can call this if the task defines it.
        Evolutionary brains can ignore it.
        """
        return 0.0
