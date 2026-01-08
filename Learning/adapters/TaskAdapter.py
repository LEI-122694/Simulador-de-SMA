from abc import ABC, abstractmethod

class TaskAdapter(ABC):
    ACTIONS = []  # subclasses override as class attribute

    @abstractmethod
    def build_state(self, agent, obs, env):
        pass

    @abstractmethod
    def valid_actions(self, agent, env, obs=None):
        pass

    @abstractmethod
    def action_to_move(self, agent, action):
        pass

    @abstractmethod
    def is_terminal(self, agent, obs, env):
        pass

    def reward(self, agent, prev_state, action, new_state, obs, step, max_steps):
        return 0.0
