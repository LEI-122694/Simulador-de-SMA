# Agents/LearningAgent.py
from Agents.Agent import Agent

class LearningAgent(Agent):
    """
    Generic learning agent:
      - adapter defines state/actions/reward
      - brain decides + learns
    """

    def __init__(self, name, env, start_pos, adapter, brain):
        super().__init__(name, env, start_pos)
        self.adapter = adapter
        self.brain = brain

        self.current_obs = None
        self.state = None
        self.prev_state = None
        self.prev_action = None

        self.last_action = None
        self.reached_goal = False

    def comunica(self, mensagem, de_agente):
        # keep empty unless you need comms later
        pass

    def observacao(self, obs):
        self.current_obs = obs
        self.state = self.adapter.build_state(self, obs, self.env)

        if self.adapter.is_terminal(self, obs, self.env):
            self.reached_goal = True

    def age(self):
        if self.reached_goal or self.current_obs is None:
            return None

        valid = self.adapter.valid_actions(self, self.env, self.current_obs)
        if not valid:
            return None

        action = self.brain.select_action(self.state, valid, mode=self.mode)

        # track for adapter + RL/evo
        self.prev_state = self.state
        self.prev_action = action
        self.last_action = action

        return self.adapter.action_to_move(self, action)

    def avaliacaoEstadoAtual(self, recompensa):
        # only RL brains will implement update; evo brains can ignore
        if (
            hasattr(self.brain, "update") and
            self.prev_state is not None and
            self.prev_action is not None
        ):
            done = self.reached_goal
            self.brain.update(self.prev_state, self.prev_action, recompensa, self.state, done)
