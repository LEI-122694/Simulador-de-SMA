# Agents/LearningAgent.py
from Agents.Agent import Agent

class LearningAgent(Agent):
    """
    Generic learning agent:
      - adapter defines state/actions/reward/terminal
      - brain decides + (optional) learns
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
        pass

    # ------------------------------------------------------------
    # Important: consistent reset between episodes
    # ------------------------------------------------------------
    def episode_reset(self):
        self.current_obs = None
        self.state = None
        self.prev_state = None
        self.prev_action = None
        self.last_action = None
        self.reached_goal = False

        # reward shapers may use this
        if hasattr(self, "visited_positions"):
            self.visited_positions.clear()

        # recurrent genome brain
        if hasattr(self.brain, "reset") and callable(getattr(self.brain, "reset")):
            self.brain.reset()

    # ------------------------------------------------------------
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

        self.prev_state = self.state
        self.prev_action = action
        self.last_action = action

        return self.adapter.action_to_move(self, action)

    def avaliacaoEstadoAtual(self, recompensa):
        if (
            hasattr(self.brain, "update") and callable(getattr(self.brain, "update")) and
            self.prev_state is not None and self.prev_action is not None
        ):
            done = self.reached_goal
            self.brain.update(self.prev_state, self.prev_action, recompensa, self.state, done)
