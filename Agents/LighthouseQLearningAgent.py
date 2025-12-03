# Agents/LighthouseQLearningAgent.py

import random
import json
from Agents.Agent import Agent


class LighthouseQLearningAgent(Agent):
    """
    Q-learning agent for the FAROL environment.

    - State:  'direcao_farol' string from the environment (N, NE, E, ..., HERE)
    - Actions: 8 moves (N, S, E, W, NE, NW, SE, SW)
    - Policy: epsilon-greedy over a tabular Q-table

    This is COMPLETELY separate from the rule-based LighthouseAgent.
    """

    ACTIONS = ["N", "S", "E", "W", "NE", "NW", "SE", "SW"]

    ACTION_TO_DELTA = {
        "N":  (-1,  0),
        "S":  ( 1,  0),
        "E":  ( 0,  1),
        "W":  ( 0, -1),
        "NE": (-1,  1),
        "NW": (-1, -1),
        "SE": ( 1,  1),
        "SW": ( 1, -1),
    }

    def __init__(self, name, env, start_pos,
                 alpha=0.3, gamma=0.95, epsilon=0.2,
                 q_table=None):
        """
        q_table: optional dict to share a learned Q-table across episodes.
        """
        super().__init__(name, env, start_pos)

        # Q[state][action] = value
        self.Q = q_table if q_table is not None else {}

        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

        self.last_state = None
        self.last_action = None

        self.current_state = None
        self.current_obs = None
        self.reached_goal = False

    # ------------------------------------------------------------
    def comunica(self, mensagem, de_agente):
        # no comms for Farol RL
        pass

    # ------------------------------------------------------------
    # OBSERVATION → RL state
    # ------------------------------------------------------------
    def observacao(self, obs):
        """
        We only use 'direcao_farol' as the discrete RL state.
        """
        self.current_obs = obs
        direction = obs.get("direcao_farol", "HERE")
        self.current_state = direction

        if direction == "HERE":
            self.reached_goal = True

        # ensure state exists in Q-table
        if direction not in self.Q:
            self.Q[direction] = {a: 0.0 for a in self.ACTIONS}

    # ------------------------------------------------------------
    # Q-learning update
    # ------------------------------------------------------------
    def avaliacaoEstadoAtual(self, recompensa):
        """
        Q(s,a) <- Q(s,a) + α [ r + γ max_a' Q(s',a') - Q(s,a) ]
        """
        # no learning in test mode
        if self.mode == "test":
            return

        if self.last_state is None or self.last_action is None:
            return

        # safety: ensure states exist
        if self.last_state not in self.Q:
            self.Q[self.last_state] = {a: 0.0 for a in self.ACTIONS}
        if self.current_state not in self.Q:
            self.Q[self.current_state] = {a: 0.0 for a in self.ACTIONS}

        q_sa = self.Q[self.last_state][self.last_action]
        max_next = max(self.Q[self.current_state].values())

        new_q = q_sa + self.alpha * (recompensa + self.gamma * max_next - q_sa)
        self.Q[self.last_state][self.last_action] = new_q

    # ------------------------------------------------------------
    # ACTION SELECTION (ε-greedy)
    # ------------------------------------------------------------
    def age(self):
        """
        Choose ε-greedy action from current_state and return next (x, y).
        """
        if self.reached_goal:
            self.last_state = None
            self.last_action = None
            return None

        state = self.current_state
        if state not in self.Q:
            self.Q[state] = {a: 0.0 for a in self.ACTIONS}

        valid_actions = self._valid_actions()
        if not valid_actions:
            self.last_state = None
            self.last_action = None
            return None

        # TEST mode → greedy
        if self.mode == "test":
            action = self._greedy_action(state, valid_actions)
        else:
            # TRAIN mode → epsilon-greedy
            if random.random() < self.epsilon:
                action = random.choice(valid_actions)
            else:
                action = self._greedy_action(state, valid_actions)

        self.last_state = state
        self.last_action = action

        dx, dy = self.ACTION_TO_DELTA[action]
        return (self.x + dx, self.y + dy)

    # ------------------------------------------------------------
    def _valid_actions(self):
        valid = []
        for a in self.ACTIONS:
            dx, dy = self.ACTION_TO_DELTA[a]
            nx, ny = self.x + dx, self.y + dy
            if self.env.is_valid_position(nx, ny):
                valid.append(a)
        return valid

    def _greedy_action(self, state, valid_actions):
        qvals = self.Q[state]
        best = max(qvals[a] for a in valid_actions)
        best_actions = [a for a in valid_actions if qvals[a] == best]
        return random.choice(best_actions)

    # ------------------------------------------------------------
    # SAVE / LOAD POLICY (for test mode)
    # ------------------------------------------------------------
    def save_policy(self, filename="policy_farol.json"):
        with open(filename, "w") as f:
            json.dump(self.Q, f)

    def load_policy(self, filename="policy_farol.json"):
        with open(filename, "r") as f:
            self.Q = json.load(f)
