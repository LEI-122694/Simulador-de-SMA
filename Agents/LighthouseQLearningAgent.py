import random
import json
from Agents.Agent import Agent


class LighthouseQLearningAgent(Agent):
    """
    Q-learning agent for the FAROL environment.
    State = (direcao_farol, blocked_N, blocked_S, blocked_E, blocked_W,
             blocked_NE, blocked_NW, blocked_SE, blocked_SW)
    Complies with enunciado: uses only sensors + direction, no coordinates.
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

        super().__init__(name, env, start_pos)

        self.Q = q_table if q_table is not None else {}

        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

        self.current_state = None
        self.last_state = None
        self.last_action = None

        self.reached_goal = False

    # ------------------------------------------------------------
    def comunica(self, mensagem, de_agente):
        pass

    # ------------------------------------------------------------
    # OBSERVATION → RL STATE
    # ------------------------------------------------------------
    def observacao(self, obs):
        """
        Build legal RL state using ONLY:
        - direction to lighthouse
        - 8 obstacle sensors
        """
        direction = obs.get("direcao_farol", "HERE")

        if direction == "HERE":
            self.reached_goal = True

        x, y = self.x, self.y
        env = self.env

        # 8-direction obstacle sensing
        blocked = {
            "N":  not env.is_valid_position(x - 1, y),
            "S":  not env.is_valid_position(x + 1, y),
            "E":  not env.is_valid_position(x, y + 1),
            "W":  not env.is_valid_position(x, y - 1),

            "NE": not env.is_valid_position(x - 1, y + 1),
            "NW": not env.is_valid_position(x - 1, y - 1),
            "SE": not env.is_valid_position(x + 1, y + 1),
            "SW": not env.is_valid_position(x + 1, y - 1),
        }

        # Full sensor state
        state = (
            direction,
            blocked["N"], blocked["S"], blocked["E"], blocked["W"],
            blocked["NE"], blocked["NW"], blocked["SE"], blocked["SW"]
        )

        self.current_state = state

        # Add to Q-table if first encounter
        if state not in self.Q:
            self.Q[state] = {a: 0.0 for a in self.ACTIONS}

    # ------------------------------------------------------------
    # Q-LEARNING UPDATE
    # ------------------------------------------------------------
    def avaliacaoEstadoAtual(self, recompensa):

        if self.mode == "test":
            return

        if self.last_state is None or self.last_action is None:
            return

        # Ensure state exists
        if self.current_state not in self.Q:
            self.Q[self.current_state] = {a: 0.0 for a in self.ACTIONS}

        q_old = self.Q[self.last_state][self.last_action]
        q_next = max(self.Q[self.current_state].values())

        self.Q[self.last_state][self.last_action] = \
            q_old + self.alpha * (recompensa + self.gamma * q_next - q_old)

    # ------------------------------------------------------------
    # ACTION SELECTION
    # ------------------------------------------------------------
    def age(self):
        if self.reached_goal:
            self.last_state = None
            self.last_action = None
            return None

        state = self.current_state
        valid_actions = self._valid_actions()

        if not valid_actions:
            return None

        if self.mode == "test":
            action = self._greedy_action(state, valid_actions)
        else:
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
        env = self.env
        valid = []
        for a in self.ACTIONS:
            dx, dy = self.ACTION_TO_DELTA[a]
            nx, ny = self.x + dx, self.y + dy
            if env.is_valid_position(nx, ny):
                valid.append(a)
        return valid

    def _greedy_action(self, state, valid_actions):
        qvals = self.Q[state]
        best_val = max(qvals[a] for a in valid_actions)
        best = [a for a in valid_actions if qvals[a] == best_val]
        return random.choice(best)

    # ------------------------------------------------------------
    # SAVE / LOAD POLICY
    # ------------------------------------------------------------
    def save_policy(self, filename="policy_farol.json"):
        serializable_Q = {str(k): v for k, v in self.Q.items()}
        with open(filename, "w") as f:
            json.dump(serializable_Q, f)

    def load_policy(self, filename="policy_farol.json"):
        with open(filename, "r") as f:
            raw = json.load(f)

        self.Q = {}
        for k, v in raw.items():
            state = eval(k)
            self.Q[state] = v
