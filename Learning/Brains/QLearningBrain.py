# QLearningBrain.py

import random
import json
from collections import defaultdict

class QLearningBrain:
    """
    Generic Q-learning brain.
    Knows NOTHING about environment or agent.
    """

    def __init__(self, alpha=0.3, gamma=0.95, epsilon=0.2, q_table=None):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.Q = q_table if q_table is not None else defaultdict(dict)

    # --------------------------------------------------
    def select_action(self, state, valid_actions, mode="train"):
        self._ensure_state(state, valid_actions)

        if mode == "train" and random.random() < self.epsilon:
            return random.choice(valid_actions)

        return self._greedy(state, valid_actions)

    # --------------------------------------------------
    def update(self, prev_state, action, reward, new_state, done):
        self._ensure_state(prev_state)
        self._ensure_state(new_state)

        q_old = self.Q[prev_state][action]
        q_next = 0.0 if done else max(self.Q[new_state].values())

        self.Q[prev_state][action] = (
            q_old + self.alpha * (reward + self.gamma * q_next - q_old)
        )

    # --------------------------------------------------
    def _greedy(self, state, valid_actions):
        best = max(self.Q[state][a] for a in valid_actions)
        best_actions = [a for a in valid_actions if self.Q[state][a] == best]
        return random.choice(best_actions)

    def _ensure_state(self, state, actions=None):
        if state not in self.Q:
            self.Q[state] = {}
        if actions:
            for a in actions:
                self.Q[state].setdefault(a, 0.0)

    # --------------------------------------------------
    def save(self, path):
        with open(path, "w") as f:
            json.dump({str(k): v for k, v in self.Q.items()}, f)

    def load(self, path):
        with open(path, "r") as f:
            raw = json.load(f)
        self.Q = {eval(k): v for k, v in raw.items()}
