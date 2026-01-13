# Learning/Brains/QLearningBrain.py
import random
import json
from collections import defaultdict
from typing import Any, Dict, List, Optional


class QLearningBrain:
    """
    Generic tabular Q-learning brain.
    Environment-agnostic: only sees (state, valid_actions).
    """

    def __init__(self, alpha=0.3, gamma=0.95, epsilon=0.2, q_table=None):
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.epsilon = float(epsilon)
        self.Q = q_table if q_table is not None else defaultdict(dict)

    # --------------------------------------------------
    def select_action(self, state, valid_actions, mode="train"):
        self._ensure_state(state, valid_actions)

        if mode == "train" and random.random() < self.epsilon:
            return random.choice(valid_actions)

        return self._greedy(state, valid_actions)

    # --------------------------------------------------
    def update(self, prev_state, action, reward, new_state, done, next_valid_actions: Optional[List[str]] = None):
        """
        Optionally accept next_valid_actions so q_next is computed safely and consistently.
        """
        self._ensure_state(prev_state, [action])

        if next_valid_actions is not None:
            self._ensure_state(new_state, next_valid_actions)
        else:
            # ensure new_state exists, but it may have no actions yet
            self._ensure_state(new_state)

        q_old = self.Q[prev_state].get(action, 0.0)

        if done:
            q_next = 0.0
        else:
            next_vals = list(self.Q[new_state].values())
            q_next = max(next_vals) if next_vals else 0.0

        self.Q[prev_state][action] = (
            q_old + self.alpha * (float(reward) + self.gamma * q_next - q_old)
        )

    # --------------------------------------------------
    def _greedy(self, state, valid_actions):
        self._ensure_state(state, valid_actions)
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
        # NOTE: eval is OK for coursework, but don't use in production.
        self.Q = {eval(k): v for k, v in raw.items()}
