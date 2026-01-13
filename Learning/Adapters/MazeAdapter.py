# Learning/Adapters/MazeAdapter.py
from Learning.Adapters.TaskAdapter import TaskAdapter


class MazeAdapter(TaskAdapter):
    """
    Maze adapter:
      - 4-direction actions
      - numeric state for BOTH Q-learning and GenomeBrain:
          walls 4 bits
        + goal-adj 4 bits
        + last_action one-hot 5 bits (None + 4 actions)
      => total 13 inputs

    Note: This still uses obs["goals"][0] to compute goal adjacency.
    If the assignment forbids goal coordinates, we can change World.observacaoPara
    to provide goal-adj sensor booleans instead.
    """

    ACTIONS = ["up", "down", "left", "right"]
    ACTION_TO_IDX = {a: i for i, a in enumerate(ACTIONS)}

    def observation_size(self) -> int:
        # 4 walls + 4 goal-adj + 5 last-action one-hot (None + 4)
        return 13

    def action_size(self) -> int:
        return len(self.ACTIONS)

    def build_state(self, agent, obs, env):
        x, y = obs["posicao"]
        gx, gy = obs["goals"][0]

        # walls (4)
        wU = 1.0 if env.is_blocked(x - 1, y) else 0.0
        wD = 1.0 if env.is_blocked(x + 1, y) else 0.0
        wL = 1.0 if env.is_blocked(x, y - 1) else 0.0
        wR = 1.0 if env.is_blocked(x, y + 1) else 0.0

        # goal adjacency (4)
        gU = 1.0 if (x - 1, y) == (gx, gy) else 0.0
        gD = 1.0 if (x + 1, y) == (gx, gy) else 0.0
        gL = 1.0 if (x, y - 1) == (gx, gy) else 0.0
        gR = 1.0 if (x, y + 1) == (gx, gy) else 0.0

        # last action one-hot (5): [None, up, down, left, right]
        last = getattr(agent, "last_action", None)
        la = [0.0] * 5
        if last is None:
            la[0] = 1.0
        else:
            la[1 + self.ACTION_TO_IDX[last]] = 1.0

        return (wU, wD, wL, wR, gU, gD, gL, gR, *la)

    def valid_actions(self, agent, env, obs=None):
        x, y = agent.x, agent.y
        candidates = {
            "up":    (x - 1, y),
            "down":  (x + 1, y),
            "left":  (x, y - 1),
            "right": (x, y + 1),
        }
        return [a for a, (nx, ny) in candidates.items() if not env.is_blocked(nx, ny)]

    def action_to_move(self, agent, action):
        x, y = agent.x, agent.y
        if action == "up":
            return (x - 1, y)
        if action == "down":
            return (x + 1, y)
        if action == "left":
            return (x, y - 1)
        if action == "right":
            return (x, y + 1)
        return (x, y)

    def is_terminal(self, agent, obs, env):
        return obs["posicao"] in obs["goals"] or getattr(agent, "reached_goal", False)

    def reward(self, agent, prev_state, action, new_state, obs, step, max_steps):
        """
        Simple RL reward for maze:
          - small step penalty
          - penalize revisits
          - big reward on reaching goal
        """
        if not hasattr(agent, "visited_positions"):
            agent.visited_positions = set()

        pos = obs["posicao"]

        r = -0.05
        if pos in agent.visited_positions:
            r -= 0.2
        else:
            agent.visited_positions.add(pos)

        if pos in obs["goals"]:
            r += 50.0 * (1 - step / max_steps)

        return r
