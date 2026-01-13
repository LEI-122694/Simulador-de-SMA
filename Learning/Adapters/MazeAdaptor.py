# Learning/Adapters/MazeAdapter.py
from Learning.Adapters.TaskAdapter import TaskAdapter


class MazeAdapter(TaskAdapter):

    ACTIONS = ["up", "down", "left", "right"]

    def build_state(self, agent, obs, env):
        x, y = obs["posicao"]
        gx, gy = obs["goals"][0]

        # walls
        wU = 1 if env.is_blocked(x - 1, y) else 0
        wD = 1 if env.is_blocked(x + 1, y) else 0
        wL = 1 if env.is_blocked(x, y - 1) else 0
        wR = 1 if env.is_blocked(x, y + 1) else 0

        # goal adjacency
        gU = 1 if (x - 1, y) == (gx, gy) else 0
        gD = 1 if (x + 1, y) == (gx, gy) else 0
        gL = 1 if (x, y - 1) == (gx, gy) else 0
        gR = 1 if (x, y + 1) == (gx, gy) else 0

        last_action = getattr(agent, "last_action", None)

        # tuple is hashable → good as Q-table key
        return (wU, wD, wL, wR, gU, gD, gL, gR, last_action)

    def valid_actions(self, agent, env, obs=None):
        x, y = agent.x, agent.y
        candidates = {
            "up":    (x - 1, y),
            "down":  (x + 1, y),
            "left":  (x, y - 1),
            "right": (x, y + 1),
        }
        return [
            a for a, (nx, ny) in candidates.items()
            if not env.is_blocked(nx, ny)
        ]

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
        r = -0.05  # step cost

        if pos in agent.visited_positions:
            r -= 0.2
        else:
            agent.visited_positions.add(pos)

        if pos in obs["goals"]:
            r += 50.0 * (1 - step / max_steps)

        return r
