# Learning/Adapters/FarolAdapter.py
from Learning.Adapters.TaskAdapter import TaskAdapter


class FarolAdapter(TaskAdapter):
    """
    Farol adapter:
      - 8-direction actions
      - numeric state for BOTH Q-learning and GenomeBrain:
          direction one-hot (9: HERE + 8 compass)
        + blocked 8 bits
      => total 17 inputs
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

    # Direction encoding
    DIRS = ["HERE", "N", "S", "E", "W", "NE", "NW", "SE", "SW"]
    DIR_TO_IDX = {d: i for i, d in enumerate(DIRS)}

    def observation_size(self) -> int:
        # 9 (direction one-hot incl HERE) + 8 (blocked bits)
        return 17

    def action_size(self) -> int:
        return len(self.ACTIONS)

    def build_state(self, agent, obs, env):
        direction = obs.get("direcao_farol", "HERE")
        if direction not in self.DIR_TO_IDX:
            direction = "HERE"

        x, y = agent.x, agent.y
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

        # direction one-hot (9)
        dir_vec = [0.0] * 9
        dir_vec[self.DIR_TO_IDX[direction]] = 1.0

        # blocked bits (8) in ACTIONS order
        blocked_vec = [1.0 if blocked[a] else 0.0 for a in self.ACTIONS]

        # Return as tuple (hashable for Q-learning; numeric for GenomeBrain)
        return tuple(dir_vec + blocked_vec)

    def valid_actions(self, agent, env, obs=None):
        valid = []
        for a in self.ACTIONS:
            dx, dy = self.ACTION_TO_DELTA[a]
            nx, ny = agent.x + dx, agent.y + dy
            if env.is_valid_position(nx, ny):
                valid.append(a)
        return valid

    def action_to_move(self, agent, action):
        dx, dy = self.ACTION_TO_DELTA[action]
        return (agent.x + dx, agent.y + dy)

    def is_terminal(self, agent, obs, env):
        return obs.get("direcao_farol") == "HERE" or getattr(agent, "reached_goal", False)

    def reward(self, agent, prev_state, action, new_state, obs, step, max_steps):
        """
        Reward shaping adapted from your old LighthouseQLearningAgent.
        Now direction is one-hot in new_state, so we decode it.
        """
        if not hasattr(agent, "visited_positions"):
            agent.visited_positions = set()

        pos = (agent.x, agent.y)

        if pos in agent.visited_positions:
            r = -2.0
        else:
            r = -0.1
            agent.visited_positions.add(pos)

        # Decode direction from one-hot
        # new_state[0:9] is one-hot
        dir_idx = max(range(9), key=lambda i: new_state[i])
        direction_now = self.DIRS[dir_idx]

        DIRECAO_MAP = {
            "N": ["N", "NE", "NW"],
            "S": ["S", "SE", "SW"],
            "E": ["E", "NE", "SE"],
            "W": ["W", "NW", "SW"],
            "NE": ["NE", "N", "E"],
            "NW": ["NW", "N", "W"],
            "SE": ["SE", "S", "E"],
            "SW": ["SW", "S", "W"],
        }

        if prev_state is not None and direction_now != "HERE":
            if action in DIRECAO_MAP.get(direction_now, []):
                r += 1.0

        if direction_now == "HERE":
            r += 100.0 * (1 - step / max_steps)
            agent.visited_positions.clear()

        return r
