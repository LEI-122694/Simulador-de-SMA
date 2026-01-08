# Learning/adapters/FarolAdapter.py
from Learning.adapters.TaskAdapter import TaskAdapter

class FarolAdapter(TaskAdapter):

    ACTIONS = ["N","S","E","W","NE","NW","SE","SW"]

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

    def build_state(self, agent, obs, env):
        direction = obs.get("direcao_farol", "HERE")

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

        return (
            direction,
            blocked["N"], blocked["S"], blocked["E"], blocked["W"],
            blocked["NE"], blocked["NW"], blocked["SE"], blocked["SW"]
        )

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

    # Reward shaping (copied from your calcula_recompensa)
    def reward(self, agent, prev_state, action, new_state, obs, step, max_steps):

        if not hasattr(agent, "visited_positions"):
            agent.visited_positions = set()

        pos = (agent.x, agent.y)

        if pos in agent.visited_positions:
            r = -2.0
        else:
            r = -0.1
            agent.visited_positions.add(pos)

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

        # new_state[0] is direction
        if prev_state is not None:
            direction_now = new_state[0]
            if action in DIRECAO_MAP.get(direction_now, []):
                r += 1.0

        if new_state[0] == "HERE":
            r += 100.0 * (1 - step / max_steps)
            agent.visited_positions.clear()

        return r
