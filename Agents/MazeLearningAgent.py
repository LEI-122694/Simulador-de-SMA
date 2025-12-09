import math
import random
from Agents.Agent import Agent

class MazeLearningAgent(Agent):
    """
    LEGAL Maze controller:
    - No coordinates
    - No distance-to-goal
    - No knowledge of direction to goal
    - ONLY wall sensors, goal adjacency, last action
    - WITH small internal recurrent memory (allowed)
    """

    INPUTS = 12            # wall(4) + goal-adj(4) + last-action(4)
    HIDDEN = 6             # internal memory units
    OUTPUTS = 4            # up/down/left/right

    ACTIONS = ["up", "down", "left", "right"]
    ACTION_TO_IDX = {"up": 0, "down": 1, "left": 2, "right": 3}

    @classmethod
    def genome_size(cls):
        return (
            cls.INPUTS * cls.HIDDEN +      # W_input
            cls.HIDDEN * cls.HIDDEN +      # W_recurrent
            cls.HIDDEN * cls.OUTPUTS       # W_output
        )

    def __init__(self, name, env, start_pos, genome=None):
        super().__init__(name, env, start_pos)

        self.genome = genome if genome is not None else self.random_genome()
        self._unpack()

        self.hidden = [0.0] * self.HIDDEN
        self.last_action = None
        self.current_obs = None
        self.reached_goal = False

    # ------------------------------------------------------------
    def random_genome(self):
        return [random.uniform(-1, 1) for _ in range(self.genome_size())]

    # ------------------------------------------------------------
    def _unpack(self):
        g = self.genome
        p = 0

        # Input → Hidden
        self.W_in = []
        for h in range(self.HIDDEN):
            self.W_in.append(g[p:p+self.INPUTS])
            p += self.INPUTS

        # Hidden → Hidden (recurrent)
        self.W_rec = []
        for h in range(self.HIDDEN):
            self.W_rec.append(g[p:p+self.HIDDEN])
            p += self.HIDDEN

        # Hidden → Output
        self.W_out = []
        for o in range(self.OUTPUTS):
            self.W_out.append(g[p:p+self.HIDDEN])
            p += self.HIDDEN

    def comunica(self, m, a):
        pass

    # ------------------------------------------------------------
    def observacao(self, obs):
        self.current_obs = obs
        if obs["posicao"] in obs["goals"]:
            self.reached_goal = True

    # ------------------------------------------------------------
    def _build_inputs(self):
        x, y = self.current_obs["posicao"]
        gx, gy = self.current_obs["goals"][0]

        # Walls
        wU = 1.0 if self.env.is_blocked(x - 1, y) else 0.0
        wD = 1.0 if self.env.is_blocked(x + 1, y) else 0.0
        wL = 1.0 if self.env.is_blocked(x, y - 1) else 0.0
        wR = 1.0 if self.env.is_blocked(x, y + 1) else 0.0

        # Goal adjacency
        gU = 1.0 if (x - 1, y) == (gx, gy) else 0.0
        gD = 1.0 if (x + 1, y) == (gx, gy) else 0.0
        gL = 1.0 if (x, y - 1) == (gx, gy) else 0.0
        gR = 1.0 if (x, y + 1) == (gx, gy) else 0.0

        # Last action one-hot
        laU = laD = laL = laR = 0.0
        if self.last_action:
            idx = self.ACTION_TO_IDX[self.last_action]
            if idx == 0: laU = 1.0
            if idx == 1: laD = 1.0
            if idx == 2: laL = 1.0
            if idx == 3: laR = 1.0

        return [wU, wD, wL, wR, gU, gD, gL, gR, laU, laD, laL, laR]

    # ------------------------------------------------------------
    def forward(self, inp):
        # Compute new hidden state
        new_hidden = []
        for h in range(self.HIDDEN):
            s = 0

            # From inputs
            for i in range(self.INPUTS):
                s += inp[i] * self.W_in[h][i]

            # From previous hidden (recurrence)
            for j in range(self.HIDDEN):
                s += self.hidden[j] * self.W_rec[h][j]

            new_hidden.append(math.tanh(s))

        self.hidden = new_hidden

        # Compute outputs
        out = []
        for o in range(self.OUTPUTS):
            s = sum(self.hidden[h] * self.W_out[o][h] for h in range(self.HIDDEN))
            out.append(s)

        return out

    # ------------------------------------------------------------
    def age(self):
        if self.reached_goal or self.current_obs is None:
            return None

        inp = self._build_inputs()
        out = self.forward(inp)

        best = out.index(max(out))
        action = self.ACTIONS[best]
        self.last_action = action

        if action == "up":    return (self.x - 1, self.y)
        if action == "down":  return (self.x + 1, self.y)
        if action == "left":  return (self.x, self.y - 1)
        if action == "right": return (self.x, self.y + 1)
        return None
