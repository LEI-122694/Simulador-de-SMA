import math
import random
from Agents.Agent import Agent


class MazeLearningAgent(Agent):
    """
    Neural-network controlled maze agent for Evolution / Novelty Search.

    Inputs to the NN (12 total):
      0: wall_up
      1: wall_down
      2: wall_left
      3: wall_right
      4: goal_up
      5: goal_down
      6: goal_left
      7: goal_right
      8-11: one-hot of last move [up, down, left, right]

    Outputs (4 neurons):
      index 0 -> up
      index 1 -> down
      index 2 -> left
      index 3 -> right

    The agent does NOT use coordinates for planning; it only uses the
    local sensors provided by the World in current_obs and last_move.
    """

    # Network architecture
    N_INPUTS = 12
    N_HIDDEN = 10
    N_OUTPUTS = 4

    # Total parameters = W1 + b1 + W2 + b2
    GENOTYPE_SIZE = (
        N_INPUTS * N_HIDDEN +     # input -> hidden weights
        N_HIDDEN +                # hidden biases
        N_HIDDEN * N_OUTPUTS +    # hidden -> output weights
        N_OUTPUTS                 # output biases
    )

    ACTIONS = ["up", "down", "left", "right"]
    DIR_TO_IDX = {"up": 0, "down": 1, "left": 2, "right": 3}

    # ------------------------------------------------------------
    # REQUIRED: constructor
    # ------------------------------------------------------------
    def __init__(self, name, env, start_pos, weights=None):
        """
        weights: flat list of floats for NN parameters.
                 If None, random weights are initialized.
        """
        super().__init__(name, env, start_pos)

        self.reached_goal = False
        self.current_obs = None
        self.last_dir = None       # last move direction (str)
        self.behavior = []         # sequence of moves (for logging / novelty if needed)

        if weights is None:
            # random from small range
            self.weights = self._random_genotype()
        else:
            self.weights = list(weights)

        # unpack flat weights into matrices
        self._unpack_network()

    # ------------------------------------------------------------
    # REQUIRED BY ABSTRACT BASE CLASS: comunica
    # ------------------------------------------------------------
    def comunica(self, mensagem, de_agente):
        # No communication used for evolutionary Maze agent
        pass

    # ------------------------------------------------------------
    # REQUIRED BY ABSTRACT BASE CLASS: observacao
    # ------------------------------------------------------------
    def observacao(self, obs):
        self.current_obs = obs
        pos = obs.get("posicao")
        goals = obs.get("goals", [])

        if goals and pos == goals[0]:
            self.reached_goal = True

    # ------------------------------------------------------------
    # REQUIRED BY ABSTRACT BASE CLASS: age
    # ------------------------------------------------------------
    def age(self):
        """
        Compute NN inputs from local sensors, run forward pass,
        choose the best action, and return the next (x, y).
        """
        if self.reached_goal or self.current_obs is None:
            return None

        inputs = self._build_inputs()
        outputs = self._forward(inputs)

        # choose best action index
        best_idx = max(range(self.N_OUTPUTS), key=lambda i: outputs[i])
        direction = self.ACTIONS[best_idx]

        # track behavior and last move
        self.behavior.append(direction)
        self.last_dir = direction

        return self._apply(direction)

    # ------------------------------------------------------------
    # CLASS HELPERS FOR EVOLUTION
    # ------------------------------------------------------------
    @classmethod
    def genotype_size(cls):
        return cls.GENOTYPE_SIZE

    @classmethod
    def _random_genotype(cls):
        """Create random flat list of all NN parameters."""
        size = cls.GENOTYPE_SIZE
        scale = 0.5
        return [random.uniform(-scale, scale) for _ in range(size)]

    # ------------------------------------------------------------
    # NETWORK UNPACKING
    # ------------------------------------------------------------
    def _unpack_network(self):
        """
        Unpack flat self.weights into W1, b1, W2, b2.
        Shapes:
          W1: [N_HIDDEN][N_INPUTS]
          b1: [N_HIDDEN]
          W2: [N_OUTPUTS][N_HIDDEN]
          b2: [N_OUTPUTS]
        """
        g = self.weights
        p = 0

        # W1
        self.W1 = []
        for h in range(self.N_HIDDEN):
            row = g[p:p + self.N_INPUTS]
            self.W1.append(row)
            p += self.N_INPUTS

        # b1
        self.b1 = g[p:p + self.N_HIDDEN]
        p += self.N_HIDDEN

        # W2
        self.W2 = []
        for o in range(self.N_OUTPUTS):
            row = g[p:p + self.N_HIDDEN]
            self.W2.append(row)
            p += self.N_HIDDEN

        # b2
        self.b2 = g[p:p + self.N_OUTPUTS]
        # p += self.N_OUTPUTS  # not really needed further

    # ------------------------------------------------------------
    # BUILD NN INPUT VECTOR (length 12)
    # ------------------------------------------------------------
    def _build_inputs(self):
        """
        Returns 12 floats:

          [ wall_up, wall_down, wall_left, wall_right,
            goal_up, goal_down, goal_left, goal_right,
            last_up, last_down, last_left, last_right ]
        """
        x, y = self.current_obs["posicao"]
        goals = self.current_obs.get("goals", [])

        # wall sensors (1 if blocked, 0 otherwise)
        wall_up    = 1.0 if self.env.is_blocked(x - 1, y) else 0.0
        wall_down  = 1.0 if self.env.is_blocked(x + 1, y) else 0.0
        wall_left  = 1.0 if self.env.is_blocked(x, y - 1) else 0.0
        wall_right = 1.0 if self.env.is_blocked(x, y + 1) else 0.0

        # goal adjacency sensors
        goal_up = goal_down = goal_left = goal_right = 0.0
        if goals:
            gx, gy = goals[0]
            if (x - 1, y) == (gx, gy):
                goal_up = 1.0
            if (x + 1, y) == (gx, gy):
                goal_down = 1.0
            if (x, y - 1) == (gx, gy):
                goal_left = 1.0
            if (x, y + 1) == (gx, gy):
                goal_right = 1.0

        # one-hot last movement
        last_up = last_down = last_left = last_right = 0.0
        if self.last_dir is not None:
            idx = self.DIR_TO_IDX[self.last_dir]
            if idx == 0:
                last_up = 1.0
            elif idx == 1:
                last_down = 1.0
            elif idx == 2:
                last_left = 1.0
            elif idx == 3:
                last_right = 1.0

        return [
            wall_up, wall_down, wall_left, wall_right,
            goal_up, goal_down, goal_left, goal_right,
            last_up, last_down, last_left, last_right
        ]

    # ------------------------------------------------------------
    # NN FORWARD PASS
    # ------------------------------------------------------------
    def _forward(self, inputs):
        # hidden layer
        hidden = []
        for h in range(self.N_HIDDEN):
            s = self.b1[h]
            row = self.W1[h]
            for i in range(self.N_INPUTS):
                s += row[i] * inputs[i]
            hidden.append(math.tanh(s))

        # output layer (linear)
        outputs = []
        for o in range(self.N_OUTPUTS):
            s = self.b2[o]
            row = self.W2[o]
            for h in range(self.N_HIDDEN):
                s += row[h] * hidden[h]
            outputs.append(s)

        return outputs

    # ------------------------------------------------------------
    # APPLY DIRECTION
    # ------------------------------------------------------------
    def _apply(self, direction):
        if direction == "up":
            return (self.x - 1, self.y)
        if direction == "down":
            return (self.x + 1, self.y)
        if direction == "left":
            return (self.x, self.y - 1)
        if direction == "right":
            return (self.x, self.y + 1)
        return (self.x, self.y)
