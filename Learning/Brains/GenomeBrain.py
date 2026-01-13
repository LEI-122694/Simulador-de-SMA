# Learning/Brains/GenomeBrain.py
import random
import math
from typing import List, Optional, Sequence


class GenomeBrain:
    """
    Recurrent neural controller (RNN) evolved by fitness/novelty.

    Key properties for your architecture:
    - Environment-agnostic
    - Works with any adapter that provides:
        * adapter.ACTIONS (fixed action order)
        * adapter.observation_size()
        * adapter.action_size()
    - Correctly chooses the best action among valid_actions
      while keeping outputs aligned with adapter.ACTIONS
    """

    def __init__(
        self,
        genome: Optional[List[float]] = None,
        inputs: int = 12,
        hidden: int = 6,
        outputs: int = 4,
        action_order: Optional[Sequence[str]] = None,
    ):
        self.INPUTS = int(inputs)
        self.HIDDEN = int(hidden)
        self.OUTPUTS = int(outputs)

        # Fixed action order (should match adapter.ACTIONS)
        self.action_order = list(action_order) if action_order is not None else None

        self.genome = genome if genome is not None else self.random_genome(
            self.INPUTS, self.HIDDEN, self.OUTPUTS
        )

        # Validate genome length
        expected = self.genome_size(self.INPUTS, self.HIDDEN, self.OUTPUTS)
        if len(self.genome) != expected:
            raise ValueError(
                f"Genome length {len(self.genome)} != expected {expected} "
                f"(inputs={self.INPUTS}, hidden={self.HIDDEN}, outputs={self.OUTPUTS})"
            )

        self._unpack()
        self.hidden_state = [0.0] * self.HIDDEN

    # --------------------------------------------------
    @staticmethod
    def genome_size(inputs: int, hidden: int = 6, outputs: int = 4) -> int:
        return inputs * hidden + hidden * hidden + hidden * outputs

    @staticmethod
    def random_genome(inputs: int, hidden: int = 6, outputs: int = 4) -> List[float]:
        n = GenomeBrain.genome_size(inputs, hidden, outputs)
        return [random.uniform(-1, 1) for _ in range(n)]

    # --------------------------------------------------
    def reset(self):
        """Reset recurrent memory between episodes."""
        self.hidden_state = [0.0] * self.HIDDEN

    # --------------------------------------------------
    def _unpack(self):
        g = self.genome
        p = 0

        # Input -> Hidden
        self.W_in = []
        for _ in range(self.HIDDEN):
            self.W_in.append(g[p:p + self.INPUTS])
            p += self.INPUTS

        # Hidden -> Hidden (recurrent)
        self.W_rec = []
        for _ in range(self.HIDDEN):
            self.W_rec.append(g[p:p + self.HIDDEN])
            p += self.HIDDEN

        # Hidden -> Output
        self.W_out = []
        for _ in range(self.OUTPUTS):
            self.W_out.append(g[p:p + self.HIDDEN])
            p += self.HIDDEN

    # --------------------------------------------------
    def forward(self, inp: Sequence[float]) -> List[float]:
        if len(inp) != self.INPUTS:
            raise ValueError(f"Input length {len(inp)} != expected {self.INPUTS}")

        # Hidden update
        new_hidden = []
        for h in range(self.HIDDEN):
            s = 0.0
            for i in range(self.INPUTS):
                s += float(inp[i]) * self.W_in[h][i]
            for j in range(self.HIDDEN):
                s += self.hidden_state[j] * self.W_rec[h][j]
            new_hidden.append(math.tanh(s))

        self.hidden_state = new_hidden

        # Output
        out = []
        for o in range(self.OUTPUTS):
            s = 0.0
            for h in range(self.HIDDEN):
                s += self.hidden_state[h] * self.W_out[o][h]
            out.append(s)

        return out

    # --------------------------------------------------
    def select_action(self, state, valid_actions, mode=None):
        """
        Choose best action among valid_actions.

        IMPORTANT: output neurons are aligned with self.action_order (adapter.ACTIONS),
        NOT with valid_actions order.
        """
        scores = self.forward(state)

        if self.action_order is None:
            # Fallback: assume outputs correspond to valid_actions order (only safe if always full set).
            best_i = max(range(len(valid_actions)), key=lambda i: scores[i])
            return valid_actions[best_i]

        # Build score dict for fixed action order
        if len(self.action_order) != self.OUTPUTS:
            raise ValueError(
                f"action_order length {len(self.action_order)} != OUTPUTS {self.OUTPUTS}"
            )

        score_map = {a: scores[i] for i, a in enumerate(self.action_order)}

        # Pick best among valid actions
        best_action = max(valid_actions, key=lambda a: score_map.get(a, float("-inf")))
        return best_action

    # --------------------------------------------------
    def update(self, *args, **kwargs):
        # Evolutionary brains don't learn online.
        pass
