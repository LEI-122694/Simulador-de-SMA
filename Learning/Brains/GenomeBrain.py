# GenomeBrain.py

import random
import math

class GenomeBrain:
    """
    Recurrent neural controller evolved by fitness.
    """

    def __init__(self, genome=None, inputs=12, hidden=6, outputs=4):
        self.INPUTS = inputs
        self.HIDDEN = hidden
        self.OUTPUTS = outputs

        self.genome = genome if genome is not None else self.random_genome()
        self._unpack()
        self.hidden = [0.0] * self.HIDDEN

    # --------------------------------------------------
    def genome_size(self):
        return (
            self.INPUTS * self.HIDDEN +
            self.HIDDEN * self.HIDDEN +
            self.HIDDEN * self.OUTPUTS
        )

    def random_genome(self):
        return [random.uniform(-1, 1) for _ in range(self.genome_size())]

    # --------------------------------------------------
    def _unpack(self):
        g = self.genome
        p = 0

        self.W_in = []
        for h in range(self.HIDDEN):
            self.W_in.append(g[p:p+self.INPUTS])
            p += self.INPUTS

        self.W_rec = []
        for h in range(self.HIDDEN):
            self.W_rec.append(g[p:p+self.HIDDEN])
            p += self.HIDDEN

        self.W_out = []
        for o in range(self.OUTPUTS):
            self.W_out.append(g[p:p+self.HIDDEN])
            p += self.HIDDEN

    # --------------------------------------------------
    def select_action(self, state, valid_actions, mode=None):
        out = self.forward(state)
        idx = max(range(len(valid_actions)), key=lambda i: out[i])
        return valid_actions[idx]

    def forward(self, inp):
        new_hidden = []
        for h in range(self.HIDDEN):
            s = sum(inp[i] * self.W_in[h][i] for i in range(self.INPUTS))
            s += sum(self.hidden[j] * self.W_rec[h][j] for j in range(self.HIDDEN))
            new_hidden.append(math.tanh(s))

        self.hidden = new_hidden

        return [
            sum(self.hidden[h] * self.W_out[o][h] for h in range(self.HIDDEN))
            for o in range(self.OUTPUTS)
        ]

    # --------------------------------------------------
    def update(self, *args, **kwargs):
        pass
