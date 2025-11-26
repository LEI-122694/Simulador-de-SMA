# LighthouseAgent.py
import random
from Agent import Agent

class LighthouseAgent(Agent):
    """
    Agente para o modo FAROL.
    Move-se em direção ao farol se visível;
    caso contrário explora aleatoriamente incluindo diagonais.
    """

    def __init__(self, name, env, start_pos):
        super().__init__(name, env, start_pos)
        self.reached_goal = False
        self.current_obs = None
        self.goal_known = None  # posição do farol quando detectado

    def comunica(self, mensagem, de_agente):
        # não faz nada, apenas para satisfazer a classe abstrata
        pass

    # ------------------------------------------------------------
    # OBSERVAÇÃO
    # ------------------------------------------------------------
    def observacao(self, obs):
        self.current_obs = obs

        goals = obs.get("goals")
        if goals:
            self.goal_known = goals[0]

            # check if already at the goal
            if (self.x, self.y) == self.goal_known:
                self.reached_goal = True

    # ------------------------------------------------------------
    # DECISÃO (AGE)
    # ------------------------------------------------------------
    def age(self):
        if self.reached_goal:
            return None
        return self._farol_move()

    # ------------------------------------------------------------
    # MOVIMENTO FAROL
    # ------------------------------------------------------------
    def _farol_move(self):
        """
        Estratégia FAROL simplificada:
        1) Mover direto segundo direção do sensor
        2) Tentar componentes vert/horiz
        3) Aleatório entre vizinhos válidos (8 direções)
        4) Nenhum movimento possível
        """
        d = self.current_obs.get("direcao_farol")

        # Se no farol
        if d == "HERE":
            self.reached_goal = True
            return None

        # dx, dy segundo sensor
        dx = -1 if "N" in d else (1 if "S" in d else 0)
        dy = -1 if "W" in d else (1 if "E" in d else 0)
        main = (self.x + dx, self.y + dy)

        # Regra 1 — mover direto
        if self.env.is_valid_position(*main):
            return main

        # Regra 2 — componentes vert/horiz
        components = []
        if dx != 0:
            components.append((self.x + dx, self.y))
        if dy != 0:
            components.append((self.x, self.y + dy))
        for c in components:
            if self.env.is_valid_position(*c):
                return c

        # Regra 3 — aleatório entre vizinhos válidos (8 direções)
        neighbors = [
            (self.x - 1, self.y), (self.x + 1, self.y),
            (self.x, self.y - 1), (self.x, self.y + 1),
            (self.x - 1, self.y - 1), (self.x - 1, self.y + 1),
            (self.x + 1, self.y - 1), (self.x + 1, self.y + 1)
        ]
        valid = [n for n in neighbors if self.env.is_valid_position(*n)]
        if valid:
            return random.choice(valid)

        # Regra 4 — sem movimento possível
        return None
