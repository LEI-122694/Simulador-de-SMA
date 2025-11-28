# LighthouseAgent.py
import random
from Agents.Agent import Agent

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
        Estratégia FAROL com debug:
        1) Mover direto segundo direção do sensor
        2) Tentar componentes vert/horiz
        3) Aleatório entre vizinhos válidos (8 direções)
        4) Nenhum movimento possível
        """

        test_mode = (self.mode == "test")

        d = self.current_obs.get("direcao_farol")
        print(f"   [{self.name}] Observed direction: {d}")

        # Se no farol
        if d == "HERE":
            print(f"   [{self.name}] 🎯 Reached lighthouse")
            self.reached_goal = True
            return None

        # dx, dy segundo sensor
        dx = -1 if "N" in d else (1 if "S" in d else 0)
        dy = -1 if "W" in d else (1 if "E" in d else 0)

        main = (self.x + dx, self.y + dy)

        # -------------------------------------------------
        # Regra 1 — mover direto
        # -------------------------------------------------
        if self.env.is_valid_position(*main):
            print(f"   [{self.name}] Regra 1 → mover direto {main}")
            return main
        else:
            print(f"   [{self.name}] Regra 1 BLOQUEADO {main}")

        # -------------------------------------------------
        # Regra 2 — componentes verticais/horizontais
        # -------------------------------------------------
        components = []
        if dx != 0:
            components.append((self.x + dx, self.y))
        if dy != 0:
            components.append((self.x, self.y + dy))

        for c in components:
            if self.env.is_valid_position(*c):
                print(f"   [{self.name}] Regra 2 → movimento componente {c}")
                return c
            else:
                print(f"   [{self.name}] Regra 2 BLOQUEADO {c}")

        # -------------------------------------------------
        # Regra 3 — Movimento aleatório (8 direções)
        # -------------------------------------------------
        print(f"   [{self.name}] Regra 3 → RANDOM")

        neighbors = [
            (self.x - 1, self.y), (self.x + 1, self.y),
            (self.x, self.y - 1), (self.x, self.y + 1),
            (self.x - 1, self.y - 1), (self.x - 1, self.y + 1),
            (self.x + 1, self.y - 1), (self.x + 1, self.y + 1)
        ]
        valid = [n for n in neighbors if self.env.is_valid_position(*n)]

        if valid:
            if test_mode:
                choice = valid[0]  # determinista
            else:
                choice = random.choice(valid)
            print(f"   [{self.name}] RANDOM → {choice}")
            return choice

        # -------------------------------------------------
        # Sem movimentos válidos
        # -------------------------------------------------
        print(f"   [{self.name}] ❌ Nenhum movimento possível")
        return None

