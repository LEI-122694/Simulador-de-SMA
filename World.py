import random

class World:
    """
    Two modes:
      - 'farol' : agents only get a compass direction toward the lighthouse.
      - 'maze'  : agents get blocked neighbors and full goal positions.
    """

    def __init__(self, height, width, goals=None, obstacles=None, mode="maze"):
        self.height = height
        self.width = width
        self.goals = set(goals or [])
        self.obstacles = set(obstacles or [])
        self.mode = mode
        self.agents = []
        self.step_count = 0

    def is_blocked(self, x, y):
        """
        True se for obstáculo ou estiver fora dos limites.
        """
        if x < 0 or y < 0 or x >= self.height or y >= self.width:
            return True
        return (x, y) in self.obstacles

    # ------------------------------------------------------------
    # OBSERVATION
    # ------------------------------------------------------------
    def observacaoPara(self, agente):
        """Return observation depending on mode."""

        # ---------------- FAROL MODE ----------------
        if self.mode == "farol":
            gx, gy = next(iter(self.goals))
            ax, ay = agente.x, agente.y

            if (ax, ay) == (gx, gy):
                return {"direcao_farol": "HERE"}

            dx = gx - ax
            dy = gy - ay

            vertical = "N" if dx < 0 else ("S" if dx > 0 else "")
            horizontal = "W" if dy < 0 else ("E" if dy > 0 else "")

            return {"direcao_farol": vertical + horizontal}

        # ---------------- MAZE MODE ----------------
        blocked = set()
        for nx, ny in self._neighbors4(agente.x, agente.y):
            if not self.is_valid_position(nx, ny):
                blocked.add((nx, ny))

        return {
            "posicao": (agente.x, agente.y),
            "neighbors": {pos: "obstacle" for pos in blocked},
            "goals": list(self.goals)
        }

    # ------------------------------------------------------------
    def atualizacao(self):
        self.step_count += 1

    # ------------------------------------------------------------
    def agir(self, accao, agente):
        if accao is None:
            return

        nx, ny = accao

        if self.is_valid_position(nx, ny):
            agente.x, agente.y = nx, ny

            if (nx, ny) in self.goals:
                agente.reached_goal = True

    # ------------------------------------------------------------
    def add_agent(self, agent):
        self.agents.append(agent)

    # ------------------------------------------------------------
    def is_valid_position(self, x, y):
        return (
            0 <= x < self.height and
            0 <= y < self.width and
            (x, y) not in self.obstacles
        )

    def _neighbors4(self, x, y):
        return [
            (x-1, y),
            (x+1, y),
            (x, y-1),
            (x, y+1)
        ]

    # ------------------------------------------------------------
    # DISPLAY (CONSOLE)
    # ------------------------------------------------------------
    def display(self):
        RED = "\033[91m"
        GREEN = "\033[92m"
        YELLOW = "\033[93m"
        CYAN = "\033[96m"
        RESET = "\033[0m"

        top_border = CYAN + "+" + "-" * (self.width * 2) + "+" + RESET
        print(top_border)

        for x in range(self.height):
            row_str = CYAN + "|" + RESET
            for y in range(self.width):

                # Obstacles
                if (x, y) in self.obstacles:
                    row_str += RED + "# " + RESET
                    continue

                # Goals
                if (x, y) in self.goals:
                    row_str += YELLOW + "* " + RESET
                    continue

                # Agents
                agent_here = next((a for a in self.agents if (a.x, a.y) == (x, y)), None)
                if agent_here:
                    row_str += GREEN + agent_here.name[0].upper() + " " + RESET
                else:
                    row_str += ". "

            row_str += CYAN + "|" + RESET
            print(row_str)

        print(CYAN + "+" + "-" * (self.width * 2) + "+" + RESET)
        print()
