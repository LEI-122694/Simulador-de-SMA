import random
from collections import deque

class ExplorerAgent:
    """
    Unified Agent for both:
      - FAROL mode (direction sensor only)
      - MAZE mode  (blocked neighbors, goal detection, communication)
    """

    def __init__(self, name, env, start_pos):
        self.name = name
        self.env = env
        self.x, self.y = start_pos
        self.env.add_agent(self)

        # State
        self.reached_goal = False
        self.current_obs = None

        # Loop memory for FAROL
        self.last_positions = deque(maxlen=8)  # recent positions

        # Shared memory
        self.known_obstacles = set()
        self.known_dead_ends = set()
        self.known_goals = set()
        self.visit_count = {}

    # --------------------------- REQUIRED API --------------------------- #

    @classmethod
    def cria(cls, ficheiro):
        print("[AGENTE] carregar parâmetros (stub)")
        return None

    def observacao(self, obs):
        self.current_obs = obs

    def avaliacaoEstadoAtual(self, recompensa):
        pass

    def instala(self, sensor):
        pass

    def comunica(self, mensagem, de_agente):
        """Receive knowledge from other agents."""
        kind, data = mensagem.split(":", 1)
        data = eval(data)

        if kind == "goal":
            self.known_goals.add(data)
        elif kind == "obstacle":
            self.known_obstacles.add(data)
        elif kind == "dead_end":
            self.known_dead_ends.add(data)

    # --------------------------- MAIN DECISION --------------------------- #

    def age(self):
        """Choose next action depending on mode."""

        # Track position for loop detection (FAROL)
        self.last_positions.append((self.x, self.y))

        if self.reached_goal:
            return None

        mode = self.env.mode

        if mode == "farol":
            return self._farol_move()

        return self._maze_move()

    # --------------------------- FAROL MODE --------------------------- #

    def _farol_move(self):
        d = self.current_obs.get("direcao_farol")

        print(f"   [{self.name}] Observed direction: {d}")

        if d == "HERE":
            print(f"   [{self.name}] 🎯 Reached goal")
            self.reached_goal = True
            return None

        # Desired main movement toward lighthouse
        dx = -1 if "N" in d else (1 if "S" in d else 0)
        dy = -1 if "W" in d else (1 if "E" in d else 0)

        main = (self.x + dx, self.y + dy)

        # ------------------------------
        # RULE 1: Direct move (avoid loops)
        # ------------------------------
        if self.env.is_valid_position(*main) and not self._is_loop(*main):
            print(f"   [{self.name}] Rule 1 → direct toward lighthouse {main}")
            return main
        else:
            print(f"   [{self.name}] Rule 1 BLOCKED or LOOP {main}")

        # ------------------------------
        # RULE 2: Try components (N only or E only etc.)
        # ------------------------------
        components = []
        if dx != 0:
            components.append((self.x + dx, self.y))
        if dy != 0:
            components.append((self.x, self.y + dy))

        for nx, ny in components:
            if self.env.is_valid_position(nx, ny) and not self._is_loop(nx, ny):
                print(f"   [{self.name}] Rule 2 → sliding move {(nx, ny)}")
                return (nx, ny)
            else:
                print(f"   [{self.name}] Rule 2 blocked/loop at {(nx, ny)}")

        # ------------------------------
        # RULE 3: Anti-loop random escape
        # ------------------------------
        print(f"   [{self.name}] Rule 3 → random escape")

        moves = [
            (self.x - 1, self.y),  # up
            (self.x + 1, self.y),  # down
            (self.x, self.y - 1),  # left
            (self.x, self.y + 1),  # right
        ]
        random.shuffle(moves)

        for nx, ny in moves:
            if self.env.is_valid_position(nx, ny) and not self._is_loop(nx, ny):
                print(f"   [{self.name}] Random escape → {(nx, ny)}")
                return (nx, ny)

        # ------------------------------
        # RULE 4: Last resort → move to any valid tile (even if loop)
        # ------------------------------
        print(f"   [{self.name}] Rule 4 → forced escape (all moves looping)")
        valid_moves = [(nx, ny) for nx, ny in moves if self.env.is_valid_position(nx, ny)]

        if valid_moves:
            best = valid_moves[0]
            print(f"   [{self.name}] Forced escape → {best}")
            return best

        print(f"   [{self.name}] ❌ No possible moves")
        return None

    # --------------------------- MAZE MODE --------------------------- #

    def _maze_move(self):
        self._mark_visit((self.x, self.y))

        # 1) If KNOW goal → greedy move
        target = None
        if self.known_goals:
            target = next(iter(self.known_goals))
        else:
            goals = self.current_obs.get("objetivos", [])
            if goals:
                target = goals[0]

        if target:
            nx, ny = self._greedy_step(target)
            if not self._is_blocked(nx, ny):
                return (nx, ny)

        # 2) Try unexplored directions
        for d in self._shuffled_dirs():
            nx, ny = self._apply(d)
            if not self._is_blocked(nx, ny):
                return (nx, ny)

        return None

    # --------------------------- MEMORY / HELPERS --------------------------- #

    def _is_loop(self, x, y):
        """Returns True if moving to (x,y) causes oscillation/backtracking."""
        return (x, y) in list(self.last_positions)[-4:]

    def _broadcast(self, msg):
        for other in self.env.agents:
            if other is not self:
                other.comunica(msg, self)

    def _parse(self, msg):
        return eval(msg.split(":", 1)[1])

    def _mark_visit(self, pos):
        self.visit_count[pos] = self.visit_count.get(pos, 0) + 1
        if self.visit_count[pos] > 3:
            self.known_dead_ends.add(pos)
            self._broadcast(f"dead_end:{pos}")

    def _is_blocked(self, x, y):
        if not self.env.is_valid_position(x, y):
            self.known_obstacles.add((x, y))
            return True

        if (x, y) in self.known_obstacles:
            return True
        if (x, y) in self.known_dead_ends:
            return True

        blocked = self.current_obs.get("vizinhos_bloqueados", set())
        if (x, y) in blocked:
            self.known_obstacles.add((x, y))
            return True

        return False

    def _greedy_step(self, goal):
        gx, gy = goal
        dx = gx - self.x
        dy = gy - self.y

        if abs(dx) > abs(dy):
            return (self.x + (1 if dx > 0 else -1), self.y)
        else:
            return (self.x, self.y + (1 if dy > 0 else -1))

    def _shuffled_dirs(self):
        dirs = ["up", "down", "left", "right"]
        random.shuffle(dirs)
        return dirs

    def _apply(self, d):
        if d == "up": return (self.x - 1, self.y)
        if d == "down": return (self.x + 1, self.y)
        if d == "left": return (self.x, self.y - 1)
        if d == "right": return (self.x, self.y + 1)
