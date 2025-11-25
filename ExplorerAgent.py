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

        # Store recent positions (for potential debugging, no loop logic)
        self.last_positions = deque(maxlen=12)

        # Knowledge sharing
        self.known_obstacles = set()
        self.known_dead_ends = set()
        self.known_goals = set()
        self.visit_count = {}

        # BFS path (used when exact goal is communicated)
        self.planned_path = []

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
        """
        Receive simple info from other agents.
        """
        kind, data = mensagem.split(":", 1)
        data = eval(data)

        if kind == "goal":
            print(f"   [{self.name}] 📩 Received GOAL from {de_agente.name}: {data}")
            self.known_goals.add(data)
            self.planned_path = []  # reset

        elif kind == "obstacle":
            self.known_obstacles.add(data)

        elif kind == "dead_end":
            self.known_dead_ends.add(data)

    # --------------------------- MAIN DECISION --------------------------- #

    def age(self):
        """Choose next action based on mode."""
        self.last_positions.append((self.x, self.y))

        if self.reached_goal:
            return None

        mode = self.env.mode

        if mode == "farol":
            return self._farol_move()

        return self._maze_move()

    # --------------------------- FAROL MODE (LOOP-FREE) --------------------------- #

    def _farol_move(self):
        """
        Pure FAROL logic (NO LOOP DETECTION AT ALL):

            0) If goal communicated → BFS toward it
            1) Move directly according to direction sensor
            2) Try component moves
            3) Random escape among valid neighbors
            4) Forced escape
        """

        d = self.current_obs.get("direcao_farol")
        print(f"   [{self.name}] Observed direction: {d}")

        # 0) If goal communicated
        if self.known_goals:
            goal = next(iter(self.known_goals))
            print(f"   [{self.name}] Using communicated goal {goal}")
            return self._follow_or_plan_bfs(goal)

        # 1) If at lighthouse
        if d == "HERE":
            print(f"   [{self.name}] 🎯 Reached goal")
            self.reached_goal = True
            return None

        # Convert direction string (e.g. "NE") to dx, dy
        dx = -1 if "N" in d else (1 if "S" in d else 0)
        dy = -1 if "W" in d else (1 if "E" in d else 0)

        main = (self.x + dx, self.y + dy)

        if self.env.is_valid_position(*main):
            print(f"   [{self.name}] Rule 1 → direct move {main}")
            return main
        print(f"   [{self.name}] Rule 1 blocked: {main}")

        # Rule 2: component moves
        candidates = []
        if dx != 0:
            candidates.append((self.x + dx, self.y))
        if dy != 0:
            candidates.append((self.x, self.y + dy))

        for c in candidates:
            if self.env.is_valid_position(*c):
                print(f"   [{self.name}] Rule 2 → sliding move {c}")
                return c
            print(f"   [{self.name}] Rule 2 blocked: {c}")

        # Rule 3: random escape
        print(f"   [{self.name}] Rule 3 → random escape")
        neighbors = [
            (self.x - 1, self.y),
            (self.x + 1, self.y),
            (self.x, self.y - 1),
            (self.x, self.y + 1),
        ]
        valid = [n for n in neighbors if self.env.is_valid_position(*n)]

        if valid:
            choice = random.choice(valid)
            print(f"   [{self.name}] Random → {choice}")
            return choice

        print(f"   [{self.name}] ❌ No valid moves")
        return None

    # ---------- BFS when goal is known (communication) ---------- #

    def _follow_or_plan_bfs(self, goal):
        gx, gy = goal

        if (self.x, self.y) == goal:
            self.reached_goal = True
            return None

        # Need to replan?
        if not self.planned_path or (self.x, self.y) not in self.planned_path:
            print(f"   [{self.name}] 🔁 Planning BFS path to {goal}")
            self.planned_path = self._bfs_path((self.x, self.y), goal)

        # BFS failed (should be rare)
        if not self.planned_path:
            print(f"   [{self.name}] ⚠ BFS failed, fallback escape")
            return self._random_neighbor()

        idx = self.planned_path.index((self.x, self.y))

        if idx == len(self.planned_path) - 1:
            self.reached_goal = True
            return None

        next_step = self.planned_path[idx + 1]
        print(f"   [{self.name}] BFS step → {next_step}")
        return next_step

    def _bfs_path(self, start, goal):
        from collections import deque

        if start == goal:
            return [start]

        q = deque([start])
        visited = {start}
        parent = {}

        while q:
            x, y = q.popleft()
            for nx, ny in [(x-1,y),(x+1,y),(x,y-1),(x,y+1)]:
                if not self.env.is_valid_position(nx, ny):
                    continue
                if (nx, ny) in visited:
                    continue

                visited.add((nx, ny))
                parent[(nx, ny)] = (x, y)

                if (nx, ny) == goal:
                    path = [(nx, ny)]
                    while path[-1] != start:
                        path.append(parent[path[-1]])
                    return list(reversed(path))

                q.append((nx, ny))

        return None

    def _random_neighbor(self):
        neighbors = [
            (self.x - 1, self.y),
            (self.x + 1, self.y),
            (self.x, self.y - 1),
            (self.x, self.y + 1),
        ]
        valid = [n for n in neighbors if self.env.is_valid_position(*n)]
        return random.choice(valid) if valid else None

    # --------------------------- MAZE MODE (unchanged) --------------------------- #

    def _maze_move(self):
        self._mark_visit((self.x, self.y))

        # If we know the goal
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

        # Explore
        for d in self._shuffled_dirs():
            nx, ny = self._apply(d)
            if not self._is_blocked(nx, ny):
                return (nx, ny)

        return None

    # --------------------------- MAZE HELPERS --------------------------- #

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
