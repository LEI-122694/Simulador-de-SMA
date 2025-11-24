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
        self.last_positions = deque(maxlen=12)  # recent positions

        # Shared memory (used more in MAZE, but also for FAROL communication)
        self.known_obstacles = set()
        self.known_dead_ends = set()
        self.known_goals = set()
        self.visit_count = {}

        # Path planning memory (used when goal is known in FAROL/MAZE)
        self.planned_path = []  # list of (x,y) from current pos to goal

    # --------------------------- REQUIRED API --------------------------- #

    @classmethod
    def cria(cls, ficheiro):
        print("[AGENTE] carregar parâmetros (stub)")
        return None

    def observacao(self, obs):
        self.current_obs = obs

    def avaliacaoEstadoAtual(self, recompensa):
        # placeholder for learning mode (future)
        pass

    def instala(self, sensor):
        # not used for now
        pass

    def comunica(self, mensagem, de_agente):
        """
        Receive knowledge from other agents.
        mensagem format: "goal:(x,y)" or "obstacle:(x,y)" or "dead_end:(x,y)"
        """
        kind, data = mensagem.split(":", 1)
        data = eval(data)  # simple tuple parsing

        if kind == "goal":
            print(f"   [{self.name}] 📩 Received GOAL from {de_agente.name}: {data}")
            self.known_goals.add(data)
            # Clear any old path so we re-plan from the new info
            self.planned_path = []
        elif kind == "obstacle":
            print(f"   [{self.name}] 📩 Received OBSTACLE from {de_agente.name}: {data}")
            self.known_obstacles.add(data)
        elif kind == "dead_end":
            print(f"   [{self.name}] 📩 Received DEAD_END from {de_agente.name}: {data}")
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

        # Maze mode
        return self._maze_move()

    # --------------------------- FAROL MODE --------------------------- #

    def _farol_move(self):
        """
        Movement logic for the Lighthouse problem.

        Priority:
          0) If knows exact goal from another agent => BFS path to that (x,y).
          1) Try moving directly according to sensor direction (N/S/E/W combinations).
          2) Try component moves (only vertical or only horizontal).
          3) Smart escape: try neighbors not in recent loop area.
          4) Forced escape: pick any valid neighbor (even if looping).
        """
        d = self.current_obs.get("direcao_farol")
        print(f"   [{self.name}] Observed direction: {d}")

        # 0) If we already know the exact goal position (via communication)
        if self.known_goals:
            target = next(iter(self.known_goals))
            print(f"   [{self.name}] ✅ Using communicated goal {target}")
            return self._follow_or_plan_bfs(target)

        # If sensor says we are exactly at lighthouse
        if d == "HERE":
            print(f"   [{self.name}] 🎯 Reached goal")
            self.reached_goal = True
            return None

        # Desired main movement toward lighthouse based on direction
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
            components.append((self.x + dx, self.y))  # pure vertical
        if dy != 0:
            components.append((self.x, self.y + dy))  # pure horizontal

        for nx, ny in components:
            if self.env.is_valid_position(nx, ny) and not self._is_loop(nx, ny):
                print(f"   [{self.name}] Rule 2 → sliding move {(nx, ny)}")
                return (nx, ny)
            else:
                print(f"   [{self.name}] Rule 2 blocked/loop at {(nx, ny)}")

        # ------------------------------
        # RULE 3: Smart escape (avoid recent loop area)
        # ------------------------------
        print(f"   [{self.name}] Rule 3 → smart escape")

        candidates = [
            (self.x - 1, self.y),  # up
            (self.x + 1, self.y),  # down
            (self.x, self.y - 1),  # left
            (self.x, self.y + 1),  # right
        ]

        # Valid and not obviously looping
        recent = set(list(self.last_positions)[-6:])
        smart_moves = [
            (nx, ny)
            for (nx, ny) in candidates
            if self.env.is_valid_position(nx, ny) and (nx, ny) not in recent
        ]

        if smart_moves:
            choice = random.choice(smart_moves)
            print(f"   [{self.name}] Smart escape → {choice}")
            return choice

        # ------------------------------
        # RULE 4: Last resort → move to any valid tile (even if loop)
        # ------------------------------
        print(f"   [{self.name}] Rule 4 → forced escape")
        valid_moves = [
            (nx, ny)
            for (nx, ny) in candidates
            if self.env.is_valid_position(nx, ny)
        ]

        if valid_moves:
            # Prefer moves that are NOT the very last position if possible
            last_pos = self.last_positions[-1] if self.last_positions else None
            non_last = [m for m in valid_moves if m != last_pos]
            if non_last:
                best = random.choice(non_last)
            else:
                best = random.choice(valid_moves)

            print(f"   [{self.name}] Forced escape → {best}")
            return best

        print(f"   [{self.name}] ❌ No possible moves")
        return None

    # ---------- BFS-BASED MOVEMENT ONCE GOAL IS KNOWN ---------- #

    def _follow_or_plan_bfs(self, goal):
        """
        Follow an existing BFS path to 'goal', or compute a new one if needed.
        This is used *after* the agent knows the exact lighthouse position
        (through communication).
        """
        gx, gy = goal

        # If we're already exactly at the goal
        if (self.x, self.y) == (gx, gy):
            self.reached_goal = True
            print(f"   [{self.name}] 🎯 Already at communicated goal {goal}")
            return None

        # Decide whether we need to (re)plan a path
        need_replan = False

        if not self.planned_path:
            need_replan = True
        else:
            # Check if current position is still on the path
            if (self.x, self.y) not in self.planned_path:
                need_replan = True
            else:
                idx = self.planned_path.index((self.x, self.y))
                if idx == len(self.planned_path) - 1:
                    # At the end of the path but not flagged as goal yet, replan
                    need_replan = True
                else:
                    next_step = self.planned_path[idx + 1]
                    if not self.env.is_valid_position(*next_step):
                        need_replan = True

        # (Re)compute BFS path if needed
        if need_replan:
            print(f"   [{self.name}] 🔁 Planning BFS path to {goal}")
            path = self._bfs_path((self.x, self.y), goal)
            if not path:
                print(f"   [{self.name}] ⚠ No BFS path found, using fallback near goal")
                return self._fallback_near_goal(goal)
            self.planned_path = path

        # Follow the planned path
        idx = self.planned_path.index((self.x, self.y))
        if idx == len(self.planned_path) - 1:
            # Last node in plan; should be goal
            self.reached_goal = True
            print(f"   [{self.name}] 🎯 Reached goal at end of path {goal}")
            return None

        next_step = self.planned_path[idx + 1]
        print(f"   [{self.name}] BFS step toward {goal} → {next_step}")
        return next_step

    def _bfs_path(self, start, goal):
        """
        Breadth-First Search on the grid, using env.is_valid_position to check
        free cells. Returns a list of positions [start, ..., goal] or None
        if no path exists.
        """
        from collections import deque

        if start == goal:
            return [start]

        q = deque([start])
        visited = {start}
        parent = {}

        while q:
            x, y = q.popleft()
            for nx, ny in [(x - 1, y), (x + 1, y),
                           (x, y - 1), (x, y + 1)]:
                if not self.env.is_valid_position(nx, ny):
                    continue
                if (nx, ny) in visited:
                    continue

                visited.add((nx, ny))
                parent[(nx, ny)] = (x, y)

                if (nx, ny) == goal:
                    # Reconstruct path
                    path = [(nx, ny)]
                    while path[-1] != start:
                        path.append(parent[path[-1]])
                    path.reverse()
                    return path

                q.append((nx, ny))

        # No path found
        return None

    def _fallback_near_goal(self, goal):
        """
        Fallback when BFS cannot find a path (should be rare).
        Tries any valid neighbor, still avoiding immediate loops if possible.
        """
        neighbors = [
            (self.x - 1, self.y),
            (self.x + 1, self.y),
            (self.x, self.y - 1),
            (self.x, self.y + 1),
        ]
        candidates = [
            n for n in neighbors
            if self.env.is_valid_position(*n) and not self._is_loop(*n)
        ]
        if candidates:
            choice = random.choice(candidates)
            print(f"   [{self.name}] Fallback escape near goal → {choice}")
            return choice

        # As a last resort, allow loop
        plain_valid = [n for n in neighbors if self.env.is_valid_position(*n)]
        if plain_valid:
            choice = random.choice(plain_valid)
            print(f"   [{self.name}] Fallback (loop allowed) near goal → {choice}")
            return choice

        print(f"   [{self.name}] ❌ No fallback moves near goal")
        return None

    # --------------------------- MAZE MODE --------------------------- #

    def _maze_move(self):
        """
        Maze logic (unchanged from your version).
        Agents:
          - Mark visits
          - Try greedy to goal if known
          - Otherwise explore randomly avoiding known obstacles/dead ends
        """
        self._mark_visit((self.x, self.y))

        # 1) If KNOW goal → greedy move (same idea as before)
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
        """
        Medium-strength loop detection:
        Detects if agent is bouncing inside the same 6-tile cycle.
        Works for loops like A-B-C-D-A-B and similar.
        """
        recent = list(self.last_positions)

        # Short loop (oscillation or 2-cell bounce)
        if (x, y) in recent[-4:]:
            return True

        # Medium loop (cycle of up to 3 cells repeating)
        unique_recent = set(recent[-6:])
        if len(unique_recent) <= 3 and len(recent) >= 6:
            return True

        return False

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
        if d == "up":
            return (self.x - 1, self.y)
        if d == "down":
            return (self.x + 1, self.y)
        if d == "left":
            return (self.x, self.y - 1)
        if d == "right":
            return (self.x, self.y + 1)
