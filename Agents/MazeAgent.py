# MazeAgent.py
import random
from collections import deque
from Agents.Agent import Agent

class MazeAgent(Agent):
    """
    Maze agent (Smart Path Alignment with goal-adjacency sensing):
      - explores the maze from the common start
      - stores its own path_from_start as movement sequence
      - when it receives the goal path from another agent:
            1) compute common prefix length k
            2) my_suffix      = my_path[k:]
               goal_suffix    = goal_path[k:]
            3) backtrack_moves = reverse(my_suffix) with opposite dirs
            4) planned_moves   = backtrack_moves + goal_suffix
      - always checks if the goal is in an adjacent cell
      - moves into the goal immediately if detected
      - no coordinates exchanged, all movement-only communication.
    """

    def __init__(self, name, env, start_pos):
        super().__init__(name, env, start_pos)

        self.reached_goal = False
        self.current_obs = None

        self.path_from_start = []     # list of movement directions from the start
        self.mode = "explore"         # explore / follow_plan / wait

        self.planned_moves = deque()
        self.visited = set()

    # ------------------------------------------------------------
    # OBSERVATION
    # ------------------------------------------------------------
    def observacao(self, obs):
        self.current_obs = obs
        pos = obs.get("posicao")

        print(f"   [{self.name}] OBS → pos={pos}, mode={self.mode}")

        if pos is not None:
            self.visited.add(pos)

        # Goal detection (standing on the goal)
        goals = obs.get("goals", [])
        if goals and pos == goals[0]:
            if not self.reached_goal:
                print(f"   [{self.name}] 🎯 Reached goal!")
            self.reached_goal = True

    # ------------------------------------------------------------
    # COMMUNICATION: prefix alignment + backtracking + suffix
    # ------------------------------------------------------------
    def comunica(self, mensagem, de_agente):
        print(f"   [{self.name}] 📩 Received message from {de_agente.name}: {mensagem}")

        if "path_from_start_to_goal" not in mensagem:
            return

        goal_path = mensagem["path_from_start_to_goal"]

        print(f"   [{self.name}] ✔ Received goal-path: {goal_path}")
        print(f"   [{self.name}] 🔍 My own path so far: {self.path_from_start}")

        # Compute the common prefix
        prefix_len = 0
        for a, b in zip(self.path_from_start, goal_path):
            if a == b:
                prefix_len += 1
            else:
                break

        print(f"   [{self.name}] 📏 Common prefix length = {prefix_len}")

        my_suffix   = self.path_from_start[prefix_len:]
        goal_suffix = goal_path[prefix_len:]

        # Reverse my_suffix for backtracking
        backtrack_moves = [self._opposite(d) for d in reversed(my_suffix)]
        print(f"   [{self.name}] 🔄 Backtracking moves to rejoin partner path: {backtrack_moves}")
        print(f"   [{self.name}] 🧭 Then follow partner suffix: {goal_suffix}")

        # Combined plan
        full_plan = backtrack_moves + goal_suffix

        self.planned_moves = deque(full_plan)
        self.mode = "follow_plan"

        print(f"   [{self.name}] ✅ Loaded plan of {len(full_plan)} moves.")

    # ------------------------------------------------------------
    # DECISION (with priority goal-adjacency detection)
    # ------------------------------------------------------------
    def age(self):

        # ---------------- PRIORITY: goal adjacent check ----------------
        goals = self.current_obs.get("goals", [])
        if goals and not self.reached_goal:
            gx, gy = goals[0]
            x, y = self.current_obs["posicao"]

            if (x-1, y) == (gx, gy):
                print(f"   [{self.name}] 🎯 Goal detected UP — stepping into goal!")
                self.path_from_start.append("up")
                return self._apply_dir("up")

            if (x+1, y) == (gx, gy):
                print(f"   [{self.name}] 🎯 Goal detected DOWN — stepping into goal!")
                self.path_from_start.append("down")
                return self._apply_dir("down")

            if (x, y-1) == (gx, gy):
                print(f"   [{self.name}] 🎯 Goal detected LEFT — stepping into goal!")
                self.path_from_start.append("left")
                return self._apply_dir("left")

            if (x, y+1) == (gx, gy):
                print(f"   [{self.name}] 🎯 Goal detected RIGHT — stepping into goal!")
                self.path_from_start.append("right")
                return self._apply_dir("right")

        # ---------------- Standing on the goal ----------------
        if self.reached_goal:
            if self.mode == "explore":
                print(f"   [{self.name}] 📨 Broadcasting found-path: {self.path_from_start}")
                self._broadcast({"path_from_start_to_goal": list(self.path_from_start)})
                self.mode = "wait"
            return None

        # ---------------- Follow plan (backtrack + partner's suffix) ----------------
        if self.mode == "follow_plan":
            if self.planned_moves:
                direction = self.planned_moves.popleft()
                print(f"   [{self.name}] 🧭 Following planned step: {direction}")
                self.path_from_start.append(direction)
                return self._apply_dir(direction)
            else:
                print(f"   [{self.name}] ✅ Finished executing plan. Entering WAIT mode.")
                self.mode = "wait"
                return None

        # ---------------- Exploration (DFS-like) ----------------
        return self._explore_step()

    # ------------------------------------------------------------
    # EXPLORATION (DFS)
    # ------------------------------------------------------------
    def _explore_step(self):
        x, y = self.current_obs["posicao"]

        print(f"   [{self.name}] 🔍 Exploring from {(x, y)}")

        directions = [
            ("up",    (x - 1, y)),
            ("down",  (x + 1, y)),
            ("left",  (x, y - 1)),
            ("right", (x, y + 1)),
        ]
        if self.mode == "train":
            random.shuffle(directions)

        # Prefer unvisited, non-blocked cells
        for d, (nx, ny) in directions:
            if not self.env.is_blocked(nx, ny) and (nx, ny) not in self.visited:
                print(f"   [{self.name}] ➡ NEW move: {d}")
                self.path_from_start.append(d)
                return (nx, ny)

        # Backtracking
        if self.path_from_start:
            last = self.path_from_start.pop()
            reverse = self._opposite(last)
            print(f"   [{self.name}] ↩ Backtracking: {last} → {reverse}")
            return self._apply_dir(reverse)

        print(f"   [{self.name}] ❌ No moves left")
        return None

    # ------------------------------------------------------------
    # HELPERS
    # ------------------------------------------------------------
    def _apply_dir(self, direction):
        if direction == "up":
            return (self.x - 1, self.y)
        if direction == "down":
            return (self.x + 1, self.y)
        if direction == "left":
            return (self.x, self.y - 1)
        if direction == "right":
            return (self.x, self.y + 1)
        return (self.x, self.y)

    def _opposite(self, d):
        return {
            "up": "down",
            "down": "up",
            "left": "right",
            "right": "left",
        }[d]
