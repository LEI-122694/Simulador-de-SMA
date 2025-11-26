# MazeAgent.py
import random
from collections import deque
from Agent import Agent

class MazeAgent(Agent):
    """
    Maze agent:
      - explores the maze from the common start
      - keeps its own path_from_start as a list of directions
      - when it reaches the goal, broadcasts that path
      - when it receives a path, returns to start then follows it
    """

    def __init__(self, name, env, start_pos):
        super().__init__(name, env, start_pos)

        self.reached_goal = False
        self.current_obs = None

        self.path_from_start = []   # movement history from start
        self.mode = "explore"       # explore / follow_plan / wait

        self.planned_moves = deque()
        self.back_steps_remaining = 0

        self.visited = set()

    # ------------------------------------------------------------
    # OBSERVATION
    # ------------------------------------------------------------
    def observacao(self, obs):
        self.current_obs = obs
        pos = obs.get("posicao")

        print(f"   [{self.name}] OBS → position={pos} mode={self.mode}")

        if pos is not None:
            self.visited.add(pos)

        goals = obs.get("goals", [])
        if goals and pos == goals[0]:
            if not self.reached_goal:
                print(f"   [{self.name}] 🎯 Goal detected at {pos}")
            self.reached_goal = True

    # ------------------------------------------------------------
    # COMMUNICATION
    # ------------------------------------------------------------
    def comunica(self, mensagem, de_agente):
        print(f"   [{self.name}] 📩 Received message from {de_agente.name}: {mensagem}")

        if "path_from_start_to_goal" in mensagem and self.mode == "explore":
            path_to_goal = mensagem["path_from_start_to_goal"]

            print(f"   [{self.name}] ✔ Processing received goal-path: {path_to_goal}")

            # Compute reverse path back to start
            back_path = self._reverse_path(self.path_from_start)

            print(f"   [{self.name}] ↩ Back-to-start path = {back_path}")
            print(f"   [{self.name}] 🚀 Full plan = back_to_start + to_goal")

            full_plan = back_path + path_to_goal

            print(f"   [{self.name}] 🧭 Full plan = {full_plan}")

            self.back_steps_remaining = len(back_path)
            self.planned_moves = deque(full_plan)
            self.mode = "follow_plan"

    # ------------------------------------------------------------
    # DECISION
    # ------------------------------------------------------------
    def age(self):
        # If reached goal
        if self.reached_goal:
            if self.mode == "explore":
                print(f"   [{self.name}] 🎉 Reached goal → Broadcasting path: {self.path_from_start}")
                self._broadcast({"path_from_start_to_goal": list(self.path_from_start)})
                self.mode = "wait"
            return None

        # If following a plan
        if self.mode == "follow_plan" and self.planned_moves:
            direction = self.planned_moves.popleft()
            print(f"   [{self.name}] 🧭 Following plan → {direction}")

            # Back-to-start phase
            if self.back_steps_remaining > 0:
                self.back_steps_remaining -= 1
                if self.path_from_start:
                    popped = self.path_from_start.pop()
                    print(f"   [{self.name}] ↩ Undoing step '{popped}' (return to start)")
            else:
                # From start to goal phase
                self.path_from_start.append(direction)
                print(f"   [{self.name}] ➕ Recording forward step '{direction}'")

            nx, ny = self._apply_dir(direction)
            return (nx, ny)

        # Otherwise explore
        return self._explore_step()

    # ------------------------------------------------------------
    # EXPLORATION (DFS-like)
    # ------------------------------------------------------------
    def _explore_step(self):
        x, y = self.current_obs["posicao"]
        blocked = set(self.current_obs["neighbors"].keys())

        print(f"   [{self.name}] 🔍 Exploring from {x,y}, visited={len(self.visited)}")

        directions = [
            ("up",    (x - 1, y)),
            ("down",  (x + 1, y)),
            ("left",  (x, y - 1)),
            ("right", (x, y + 1)),
        ]
        random.shuffle(directions)

        # Try new cells first
        for d, (nx, ny) in directions:
            if not self.env.is_blocked(nx, ny) and (nx, ny) not in self.visited:
                print(f"   [{self.name}] ➡ Moving {d} to NEW cell {(nx,ny)}")
                self.path_from_start.append(d)
                return (nx, ny)

        # Otherwise backtrack
        if self.path_from_start:
            last = self.path_from_start.pop()
            back = self._opposite(last)
            bx, by = self._apply_dir(back)
            print(f"   [{self.name}] ↩ Dead end → Backtracking: {last} → {back}")
            return (bx, by)

        print(f"   [{self.name}] ❌ Stuck, nowhere to go")
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

    def _opposite(self, direction):
        return {
            "up": "down",
            "down": "up",
            "left": "right",
            "right": "left",
        }[direction]

    def _reverse_path(self, path):
        return [self._opposite(d) for d in reversed(path)]
