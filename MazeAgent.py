# MazeAgent.py
from Agent import Agent
import random
from collections import deque

class MazeAgent(Agent):
    """
    Agent for MAZE mode.
    Explores the maze, avoids obstacles, shares discovered locations.
    """

    def __init__(self, name, env, start_pos):
        super().__init__(name, env, start_pos)

        self.visited = set()
        self.frontier_obstacles = set()   # obstacles learned from others
        self.goal_known = None            # for BFS when discovered

    # ------------------------------------------------------------
    # OBSERVATION
    # ------------------------------------------------------------
    def observacao(self, obs):
        self.current_obs = obs

        # store visited
        self.visited.add((self.x, self.y))

        # read obstacles from neighbors
        neigh = obs.get("neighbors", {})
        for (nx, ny), typename in neigh.items():
            if typename == "obstacle":
                self.frontier_obstacles.add((nx, ny))

        # goal found?
        if obs.get("goals"):
            self.goal_known = obs["goals"][0]
            self._broadcast({"goal": self.goal_known})

    # ------------------------------------------------------------
    # COMMUNICATION
    # ------------------------------------------------------------
    def comunica(self, mensagem, de_agente):
        if "goal" in mensagem and self.goal_known is None:
            self.goal_known = mensagem["goal"]

        if "obstacles" in mensagem:
            self.frontier_obstacles.update(mensagem["obstacles"])

    # ------------------------------------------------------------
    # DECISION (AGE)
    # ------------------------------------------------------------
    def age(self):
        # If goal known and reachable → BFS to it
        if self.goal_known:
            path = self._bfs(self.goal_known)
            if len(path) >= 2:
                return path[1]

        # otherwise explore maze
        nxt = self._explore_maze()
        if nxt:
            return nxt

        # stuck: broadcast knowledge so others avoid same trap
        self._broadcast({"obstacles": list(self.frontier_obstacles)})
        return None

    # ------------------------------------------------------------
    # INTERNAL HELPERS
    # ------------------------------------------------------------
    def _explore_maze(self):
        dirs = [(1,0), (-1,0), (0,1), (0,-1)]
        random.shuffle(dirs)

        for dx, dy in dirs:
            nx, ny = self.x + dx, self.y + dy
            if not self._is_blocked(nx, ny):
                return (nx, ny)

        return None

    def _is_blocked(self, x, y):
        return (x, y) in self.frontier_obstacles or self.env.is_blocked(x, y)

    # ------------------------------------------------------------
    # BFS PATHFINDING
    # ------------------------------------------------------------
    def _bfs(self, goal):
        start = (self.x, self.y)
        q = deque([start])
        visited = {start: None}

        while q:
            cx, cy = q.popleft()
            if (cx, cy) == goal:
                break

            for dx, dy in [(1,0), (-1,0), (0,1), (0,-1)]:
                nx, ny = cx + dx, cy + dy
                if (nx, ny) not in visited and not self._is_blocked(nx, ny):
                    visited[(nx, ny)] = (cx, cy)
                    q.append((nx, ny))

        if goal not in visited:
            return []

        # reconstruct path
        path = []
        cur = goal
        while cur is not None:
            path.append(cur)
            cur = visited[cur]
        return list(reversed(path))
