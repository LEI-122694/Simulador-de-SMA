# MazeAgent.py
from Agent import Agent
import random
from collections import deque

class MazeAgent(Agent):
    """
    Agente para o modo MAZE.
    Explora labirintos, evita obstáculos e registra recompensas.
    Suporta Learning Mode (Q-learning) e Test Mode (política fixa).
    """

    def __init__(self, name, env, start_pos, mode="test"):
        super().__init__(name, env, start_pos)
        self.visited = set()
        self.frontier_obstacles = set()
        self.goal_known = None
        self.mode = mode
        self.q_table = {}  # usado no learning
        self.rewards = []
        self.reached_goal = False

    # ------------------------------------------------------------
    # OBSERVAÇÃO
    # ------------------------------------------------------------
    def observacao(self, obs):
        self.current_obs = obs
        self.visited.add((self.x, self.y))

        # registra obstáculos vizinhos
        neigh = obs.get("neighbors", {})
        for (nx, ny), typename in neigh.items():
            if typename == "obstacle":
                self.frontier_obstacles.add((nx, ny))

        # goal encontrado
        if obs.get("goals"):
            self.goal_known = obs["goals"][0]
            self._broadcast({"goal": self.goal_known})

    # ------------------------------------------------------------
    # COMUNICAÇÃO
    # ------------------------------------------------------------
    def comunica(self, mensagem, de_agente):
        if "goal" in mensagem and self.goal_known is None:
            self.goal_known = mensagem["goal"]

        if "obstacles" in mensagem:
            self.frontier_obstacles.update(mensagem["obstacles"])

    # ------------------------------------------------------------
    # AVALIAÇÃO DE RECOMPENSA (Learning Mode)
    # ------------------------------------------------------------
    def avaliacaoEstadoAtual(self, recompensa):
        self.rewards.append(recompensa)
        if self.mode == "learning":
            state = (self.x, self.y)
            self.q_table[state] = self.q_table.get(state, 0) + 0.1 * (recompensa - self.q_table.get(state, 0))

    # ------------------------------------------------------------
    # DECISÃO
    # ------------------------------------------------------------
    def age(self):
        # Se goal conhecido → BFS
        if self.goal_known:
            path = self._bfs(self.goal_known)
            if len(path) >= 2:
                return path[1]

        # Learning mode → usa Q-table
        if self.mode == "learning":
            state = (self.x, self.y)
            if random.random() < 0.2:  # exploração
                nxt = self._explore_maze()
            else:  # exploit
                nxt = max(
                    [(s, self.q_table.get(s, 0)) for s in self._valid_neighbors()],
                    key=lambda x: x[1],
                    default=(None, None)
                )[0]
            if nxt:
                return nxt

        # fallback → exploração aleatória
        return self._explore_maze()

    # ------------------------------------------------------------
    # MOVIMENTO ALEATÓRIO
    # ------------------------------------------------------------
    def _explore_maze(self):
        dirs = [(1,0), (-1,0), (0,1), (0,-1)]
        random.shuffle(dirs)
        for dx, dy in dirs:
            nx, ny = self.x + dx, self.y + dy
            if not self._is_blocked(nx, ny):
                return (nx, ny)
        return None

    def _valid_neighbors(self):
        return [(self.x+dx, self.y+dy) for dx, dy in [(1,0), (-1,0), (0,1), (0,-1)] if not self._is_blocked(self.x+dx, self.y+dy)]

    def _is_blocked(self, x, y):
        return (x, y) in self.frontier_obstacles or self.env.is_blocked(x, y)

    # ------------------------------------------------------------
    # BFS para pathfinding
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
                nx, ny = cx+dx, cy+dy
                if (nx, ny) not in visited and not self._is_blocked(nx, ny):
                    visited[(nx, ny)] = (cx, cy)
                    q.append((nx, ny))

        if goal not in visited:
            return []

        # reconstrói caminho
        path = []
        cur = goal
        while cur is not None:
            path.append(cur)
            cur = visited[cur]
        return list(reversed(path))
