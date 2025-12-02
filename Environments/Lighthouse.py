# Lighthouse.py
import random
import json
from collections import deque
from Environments.World import World
from Agents.LighthouseFixedAgent import LighthouseAgent

HEIGHT = 10
WIDTH = 10
OBSTACLE_RATIO = 0.30     # densidade de obstáculos
MIN_START_DIST = 14       # distância Manhattan mínima ao farol


# -----------------------------------------------------
# BFS — verifica se existe caminho até ao farol
# -----------------------------------------------------
def is_reachable(start, goal, obstacles):
    q = deque([start])
    visited = {start}

    while q:
        x, y = q.popleft()

        if (x, y) == goal:
            return True

        for nx, ny in [(x-1, y), (x+1, y), (x, y-1), (x, y+1)]:
            if (0 <= nx < HEIGHT and
                0 <= ny < WIDTH and
                (nx, ny) not in obstacles and
                (nx, ny) not in visited):
                visited.add((nx, ny))
                q.append((nx, ny))

    return False


# -----------------------------------------------------
# SETUP FAROL — carregar de JSON (modo TESTE)
# -----------------------------------------------------
def setup_lighthouse_from_json(filename):
    with open(filename, "r") as f:
        data = json.load(f)

    height = data["height"]
    width = data["width"]
    goals = [tuple(g) for g in data["goals"]]
    obstacles = {tuple(o) for o in data["obstacles"]}

    env = World(
        height=height,
        width=width,
        goals=goals,
        obstacles=obstacles,
        mode="farol"
    )

    agents = []
    for name, pos in data["start_positions"].items():
        agent = LighthouseAgent(name, env, start_pos=tuple(pos))
        agents.append(agent)

    print(f"[Farol JSON] Loaded map from {filename}")
    return env, agents


# -----------------------------------------------------
# SETUP FAROL — gerado aleatoriamente (modo TREINO)
# -----------------------------------------------------
def setup_lighthouse_random():
    all_cells = [(x, y) for x in range(HEIGHT) for y in range(WIDTH)]
    total_cells = HEIGHT * WIDTH
    num_obstacles = int(total_cells * OBSTACLE_RATIO)

    while True:
        # 1) Farol aleatório
        goal = random.choice(all_cells)

        # 2) Obstáculos aleatórios
        candidate_cells = [c for c in all_cells if c != goal]
        obstacles = set(random.sample(candidate_cells, num_obstacles))

        # 3) Células livres restantes
        free = [c for c in all_cells if c not in obstacles and c != goal]
        if len(free) < 2:
            continue  # mapa impossível

        # 4) Spawns aleatórios
        start_A = random.choice(free)
        free.remove(start_A)
        start_B = random.choice(free)

        # 5) Distância mínima ao farol
        def manhattan(a, b):
            return abs(a[0] - b[0]) + abs(a[1] - b[1])

        if manhattan(start_A, goal) < MIN_START_DIST:
            continue
        if manhattan(start_B, goal) < MIN_START_DIST:
            continue

        # 6) Garantir caminho até ao farol
        if not is_reachable(start_A, goal, obstacles):
            continue
        if not is_reachable(start_B, goal, obstacles):
            continue

        # Mapa válido encontrado
        break

    # Debug (opcional)
    print(f"[Farol] Lighthouse at = {goal}")
    print(f"[Farol] Start A = {start_A}")
    print(f"[Farol] Start B = {start_B}")
    print(f"[Farol] Obstacles = {len(obstacles)} ({int(OBSTACLE_RATIO * 100)}%)")

    # Criar ambiente
    env = World(
        height=HEIGHT,
        width=WIDTH,
        goals=[goal],
        obstacles=obstacles,
        mode="farol"
    )

    # Criar agentes
    a1 = LighthouseAgent("A", env, start_pos=start_A)
    a2 = LighthouseAgent("B", env, start_pos=start_B)

    return env, [a1, a2]


# -----------------------------------------------------
# SETUP UNIFICADO — modo TEST / TRAIN
# -----------------------------------------------------
def setup_lighthouse(mode="test", json_file="Resources/farol_map_1.json"):
    """
    mode = "test"  → carregar mapa de JSON
    mode = "train" → gerar mapa aleatório
    """
    if mode == "test":
        return setup_lighthouse_from_json(json_file)
    else:
        return setup_lighthouse_random()
