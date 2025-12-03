# Lighthouse.py
import random
import json
from collections import deque

from Environments.World import World
from Agents.LighthouseFixedAgent import LighthouseFixedAgent
#from Agents.LighthouseLearningAgent import LighthouseLearningAgent


# ---------------------------------------------------------
# CONFIGURAÇÕES DO MAPA
# ---------------------------------------------------------
HEIGHT = 10
WIDTH = 10
OBSTACLE_RATIO = 0.30
MIN_START_DIST = 14



# ---------------------------------------------------------
# BFS – verifica se existe caminho até ao farol
# ---------------------------------------------------------
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



# ---------------------------------------------------------
# 1) Carregar mapa FIXO do JSON
# ---------------------------------------------------------
def load_fixed_map(filename):
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

    return env, data["start_positions"], goals[0], obstacles



# ---------------------------------------------------------
# 2) Gerar mapa RANDOM para FixedAgent
# ---------------------------------------------------------
def generate_random_map():
    all_cells = [(x, y) for x in range(HEIGHT) for y in range(WIDTH)]
    total_cells = HEIGHT * WIDTH
    num_obstacles = int(total_cells * OBSTACLE_RATIO)

    while True:
        goal = random.choice(all_cells)

        candidate_cells = [c for c in all_cells if c != goal]
        obstacles = set(random.sample(candidate_cells, num_obstacles))

        free = [c for c in all_cells if c not in obstacles and c != goal]
        if len(free) < 2:
            continue

        start_A = random.choice(free)
        free.remove(start_A)
        start_B = random.choice(free)

        def manhattan(a, b):
            return abs(a[0] - b[0]) + abs(a[1] - b[1])

        if manhattan(start_A, goal) < MIN_START_DIST:
            continue
        if manhattan(start_B, goal) < MIN_START_DIST:
            continue

        if not is_reachable(start_A, goal, obstacles):
            continue
        if not is_reachable(start_B, goal, obstacles):
            continue

        break

    env = World(
        height=HEIGHT,
        width=WIDTH,
        goals=[goal],
        obstacles=obstacles,
        mode="farol"
    )

    start_positions = {
        "A": start_A,
        "B": start_B
    }

    return env, start_positions, goal, obstacles



# ---------------------------------------------------------
# 3) SETUP UNIFICADO (usa agent_type + map_type)
# ---------------------------------------------------------
def setup_lighthouse(agent_type="fixed", map_type="fixed", json_file="Resources/farol_map_1.json"):
    """
    agent_type = "fixed" ou "learning"
    map_type   = "fixed" ou "random"
    """

    # ---------------------------------------------------------
    # REGRA IMPORTANTE
    # ---------------------------------------------------------
    if agent_type == "learning" and map_type == "random":
        raise ValueError("❌ Erro: LearningAgent só pode ser usado com mapa FIXED.")

    # ---------------------------------------------------------
    # MAPA
    # ---------------------------------------------------------
    if map_type == "fixed":
        env, start_positions, goal, obstacles = load_fixed_map(json_file)

    elif map_type == "random":
        env, start_positions, goal, obstacles = generate_random_map()

    else:
        raise ValueError("map_type deve ser 'fixed' ou 'random'")


    # ---------------------------------------------------------
    # CRIAÇÃO DOS AGENTES
    # ---------------------------------------------------------
    agents = []

    for name, pos in start_positions.items():

        if agent_type == "fixed":
            agent = LighthouseFixedAgent(name, env, start_pos=tuple(pos))

        #elif agent_type == "learning":
            #agent = LighthouseLearningAgent(name, env, start_pos=tuple(pos))

        else:
            raise ValueError("agent_type deve ser 'fixed' ou 'learning'")

        agents.append(agent)

    return env, agents
