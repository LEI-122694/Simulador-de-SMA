# Environments/Maze.py
import json
from Environments.World import World
from Agents.Fixed.MazeFixedAgent import MazeFixedAgent
from Environments.RandomMazeGenerator import generate_maze


# ---------------------------------------------------------
# 1) Carregar mapa FIXO de JSON
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
        mode="maze"
    )

    return env, data["start_positions"], goals, obstacles


# ---------------------------------------------------------
# 2) Gerar mapa RANDOM para FixedAgent
# ---------------------------------------------------------
def generate_random_maze():
    height = 11
    width = 11

    maze_data = generate_maze(height, width)
    walls = maze_data["walls"]
    entrances = maze_data["entrances"]

    start_A = entrances[0]
    start_B = entrances[0]   # pode ser igual, é permitido
    goal = entrances[1]

    print(f"[Maze] Start A = {start_A}, Start B = {start_B}, Goal = {goal}")

    env = World(
        height=height,
        width=width,
        goals=[goal],
        obstacles=walls,
        mode="maze"
    )

    start_positions = {
        "A": start_A,
        "B": start_B
    }

    return env, start_positions, [goal], walls


# ---------------------------------------------------------
# 3) SETUP UNIFICADO (usa agent_type + map_type)
#    Agora: este ficheiro só cria agentes FIXED.
#    Agentes "learning" são criados no Main.py ou nos scripts de treino.
# ---------------------------------------------------------
def setup_maze(agent_type="fixed", map_type="fixed", json_file="Resources/maze_map_1.json"):
    """
    agent_type = "fixed"
    map_type   = "fixed" ou "random"
    """

    if agent_type != "fixed":
        raise ValueError(
            "setup_maze já não constrói agentes 'learning'. "
            "Use Main.py ou os scripts de treino com LearningAgent."
        )

    # MAPA
    if map_type == "fixed":
        env, start_positions, goals, obstacles = load_fixed_map(json_file)
    elif map_type == "random":
        env, start_positions, goals, obstacles = generate_random_maze()
    else:
        raise ValueError("map_type deve ser 'fixed' ou 'random'")

    # AGENTES FIXOS
    agents = []
    for name, pos in start_positions.items():
        agent = MazeFixedAgent(name, env, start_pos=tuple(pos))
        agents.append(agent)

    return env, agents
