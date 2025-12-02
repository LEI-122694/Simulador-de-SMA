# Maze.py
import json
from Environments.World import World
from Agents.MazeFixedAgent import MazeAgent
from Environments.RandomMazeGenerator import generate_maze


# -----------------------------------------------------
# SETUP MAZE — carregar de JSON (modo TESTE)
# -----------------------------------------------------
def setup_maze_from_json(filename):
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

    agents = []
    for name, pos in data["start_positions"].items():
        agent = MazeAgent(name, env, start_pos=tuple(pos))
        agents.append(agent)

    print(f"[Maze JSON] Loaded map from {filename}")
    return env, agents


# -----------------------------------------------------
# SETUP MAZE — gerado aleatoriamente (modo TREINO)
# -----------------------------------------------------
def setup_maze_random():
    height = 11
    width = 11

    maze_data = generate_maze(height, width)
    walls = maze_data["walls"]
    entrances = maze_data["entrances"]

    start_pos = entrances[0]
    goal = entrances[1]

    print(f"[Maze] Start = {start_pos}, Goal = {goal}")

    env = World(
        height=height,
        width=width,
        goals=[goal],
        obstacles=walls,
        mode="maze"
    )

    a1 = MazeAgent("A", env, start_pos=start_pos)
    a2 = MazeAgent("B", env, start_pos=start_pos)

    return env, [a1, a2]


# -----------------------------------------------------
# SETUP UNIFICADO — modo TEST / TRAIN
# -----------------------------------------------------
def setup_maze(mode="test", json_file="Resources/maze_map_1.json"):
    """
    mode = "test"  → carregar mapa de JSON
    mode = "train" → gerar mapa aleatório
    """
    if mode == "test":
        return setup_maze_from_json(json_file)
    else:
        return setup_maze_random()
