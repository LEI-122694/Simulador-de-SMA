import json
from Environments.World import World
from Agents.LighthouseAgent import LighthouseAgent
from Agents.MazeAgent import MazeAgent
from Main import MotorDeSimulacao


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

    return env, agents

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

    return env, agents


# ------------------------------
# Teste rápido
# ------------------------------
if __name__ == "__main__":
    #env, agents = setup_lighthouse_from_json("Resources/farol_map_1.json")
    #for a in agents:
        #a.set_mode("test")  # ou "train"

    env, agents = setup_maze_from_json("Resources/maze_map_2.json")

    for a in agents:
        a.set_mode("test")  # ou "train"


    motor = MotorDeSimulacao(env, agents)
    motor.executa()
