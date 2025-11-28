import json
import os
from Environments.Lighthouse import setup_lighthouse
from Environments.RandomMazeGenerator import generate_maze

# Pasta onde vamos salvar os mapas
RESOURCES_DIR = "Resources"
os.makedirs(RESOURCES_DIR, exist_ok=True)

# ------------------------
# FUNÇÃO PARA SALVAR FAROL
# ------------------------
def save_lighthouse_map(filename):
    env, agents = setup_lighthouse()

    # Construir dicionário
    data = {
        "height": env.height,
        "width": env.width,
        "goals": list(env.goals),
        "obstacles": list(env.obstacles),
        "start_positions": {agent.name: [agent.x, agent.y] for agent in agents}
    }

    # Salvar JSON
    filepath = os.path.join(RESOURCES_DIR, filename)
    with open(filepath, "w") as f:
        json.dump(data, f, indent=4)
    print(f"[Farol] Mapa salvo em {filepath}")

# ------------------------
# FUNÇÃO PARA SALVAR MAZE
# ------------------------
def save_maze_map(filename, height=11, width=11):
    maze_data = generate_maze(height, width)
    walls = list(maze_data["walls"])
    entrances = maze_data["entrances"]

    data = {
        "height": height,
        "width": width,
        "goals": [entrances[1]],
        "obstacles": walls,
        "start_positions": {
            "A": entrances[0],
            "B": entrances[0]
        }
    }

    filepath = os.path.join(RESOURCES_DIR, filename)
    with open(filepath, "w") as f:
        json.dump(data, f, indent=4)
    print(f"[Maze] Mapa salvo em {filepath}")

# ------------------------
# EXEMPLO DE USO
# ------------------------
if __name__ == "__main__":
    # Gerar 3 mapas de Farol
    for i in range(1, 4):
        save_lighthouse_map(f"farol_map_{i}.json")

    # Gerar 5 mapas de Maze
    for i in range(1, 4):
        save_maze_map(f"maze_map_{i}.json")
