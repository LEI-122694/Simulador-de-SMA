from World import World
from MazeAgent import MazeAgent
from RandomMazeGenerator import generate_maze
import random

def setup_maze():
    height = 21
    width  = 21

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