# Lighthouse.py
import random
from World import World
from LighthouseAgent import LighthouseAgent

HEIGHT = 10
WIDTH = 10
OBSTACLE_RATIO = 0.30  # 30% obstáculos

def setup_lighthouse():
    """
    FAROL environment (randomizado):
      - 10x10 grid
      - Farol em célula aleatória
      - ~30% obstáculos
      - Agentes começam em posições aleatórias
    """
    all_cells = [(x, y) for x in range(HEIGHT) for y in range(WIDTH)]
    total_cells = HEIGHT * WIDTH
    num_obstacles = int(total_cells * OBSTACLE_RATIO)

    # 1) Farol aleatório
    goal = random.choice(all_cells)

    # 2) Obstáculos aleatórios (não bloqueiam o farol)
    candidate_cells = [c for c in all_cells if c != goal]
    obstacles = set(random.sample(candidate_cells, num_obstacles))

    # 3) Células livres
    free = [c for c in all_cells if c not in obstacles and c != goal]

    # 4) Posições aleatórias dos agentes
    start_A = random.choice(free)
    free.remove(start_A)
    start_B = random.choice(free)

    # 5) Debug
    print(f"[Farol] Lighthouse at = {goal}")
    print(f"[Farol] Start A = {start_A}")
    print(f"[Farol] Start B = {start_B}")
    print(f"[Farol] Obstacles = {len(obstacles)} ({int(OBSTACLE_RATIO*100)}%)")

    # 6) Construir ambiente e agentes
    env = World(
        height=HEIGHT,
        width=WIDTH,
        goals=[goal],
        obstacles=obstacles,
        mode="farol"
    )

    a1 = LighthouseAgent("A", env, start_pos=start_A)
    a2 = LighthouseAgent("B", env, start_pos=start_B)

    env.add_agent(a1)
    env.add_agent(a2)

    return env, [a1, a2]
