# Lighthouse.py
import random
from World import World
from ExplorerAgent import ExplorerAgent

HEIGHT = 10
WIDTH = 10
OBSTACLE_RATIO = 0.25  # 20% of cells


def _is_reachable(start, goal, obstacles):
    """Simple BFS to check if goal is reachable from start."""
    from collections import deque

    q = deque([start])
    visited = {start}

    while q:
        x, y = q.popleft()
        if (x, y) == goal:
            return True

        for nx, ny in [(x-1, y), (x+1, y), (x, y-1), (x, y+1)]:
            if 0 <= nx < HEIGHT and 0 <= ny < WIDTH \
               and (nx, ny) not in obstacles \
               and (nx, ny) not in visited:
                visited.add((nx, ny))
                q.append((nx, ny))

    return False


def setup_lighthouse():
    """
    FAROL environment (randomized):
      - 10x10 grid
      - 1 lighthouse at RANDOM valid cell
      - ~20% cells become random obstacles
      - 2 agents start at random valid cells
      - Guarantee both agents can reach the lighthouse (BFS check)
    """

    all_cells = [(x, y) for x in range(HEIGHT) for y in range(WIDTH)]
    total_cells = HEIGHT * WIDTH
    num_obstacles = int(total_cells * OBSTACLE_RATIO)

    while True:
        # -------------------------------------------------
        # 1) Randomly choose a GOAL position
        # -------------------------------------------------
        goal = random.choice(all_cells)

        # -------------------------------------------------
        # 2) Random obstacle placement (avoid the goal)
        # -------------------------------------------------
        candidate_cells = [c for c in all_cells if c != goal]
        obstacles = set(random.sample(candidate_cells, num_obstacles))

        # -------------------------------------------------
        # 3) Free cells = empty area (no goal, no obstacles)
        # -------------------------------------------------
        free = [c for c in all_cells if c not in obstacles and c != goal]

        if len(free) < 2:
            continue

        # -------------------------------------------------
        # 4) Choose random agent starting positions
        # -------------------------------------------------
        start_A = random.choice(free)
        free.remove(start_A)

        start_B = random.choice(free)

        # -------------------------------------------------
        # 5) Ensure BOTH agents can reach the goal
        # -------------------------------------------------
        if _is_reachable(start_A, goal, obstacles) and _is_reachable(start_B, goal, obstacles):
            break
        # otherwise regenerate map

    # -------------------------------------------------
    # 6) Print debug info
    # -------------------------------------------------
    print(f"[Farol] Lighthouse at = {goal}")
    print(f"[Farol] Start A = {start_A}")
    print(f"[Farol] Start B = {start_B}")
    print(f"[Farol] Obstacles = {len(obstacles)} ({int(OBSTACLE_RATIO*100)}%)")

    # -------------------------------------------------
    # 7) Build environment and agents
    # -------------------------------------------------
    env = World(
        height=HEIGHT,
        width=WIDTH,
        goals=[goal],
        obstacles=obstacles,
        mode="farol"
    )

    a1 = ExplorerAgent("A", env, start_pos=start_A)
    a2 = ExplorerAgent("B", env, start_pos=start_B)

    return env, [a1, a2]
