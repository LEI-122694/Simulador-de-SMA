from World import World
from ExplorerAgent import ExplorerAgent

def setup_maze():
    """
    Clean, hand-designed professional maze (Option A).
    9x10 grid, solvable, real-corrior structure.
    """

    walls = {
        # Row 0
        (0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (0, 6), (0, 7), (0, 8), (0, 9),

        # Row 1
        (1, 0), (1, 2), (1, 7), (1, 9),

        # Row 2
        (2, 0), (2, 2), (2, 4), (2, 5), (2, 6), (2, 7), (2, 9),

        # Row 3
        (3, 0), (3, 9),

        # Row 4
        (4, 0), (4, 2), (4, 5), (4, 7), (4, 9),

        # Row 5
        (5, 0), (5, 2), (5, 3), (5, 4),
        (5, 5), (5, 6), (5, 7), (5, 9),

        # Row 6
        (6, 0), (6, 2), (6, 4), (6, 9),

        # Row 7
        (7, 0), (7, 4), (7, 6), (7, 7), (7, 9),

        # Row 8
        (8, 0), (8, 1), (8, 2), (8, 3), (8, 4), (8, 6), (8, 7), (8, 8), (8, 9)
    }

    goal = (8, 6)

    env = World(
        height=9,
        width=10,
        goals=[goal],
        obstacles=walls,
        mode="maze"
    )

    # Agents start in valid walkable positions
    a1 = ExplorerAgent("A", env, start_pos=(1, 1))
    a2 = ExplorerAgent("B", env, start_pos=(7, 8))

    return env, [a1, a2]
