from World import World
from ExplorerAgent import ExplorerAgent

def setup_lighthouse():
    """10x10 Lighthouse environment with a more complex but solvable maze-like obstacle layout."""

    goal = (8, 8)

    obstacles = {
        # Top rows – partial walls
        (0,0),(0,1),
        (1,0),(1,1),(1,3),(1,5),

        # Middle left & center walls
        (2,5),(2,7),(2,8),
        (3,1),(3,5),(3,7),
        (4,1),(4,3),(4,4),(4,5),(4,6),(4,7),

        # Lower-middle corridors

        (6,4),(6,5),(6,7),

        # Right-side semi-walls

        # Bottom fully closed
        (9,0),(9,1),(9,2),(9,3),(9,4),
        (9,5),(9,6),(9,7),(9,8),(9,9),
    }

    env = World(
        height=10,
        width=10,
        goals=[goal],
        obstacles=obstacles,
        mode="farol"
    )

    # Starting positions are valid and solvable
    a1 = ExplorerAgent("A", env, start_pos=(0, 9))
    a2 = ExplorerAgent("B", env, start_pos=(2, 3))

    return env, [a1, a2]
