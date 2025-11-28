import random

def generate_maze(height, width):
    """
    Generates a perfect maze using recursive backtracking.
    Returns:
        {
            "walls": set((i,j)),
            "entrances": [(r1,c1), (r2,c2)]
        }
    """

    # Ensure odd dimensions for good DFS structure
    H = height if height % 2 == 1 else height - 1
    W = width  if width  % 2 == 1 else width  - 1

    # Start with all walls
    maze = [[1 for _ in range(W)] for _ in range(H)]

    # Random odd start cell
    start_r = random.randrange(1, H, 2)
    start_c = random.randrange(1, W, 2)
    maze[start_r][start_c] = 0

    stack = [(start_r, start_c)]
    dirs = [(-2, 0), (2, 0), (0, -2), (0, 2)]  # 2-step moves

    # ---------------- DFS CARVING ----------------
    while stack:
        r, c = stack[-1]
        neighbors = []

        for dr, dc in dirs:
            nr, nc = r + dr, c + dc
            if 1 <= nr < H - 1 and 1 <= nc < W - 1 and maze[nr][nc] == 1:
                neighbors.append((nr, nc, dr, dc))

        if neighbors:
            nr, nc, dr, dc = random.choice(neighbors)
            maze[r + dr//2][c + dc//2] = 0
            maze[nr][nc] = 0
            stack.append((nr, nc))
        else:
            stack.pop()

    # ----------------------------------------------
    # FORCE 2 OPPOSITE EDGE ENTRANCES (proper corridors)
    # ----------------------------------------------
    edge_sets = {
        "top":    [(0, c)     for c in range(1, W-1, 2)],
        "bottom": [(H-1, c)   for c in range(1, W-1, 2)],
        "left":   [(r, 0)     for r in range(1, H-1, 2)],
        "right":  [(r, W-1)   for r in range(1, H-1, 2)],
    }

    opposite_pairs = [
        ("top", "bottom"),
        ("bottom", "top"),
        ("left", "right"),
        ("right", "left"),
    ]

    edge1, edge2 = random.choice(opposite_pairs)

    e1 = random.choice(edge_sets[edge1])
    e2 = random.choice(edge_sets[edge2])

    entrances = [e1, e2]

    def carve_entrance(er, ec):
        """Carve a proper 2-cell deep corridor at the entrance."""
        maze[er][ec] = 0
        if er == 0:          # top entrance
            maze[1][ec] = 0
        elif er == H - 1:    # bottom entrance
            maze[H - 2][ec] = 0
        elif ec == 0:        # left entrance
            maze[er][1] = 0
        elif ec == W - 1:    # right entrance
            maze[er][W - 2] = 0

    # Carve both entrances
    carve_entrance(*e1)
    carve_entrance(*e2)

    # ----------------------------------------------
    # CONVERT TO WALL SET FOR THE WORLD
    # ----------------------------------------------
    walls = set()
    for i in range(height):
        for j in range(width):
            # Use generated maze for valid area
            if i < H and j < W:
                if maze[i][j] == 1:
                    walls.add((i, j))
            else:
                # Padding area = walls
                walls.add((i, j))

    return {
        "walls": walls,
        "entrances": entrances
    }
