import os
import random
import json
import math
import matplotlib.pyplot as plt

from Environments.Maze import load_fixed_map
from Environments.World import World
from Agents.MazeLearningAgent import MazeLearningAgent


# ============================================================
# PATHS / CONSTANTS
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
MAP_FILE = os.path.join(BASE_DIR, "Resources", "maze_map_1.json")
BEST_GENOTYPE_FILE = os.path.join(BASE_DIR, "maze_best_genotype.json")

POP_SIZE = 40
GENERATIONS = 40
STEPS_PER_EPISODE = 250

ELITE_FRACTION = 0.2        # top % kept as parents
MUTATION_RATE = 0.1         # probability each gene is mutated
MUTATION_STD = 0.3          # std-dev of gaussian noise

K_NEIGHBORS = 10            # k-NN for novelty
ARCHIVE_ADD_TOP_K = 3       # add this many most novel per generation to archive


# ============================================================
# BEHAVIOR / NOVELTY
# ============================================================

def behavior_distance(b1, b2):
    """
    Jaccard distance between two sets of visited cells.
    Distance in [0,1]. 0 = identical, 1 = completely different.
    """
    if not b1 and not b2:
        return 0.0
    inter = len(b1 & b2)
    union = len(b1 | b2)
    if union == 0:
        return 0.0
    return 1.0 - inter / union


def compute_novelty(behaviors, archive, k=K_NEIGHBORS):
    """
    Compute novelty score for each behavior, using average distance
    to k nearest neighbors in (archive + current population).
    behaviors: list of sets of visited (x,y)
    """
    all_behaviors = archive + behaviors
    n = len(behaviors)
    novelties = [0.0] * n

    if len(all_behaviors) <= 1:
        # no reference yet
        return novelties

    for i, beh in enumerate(behaviors):
        dists = []
        for other in all_behaviors:
            if other is beh:
                continue
            d = behavior_distance(beh, other)
            dists.append(d)

        if not dists:
            novelties[i] = 0.0
        else:
            dists.sort(reverse=True)  # higher distance = more novel
            k_use = min(k, len(dists))
            novelties[i] = sum(dists[:k_use]) / k_use

    return novelties


def update_archive(archive, behaviors, novelties, add_top_k=ARCHIVE_ADD_TOP_K):
    """
    Add the top-k most novel behaviors to the archive (if not too similar).
    """
    if not behaviors:
        return archive

    indices = list(range(len(behaviors)))
    indices.sort(key=lambda i: novelties[i], reverse=True)

    for idx in indices[:add_top_k]:
        cand = behaviors[idx]

        if not archive:
            archive.append(cand)
            continue

        min_dist = min(behavior_distance(cand, a) for a in archive)
        # only add if sufficiently different from existing archive behaviors
        if min_dist > 0.1:
            archive.append(cand)

    return archive


# ============================================================
# GENOTYPE HELPERS
# ============================================================

def random_genotype():
    size = MazeLearningAgent.genotype_size()
    scale = 0.5
    return [random.uniform(-scale, scale) for _ in range(size)]


def mutate_genotype(genes, rate=MUTATION_RATE, std=MUTATION_STD):
    new = genes[:]
    for i in range(len(new)):
        if random.random() < rate:
            new[i] += random.gauss(0.0, std)
    return new


def crossover_genotypes(parent1, parent2):
    """
    Uniform crossover between two parent genotypes.
    """
    assert len(parent1) == len(parent2)
    child = []
    for g1, g2 in zip(parent1, parent2):
        child.append(g1 if random.random() < 0.5 else g2)
    return child


# ============================================================
# SIMULATION OF ONE GENOTYPE
# ============================================================

def simulate_genotype(genes, base_env, start_pos, goals, obstacles, steps=STEPS_PER_EPISODE):
    """
    Evaluate a single neural agent (given its genotype) in the maze.
    Returns:
      visited_set (set of (x,y)),
      reached_goal (bool)
    """
    # New env with same layout
    env = World(
        height=base_env.height,
        width=base_env.width,
        goals=goals,
        obstacles=obstacles,
        mode="maze"
    )

    agent = MazeLearningAgent("EVO", env, start_pos=start_pos, weights=genes)
    agent.set_mode("test")  # no learning, just NN control
    env.agents = [agent]

    visited = set()

    for _ in range(steps):
        obs = env.observacaoPara(agent)
        agent.observacao(obs)

        visited.add((agent.x, agent.y))

        accao = agent.age()
        env.agir(accao, agent)

        env.atualizacao()

        if agent.reached_goal:
            break

    return visited, agent.reached_goal


# ============================================================
# TRAINING LOOP (PURE NOVELTY SEARCH)
# ============================================================

def train_maze(map_file=MAP_FILE):
    """
    Pure Novelty Search evolution of MazeLearningAgent on a fixed maze map.
    Returns a history dict with stats and saves best genotype to disk.
    """

    # Load base map once
    base_env, start_positions, goals, obstacles = load_fixed_map(map_file)
    start_A = tuple(start_positions["A"])
    goals = [tuple(g) for g in goals]

    # Initialize population
    population = [random_genotype() for _ in range(POP_SIZE)]

    archive = []
    best_genotype = None
    best_novelty_ever = -1.0

    history_best_novelty = []
    history_mean_novelty = []
    history_archive_size = []
    history_goal_count = []

    for gen in range(GENERATIONS):
        print(f"\n===== MAZE GENERATION {gen+1}/{GENERATIONS} =====")

        behaviors = []
        reached_flags = []

        # --- Evaluate all individuals: produce behaviors (visited sets) ---
        for i, genes in enumerate(population):
            visited, reached = simulate_genotype(
                genes, base_env, start_A, goals, obstacles, steps=STEPS_PER_EPISODE
            )
            behaviors.append(visited)
            reached_flags.append(reached)

        # --- Compute novelty for each behavior ---
        novelties = compute_novelty(behaviors, archive, k=K_NEIGHBORS)

        # --- Update archive with most novel behaviors ---
        archive = update_archive(archive, behaviors, novelties)

        # --- Stats & logging ---
        best_nov = max(novelties)
        mean_nov = sum(novelties) / len(novelties)
        goals_reached = sum(1 for r in reached_flags if r)

        history_best_novelty.append(best_nov)
        history_mean_novelty.append(mean_nov)
        history_archive_size.append(len(archive))
        history_goal_count.append(goals_reached)

        print(f"Best novelty: {best_nov:.3f} | Mean novelty: {mean_nov:.3f} | "
              f"Goals reached this gen: {goals_reached} | Archive size: {len(archive)}")

        # track best-ever individual (by novelty)
        best_index = max(range(len(population)), key=lambda i: novelties[i])
        if novelties[best_index] > best_novelty_ever or best_genotype is None:
            best_genotype = population[best_index][:]
            best_novelty_ever = novelties[best_index]

        # --- Selection (pure novelty) ---
        indices = list(range(len(population)))
        indices.sort(key=lambda i: novelties[i], reverse=True)

        n_elite = max(1, int(ELITE_FRACTION * POP_SIZE))
        elites = [population[i] for i in indices[:n_elite]]

        # --- Reproduction ---
        new_population = elites[:]  # keep elites unchanged

        while len(new_population) < POP_SIZE:
            p1 = random.choice(elites)
            p2 = random.choice(elites)
            child = crossover_genotypes(p1, p2)
            child = mutate_genotype(child)
            new_population.append(child)

        population = new_population

    # Save best genotype found
    if best_genotype is not None:
        save_best_genotype(best_genotype)

    history = {
        "best_novelty": history_best_novelty,
        "mean_novelty": history_mean_novelty,
        "archive_size": history_archive_size,
        "goal_count": history_goal_count,
    }

    return history


# ============================================================
# SAVE / LOAD BEST GENOTYPE
# ============================================================

def save_best_genotype(genes, filename=BEST_GENOTYPE_FILE):
    with open(filename, "w") as f:
        json.dump(genes, f)


def load_best_genotype(filename=BEST_GENOTYPE_FILE):
    with open(filename, "r") as f:
        return json.load(f)


# ============================================================
# PLOTS
# ============================================================

def plot_novelty(history):
    gens = range(1, len(history["best_novelty"]) + 1)

    plt.figure(figsize=(10, 6))

    # Best & mean novelty
    plt.subplot(2, 1, 1)
    plt.plot(gens, history["best_novelty"], label="Best novelty")
    plt.plot(gens, history["mean_novelty"], label="Mean novelty")
    plt.ylabel("Novelty")
    plt.legend()
    plt.grid(True)

    # Archive size + goals reached
    plt.subplot(2, 1, 2)
    plt.plot(gens, history["archive_size"], label="Archive size")
    plt.plot(gens, history["goal_count"], label="Goals per generation")
    plt.xlabel("Generation")
    plt.ylabel("Count")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()


# ============================================================
# BUILD TRAINED AGENT FOR MAIN.PY
# ============================================================

def load_trained_maze_agent(map_file=MAP_FILE):
    """
    Load best genotype from disk, create a MazeLearningAgent in an env,
    and return (env, [agent]) ready for MotorDeSimulacao.
    """
    base_env, start_positions, goals, obstacles = load_fixed_map(map_file)
    start_A = tuple(start_positions["A"])
    goals = [tuple(g) for g in goals]

    genes = load_best_genotype()

    env = World(
        height=base_env.height,
        width=base_env.width,
        goals=goals,
        obstacles=obstacles,
        mode="maze"
    )

    agent = MazeLearningAgent("EVO", env, start_pos=start_A, weights=genes)
    agent.set_mode("test")
    env.agents = [agent]

    return env, [agent]


# ============================================================
# OPTIONAL: standalone run
# ============================================================

if __name__ == "__main__":
    hist = train_maze(MAP_FILE)
    plot_novelty(hist)
    # For full visual test, better run via Main.py using MotorDeSimulacao.
