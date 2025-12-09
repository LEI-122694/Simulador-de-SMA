# Training/TrainMaze.py

import os
import random
import math

import numpy as np
import matplotlib.pyplot as plt

from Environments.Maze import load_fixed_map
from Agents.MazeLearningAgent import MazeLearningAgent


# ============================================================
# HYPERPARAMETERS
# ============================================================

POP_SIZE        = 80          # number of individuals per generation
GENERATIONS     = 80          # evolutionary generations
STEPS_PER_AGENT = 600         # simulation steps per evaluation

K_NEIGHBORS     = 10          # k for k-NN novelty
MUTATION_RATE   = 0.20        # prob. of mutating each gene
MUTATION_STD    = 0.60        # Gaussian noise

ARCHIVE_ADD_TOP = 5            # how many most-novel individuals to add per gen


# ============================================================
# BEHAVIOUR CHARACTERISATION
# ============================================================

def behaviour_vector(env, agent):
    """
    Behaviour descriptor = final position (x, y),
    exactly as in the maze example from the slides.
    """
    return np.array([agent.x, agent.y], dtype=float)


# ============================================================
# EVALUATION OF ONE GENOME
# ============================================================

def evaluate_individual(template_env, genome, start_pos, goal):
    """
    Run one agent with given genome in a CLONED environment.
    Returns:
        - behaviour vector (final (x,y))
        - reached_goal (bool)
    """
    env = template_env.clone()

    agent = MazeLearningAgent("EVO", env, start_pos, genome)
    env.agents = [agent]

    for _ in range(STEPS_PER_AGENT):
        obs = env.observacaoPara(agent)
        agent.observacao(obs)
        action = agent.age()
        env.agir(action, agent)
        env.atualizacao()

        if agent.reached_goal:
            break

    bv = behaviour_vector(env, agent)
    return bv, agent.reached_goal


# ============================================================
# NOVELTY METRIC
# ============================================================

def novelty_of(bv, neighbours, k=K_NEIGHBORS):
    """
    Novelty = average Euclidean distance to k nearest neighbours
    in behaviour space (archive ∪ current population).
    """
    if len(neighbours) == 0:
        return 0.0

    dists = [np.linalg.norm(bv - other) for other in neighbours]
    dists.sort()
    k_eff = min(k, len(dists))
    return sum(dists[:k_eff]) / float(k_eff)


# ============================================================
# MUTATION
# ============================================================

def mutate_genome(parent):
    child = []
    for g in parent:
        if random.random() < MUTATION_RATE:
            g = g + random.gauss(0.0, MUTATION_STD)
        child.append(g)
    return child


# ============================================================
# MAIN TRAINING LOOP
# ============================================================

def train_maze(map_file):
    """
    Novelty Search evolution for the MAZE task.

    Returns:
        best_novelty_per_gen, mean_novelty_per_gen, archive, goals
    """

    # 1) Load fixed map from JSON
    template_env, start_positions, goals, _ = load_fixed_map(map_file)
    start_pos = tuple(start_positions["A"])
    goal = goals[0]

    # 2) Initial random population of genomes
    genome_size = MazeLearningAgent.genome_size()
    def random_genome():
        return [random.uniform(-1, 1) for _ in range(genome_size)]

    population = [random_genome() for _ in range(POP_SIZE)]

    # Novelty archive: list of behaviour vectors (np.array)
    archive = []

    # For plotting
    best_novels = []
    mean_novels = []

    best_overall_genome = None
    best_overall_novelty = -math.inf

    for gen in range(GENERATIONS):
        print(f"\n===== GENERATION {gen+1}/{GENERATIONS} =====")

        behaviours = []
        reached_flags = []

        # 3) Evaluate each genome
        for idx, genome in enumerate(population):
            bv, reached = evaluate_individual(template_env, genome, start_pos, goal)
            behaviours.append(bv)
            reached_flags.append(reached)

        # 4) Compute novelty scores
        novelties = []
        for i, bv in enumerate(behaviours):
            # neighbours = archive ∪ (population except self)
            others = archive + [behaviours[j] for j in range(len(behaviours)) if j != i]
            nov = novelty_of(bv, others, K_NEIGHBORS)
            novelties.append(nov)

        # 5) Stats
        best_n = max(novelties)
        mean_n = sum(novelties) / len(novelties)

        best_novels.append(best_n)
        mean_novels.append(mean_n)

        num_reached = sum(1 for f in reached_flags if f)
        print(f"  best novelty = {best_n:.3f} | mean = {mean_n:.3f} | reached goal = {num_reached}")

        # Track best overall (by novelty; optionally prefer those that reach goal)
        best_idx_gen = max(range(len(population)), key=lambda i: novelties[i])
        if best_novelties := novelties[best_idx_gen] > best_overall_novelty:
            best_overall_novelty = novelties[best_idx_gen]
            best_overall_genome = population[best_idx_gen][:]

        # 6) Archive update: add top ARCHIVE_ADD_TOP most novel behaviours
        sorted_idx = sorted(range(len(population)),
                            key=lambda i: novelties[i],
                            reverse=True)

        for i in sorted_idx[:ARCHIVE_ADD_TOP]:
            archive.append(behaviours[i])

        print(f"  archive size = {len(archive)}")

        # 7) Selection & reproduction (pure novelty-based)
        PARENTS = max(5, POP_SIZE // 5)
        parents = [population[i] for i in sorted_idx[:PARENTS]]

        # Elitism: copy the top 2 genomes directly
        ELITE = 2
        new_population = [population[i][:] for i in sorted_idx[:ELITE]]

        # Fill the rest via mutation of parents
        while len(new_population) < POP_SIZE:
            p = random.choice(parents)
            c = mutate_genome(p)
            new_population.append(c)

        population = new_population

    # ---------------------------------------------
    # Save best genome to disk
    # ---------------------------------------------
    base_dir = os.path.dirname(os.path.dirname(__file__))
    save_path = os.path.join(base_dir, "maze_best_genome.txt")

    with open(save_path, "w") as f:
        f.write(",".join(str(x) for x in best_overall_genome))

    print(f"\n✅ Saved best genome to: {save_path}")

    return best_novels, mean_novels, archive, goals


# ============================================================
# PLOT NOVELTY CURVE
# ============================================================

def plot_novelty(best, mean, archive, goals):
    """
    Plots best and mean novelty over generations.
    """
    plt.figure(figsize=(9, 5))
    plt.plot(best, label="Best novelty", linewidth=2)
    plt.plot(mean, label="Mean novelty", linestyle="--")
    plt.xlabel("Generation")
    plt.ylabel("Novelty")
    plt.title("Maze – Novelty Search")
    plt.grid(True)
    plt.legend()
    plt.show()


# ============================================================
# LOAD TRAINED MAZE AGENT (BEST GENOME)
# ============================================================

def load_trained_maze_agent(map_file):
    """
    Load best genome from maze_best_genome.txt and create
    a MazeLearningAgent ready to be simulated.
    """
    env, start_positions, goals, obstacles = load_fixed_map(map_file)
    start_pos = tuple(start_positions["A"])

    base_dir = os.path.dirname(os.path.dirname(__file__))
    load_path = os.path.join(base_dir, "maze_best_genome.txt")

    if not os.path.exists(load_path):
        raise FileNotFoundError(
            f"{load_path} not found. Run train_maze() first to create it."
        )

    with open(load_path, "r") as f:
        genome = [float(x) for x in f.read().strip().split(",")]

    agent = MazeLearningAgent("EVO", env, start_pos, genome)
    env.agents = [agent]

    return env, [agent]
