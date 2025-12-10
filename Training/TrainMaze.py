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

POP_SIZE        = 40
GENERATIONS     = 150
STEPS_PER_AGENT = 100

K_NEIGHBORS     = 10
MUTATION_RATE   = 0.10
MUTATION_STD    = 0.60

ARCHIVE_ADD_TOP = 5

ALPHA = 0.05  # peso da novelty vs fitness

# ============================================================
# BEHAVIOUR CHARACTERISATION
# ============================================================

def behaviour_vector(env, agent):
    return np.array([agent.x, agent.y], dtype=float)

# ============================================================
# FITNESS (recompensa chegar ao objetivo + penalização por ciclos)
# ============================================================

def fitness_of(agent, visited_cells, steps, goal_reward=20.0, new_cell_reward=0.5, repeat_penalty=0.2):
    """
    Fitness SEM usar distância ao objetivo.
    - Recompensa chegar ao objetivo.
    - Penaliza passos.
    - Penaliza células repetidas e recompensa células novas.
    """
    score = 0.0

    # Penaliza visitas repetidas
    for count in visited_cells.values():
        if count > 1:
            score -= repeat_penalty * (count - 1)

    # Recompensa novas células visitadas
    score += new_cell_reward * len(visited_cells)

    # Bônus por atingir o goal
    if agent.reached_goal:
        score += goal_reward / (steps + 1)
    else:
        score += 1.0 / (steps + 1)

    return score

# ============================================================
# EVALUATION OF ONE GENOME
# ============================================================

def evaluate_individual(template_env, genome, start_pos, goal):
    env = template_env.clone()
    agent = MazeLearningAgent("EVO", env, start_pos, genome)
    env.agents = [agent]

    visited_cells = {}  # (x, y) -> count
    steps_taken = 0

    for _ in range(STEPS_PER_AGENT):
        obs = env.observacaoPara(agent)
        agent.observacao(obs)
        action = agent.age()
        env.agir(action, agent)
        env.atualizacao()
        steps_taken += 1

        pos = (agent.x, agent.y)
        visited_cells[pos] = visited_cells.get(pos, 0) + 1

        if agent.reached_goal:
            break

    bv = behaviour_vector(env, agent)
    return bv, agent.reached_goal, agent, steps_taken, visited_cells

# ============================================================
# NOVELTY METRIC
# ============================================================

def novelty_of(bv, neighbours, k=K_NEIGHBORS):
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
    template_env, start_positions, goals, _ = load_fixed_map(map_file)
    start_pos = tuple(start_positions["A"])
    goal = goals[0]

    genome_size = MazeLearningAgent.genome_size()
    def random_genome():
        return [random.uniform(-1, 1) for _ in range(genome_size)]

    population = [random_genome() for _ in range(POP_SIZE)]
    archive = []

    best_novels = []
    mean_novels = []
    reached_per_gen = []

    best_overall_genome = None
    best_overall_hybrid = -math.inf

    for gen in range(GENERATIONS):
        print(f"\n===== GENERATION {gen+1}/{GENERATIONS} =====")

        behaviours = []
        reached_flags = []
        agents_list = []
        steps_list = []
        visited_list = []

        # Evaluate genomes
        for genome in population:
            bv, reached, agent, steps, visited_cells = evaluate_individual(template_env, genome, start_pos, goal)
            behaviours.append(bv)
            reached_flags.append(reached)
            agents_list.append(agent)
            steps_list.append(steps)
            visited_list.append(visited_cells)

        # Compute novelty
        novelties = []
        for i, bv in enumerate(behaviours):
            others = archive + [behaviours[j] for j in range(len(behaviours)) if j != i]
            nov = novelty_of(bv, others, K_NEIGHBORS)
            novelties.append(nov)

        # Hybrid score (novelty + fitness)
        hybrid_scores = []
        for i, agent in enumerate(agents_list):
            fit = fitness_of(agent, visited_list[i], steps_list[i])
            hybrid = ALPHA * novelties[i] + (1 - ALPHA) * fit
            hybrid_scores.append(hybrid)

        # Stats
        best_n = max(novelties)
        mean_n = sum(novelties) / len(novelties)
        best_novels.append(best_n)
        mean_novels.append(mean_n)
        reached_per_gen.append(sum(1 for f in reached_flags if f))

        print(f"  best novelty = {best_n:.3f} | mean = {mean_n:.3f} | reached goal = {reached_per_gen[-1]}")

        # Track best overall
        best_idx_gen = max(range(len(population)), key=lambda i: hybrid_scores[i])
        if hybrid_scores[best_idx_gen] > best_overall_hybrid:
            best_overall_hybrid = hybrid_scores[best_idx_gen]
            best_overall_genome = population[best_idx_gen][:]

        # Archive update
        sorted_idx = sorted(range(len(population)), key=lambda i: novelties[i], reverse=True)
        for i in sorted_idx[:ARCHIVE_ADD_TOP]:
            archive.append(behaviours[i])
        print(f"  archive size = {len(archive)}")

        # Selection & reproduction
        sorted_parents = sorted(range(len(population)), key=lambda i: hybrid_scores[i], reverse=True)
        PARENTS = 10
        parents = [population[i] for i in sorted_parents[:PARENTS]]

        ELITE = 10
        new_population = [population[i][:] for i in sorted_parents[:ELITE]]

        while len(new_population) < POP_SIZE:
            p = random.choice(parents)
            c = mutate_genome(p)
            new_population.append(c)

        population = new_population

    # Save best genome
    base_dir = os.path.dirname(os.path.dirname(__file__))
    save_path = os.path.join(base_dir, "maze_best_genome.txt")
    with open(save_path, "w") as f:
        f.write(",".join(str(x) for x in best_overall_genome))
    print(f"\n✅ Saved best genome to: {save_path}")

    return best_novels, mean_novels, archive, goals, reached_per_gen

# ============================================================
# PLOT NOVELTY CURVE + Reached Goals
# ============================================================

def plot_novelty(best, mean, archive, goals, reached_per_gen):
    plt.figure(figsize=(10, 5))
    plt.plot(best, label="Best novelty", linewidth=2)
    plt.plot(mean, label="Mean novelty", linestyle="--")
    plt.plot(reached_per_gen, label="Reached goal", linestyle=":", linewidth=2, color="green")
    plt.xlabel("Generation")
    plt.ylabel("Value")
    plt.title("Maze – Novelty Search + Reached Goal")
    plt.grid(True)
    plt.legend()
    plt.show()

# ============================================================
# LOAD TRAINED MAZE AGENT
# ============================================================

def load_trained_maze_agent(map_file):
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
