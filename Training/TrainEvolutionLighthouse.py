# TrainEvolutionLighthouse.py

import os
import random
import math
import numpy as np
import matplotlib.pyplot as plt

from Learning.Brains.GenomeBrain import GenomeBrain
from Learning.Adapters.FarolAdapter import FarolAdapter
from Agents.LearningAgent import LearningAgent
from Environments.Lighthouse import load_fixed_map


# ============================================================
# HYPERPARAMETERS
# ============================================================

POP_SIZE        = 40
GENERATIONS     = 80
STEPS_PER_AGENT = 200

MUTATION_RATE   = 0.15
MUTATION_STD    = 0.5

ELITE_SIZE      = 8


# ============================================================
# FITNESS FUNCTION (Farol)
# ============================================================

def fitness_of(agent, steps, max_steps):
    """
    Fitness simples:
    - Grande recompensa se chegar ao farol
    - Penalização por passos
    """
    if agent.reached_goal:
        return 50.0 + (max_steps - steps)
    else:
        return -steps * 0.1


# ============================================================
# EVALUATE ONE GENOME
# ============================================================

def evaluate_individual(template_env, genome, start_pos, adapter):
    env = template_env.clone()

    brain = GenomeBrain(genome)
    agent = LearningAgent("EVO", env, start_pos, adapter, brain)
    agent.set_mode("test")

    env.agents = [agent]

    steps = 0

    for step in range(STEPS_PER_AGENT):
        obs = env.observacaoPara(agent)
        agent.observacao(obs)

        action = agent.age()
        env.agir(action, agent)
        env.atualizacao()

        steps += 1

        if agent.reached_goal:
            break

    fit = fitness_of(agent, steps, STEPS_PER_AGENT)
    return fit, agent.reached_goal


# ============================================================
# MUTATION
# ============================================================

def mutate(genome):
    child = []
    for g in genome:
        if random.random() < MUTATION_RATE:
            g += random.gauss(0.0, MUTATION_STD)
        child.append(g)
    return child


# ============================================================
# MAIN TRAINING LOOP
# ============================================================

def train_evolution_farol(map_file):
    env, start_positions, goal, obstacles = load_fixed_map(map_file)
    start_pos = tuple(start_positions["A"])

    adapter = FarolAdapter()

    genome_size = GenomeBrain.genome_size(adapter.observation_size(),
                                          adapter.action_size())

    def random_genome():
        return [random.uniform(-1, 1) for _ in range(genome_size)]

    population = [random_genome() for _ in range(POP_SIZE)]

    best_fitness = []
    mean_fitness = []
    reached_list = []

    best_genome = None
    best_score = -math.inf

    for gen in range(GENERATIONS):
        print(f"\n===== GENERATION {gen+1}/{GENERATIONS} =====")

        scores = []
        reached = 0

        for genome in population:
            fit, goal = evaluate_individual(env, genome, start_pos, adapter)
            scores.append((fit, genome))

            if goal:
                reached += 1

        scores.sort(key=lambda x: x[0], reverse=True)

        best = scores[0][0]
        mean = sum(s for s, _ in scores) / len(scores)

        best_fitness.append(best)
        mean_fitness.append(mean)
        reached_list.append(reached)

        print(f"  best fitness = {best:.2f} | mean = {mean:.2f} | reached = {reached}")

        if best > best_score:
            best_score = best
            best_genome = scores[0][1][:]

        # Selection
        elites = [g for _, g in scores[:ELITE_SIZE]]
        new_population = elites[:]

        while len(new_population) < POP_SIZE:
            parent = random.choice(elites)
            child = mutate(parent)
            new_population.append(child)

        population = new_population

    # --------------------------------------------------------
    # SAVE BEST GENOME
    # --------------------------------------------------------
    base_dir = os.path.dirname(os.path.dirname(__file__))
    save_path = os.path.join(base_dir, "farol_best_genome.txt")

    with open(save_path, "w") as f:
        f.write(",".join(str(x) for x in best_genome))

    print(f"\n✅ Best genome saved to: {save_path}")

    return best_fitness, mean_fitness, reached_list


# ============================================================
# PLOT
# ============================================================

def plot_evolution(best, mean, reached):
    plt.figure(figsize=(10, 5))
    plt.plot(best, label="Best fitness")
    plt.plot(mean, label="Mean fitness")
    plt.plot(reached, label="Reached goal", linestyle=":")
    plt.legend()
    plt.grid(True)
    plt.title("Evolution – Farol")
    plt.show()
