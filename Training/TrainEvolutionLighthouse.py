# Training/TrainEvolutionLighthouse.py
import os
import random
import math
import matplotlib.pyplot as plt

from Learning.Brains.GenomeBrain import GenomeBrain
from Learning.Adapters.FarolAdapter import FarolAdapter
from Agents.LearningAgent import LearningAgent
from Environments.Lighthouse import load_fixed_map

POP_SIZE        = 40
GENERATIONS     = 80
STEPS_PER_AGENT = 200

MUTATION_RATE   = 0.15
MUTATION_STD    = 0.5
ELITE_SIZE      = 8


def fitness_of(agent, steps, max_steps):
    if agent.reached_goal:
        return 50.0 + (max_steps - steps)
    return -steps * 0.1


def mutate(genome):
    child = []
    for g in genome:
        if random.random() < MUTATION_RATE:
            g += random.gauss(0.0, MUTATION_STD)
        child.append(g)
    return child


def evaluate_individual(template_env, genome, start_pos, adapter):
    env = template_env.clone()

    brain = GenomeBrain(
        genome=genome,
        inputs=adapter.observation_size(),
        hidden=6,
        outputs=adapter.action_size(),
        action_order=adapter.ACTIONS
    )
    brain.reset()

    agent = LearningAgent("EVO", env, start_pos, adapter, brain)
    agent.set_mode("test")
    env.agents = [agent]

    steps = 0

    for step in range(1, STEPS_PER_AGENT + 1):
        obs = env.observacaoPara(agent)
        agent.observacao(obs)

        move = agent.age()
        env.agir(move, agent)
        env.atualizacao()

        # observe again so terminal is consistent
        obs2 = env.observacaoPara(agent)
        agent.observacao(obs2)

        steps = step
        if agent.reached_goal:
            break

    fit = fitness_of(agent, steps, STEPS_PER_AGENT)
    return fit, agent.reached_goal


def train_evolution_farol(map_file: str):
    template_env, start_positions, _, _ = load_fixed_map(map_file)
    start_pos = tuple(start_positions["A"])

    adapter = FarolAdapter()

    genome_size = GenomeBrain.genome_size(
        inputs=adapter.observation_size(),
        hidden=6,
        outputs=adapter.action_size()
    )

    def random_genome():
        return [random.uniform(-1, 1) for _ in range(genome_size)]

    population = [random_genome() for _ in range(POP_SIZE)]

    best_fitness = []
    mean_fitness = []
    reached_list = []

    best_genome = None
    best_score = -math.inf

    for gen in range(GENERATIONS):
        print(f"\n===== FAROL EVO GENERATION {gen+1}/{GENERATIONS} =====")

        scores = []
        reached = 0

        for genome in population:
            fit, ok = evaluate_individual(template_env, genome, start_pos, adapter)
            scores.append((fit, genome))
            if ok:
                reached += 1

        scores.sort(key=lambda x: x[0], reverse=True)

        best = scores[0][0]
        mean = sum(s for s, _ in scores) / len(scores)

        best_fitness.append(best)
        mean_fitness.append(mean)
        reached_list.append(reached)

        print(f"  best={best:.2f} | mean={mean:.2f} | reached={reached}/{POP_SIZE}")

        if best > best_score:
            best_score = best
            best_genome = scores[0][1][:]

        elites = [g for _, g in scores[:ELITE_SIZE]]
        new_population = elites[:]

        while len(new_population) < POP_SIZE:
            parent = random.choice(elites)
            new_population.append(mutate(parent))

        population = new_population

    base_dir = os.path.dirname(os.path.dirname(__file__))
    save_path = os.path.join(base_dir, "farol_best_genome.txt")
    with open(save_path, "w") as f:
        f.write(",".join(str(x) for x in best_genome))

    print(f"\n✅ Best genome saved to: {save_path}")
    return best_fitness, mean_fitness, reached_list, save_path


def plot_evolution(best, mean, reached):
    plt.figure(figsize=(10, 5))
    plt.plot(best, label="Best fitness")
    plt.plot(mean, label="Mean fitness")
    plt.plot(reached, label="Reached goal", linestyle=":")
    plt.legend()
    plt.grid(True)
    plt.title("Evolution – Farol")
    plt.show()


if __name__ == "__main__":
    BASE = os.path.dirname(os.path.dirname(__file__))
    MAP_FILE = os.path.join(BASE, "Resources", "farol_map_2.json")
    best, mean, reached, _ = train_evolution_farol(MAP_FILE)
    plot_evolution(best, mean, reached)
