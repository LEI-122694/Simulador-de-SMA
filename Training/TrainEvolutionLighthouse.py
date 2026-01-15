# Training/TrainEvolutionLighthouse.py
import random
import math
import matplotlib.pyplot as plt
import Config as C

from Learning.Brains.GenomeBrain import GenomeBrain
from Learning.Adapters.FarolAdapter import FarolAdapter
from Agents.LearningAgent import LearningAgent
from Environments.Lighthouse import load_fixed_map

POP_SIZE        = C.EVO_POP_SIZE
GENERATIONS     = C.EVO_GENERATIONS
STEPS_PER_AGENT = C.EVO_STEPS_PER_AGENT

K_NEIGHBORS     = C.K_NEIGHBORS
MUTATION_RATE   = C.EVO_MUTATION_RATE
MUTATION_STD    = C.EVO_MUTATION_STD

ARCHIVE_ADD_TOP = C.ARCHIVE_ADD_TOP
ALPHA           = C.NOVELTY_ALPHA

PARENTS         = C.EVO_PARENTS
ELITE           = C.EVO_ELITE
HIDDEN          = C.EVO_HIDDEN


def fitness_of(agent, steps, max_steps):
    if agent.reached_goal:
        return 50.0 + (max_steps - steps)
    return -steps * 0.1


def behaviour_descriptor(agent):
    return (float(agent.x), float(agent.y))


def euclidean(a, b):
    return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)


def novelty_of(desc, neighbours, k=K_NEIGHBORS):
    if not neighbours:
        return 0.0
    dists = [euclidean(desc, other) for other in neighbours]
    dists.sort()
    k_eff = min(k, len(dists))
    return sum(dists[:k_eff]) / float(k_eff)


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
        hidden=HIDDEN,
        outputs=adapter.action_size(),
        action_order=adapter.ACTIONS
    )
    brain.reset()

    agent = LearningAgent("EVO", env, start_pos, adapter, brain)
    agent.set_mode("test")
    env.agents = [agent]

    steps_used = STEPS_PER_AGENT
    for step in range(1, STEPS_PER_AGENT + 1):
        obs = env.observacaoPara(agent)
        agent.observacao(obs)

        move = agent.age()
        env.agir(move, agent)
        env.atualizacao()

        obs2 = env.observacaoPara(agent)
        agent.observacao(obs2)

        steps_used = step
        if agent.reached_goal:
            break

    desc = behaviour_descriptor(agent)
    fit = fitness_of(agent, steps_used, STEPS_PER_AGENT)
    return desc, agent.reached_goal, fit


def train_evolution_farol(map_file: str):
    template_env, start_positions, _, _ = load_fixed_map(map_file)
    start_pos = tuple(start_positions["A"])
    adapter = FarolAdapter()

    genome_size = GenomeBrain.genome_size(
        inputs=adapter.observation_size(),
        hidden=HIDDEN,
        outputs=adapter.action_size()
    )

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
        print(f"\n===== FAROL GENERATION {gen+1}/{GENERATIONS} =====")

        behaviours, reached_flags, fitnesses = [], [], []

        for genome in population:
            desc, reached, fit = evaluate_individual(template_env, genome, start_pos, adapter)
            behaviours.append(desc)
            reached_flags.append(reached)
            fitnesses.append(fit)

        novelties = []
        for i, desc in enumerate(behaviours):
            others = archive + [behaviours[j] for j in range(len(behaviours)) if j != i]
            novelties.append(novelty_of(desc, others, K_NEIGHBORS))

        hybrid_scores = [
            ALPHA * novelties[i] + (1.0 - ALPHA) * fitnesses[i]
            for i in range(len(population))
        ]

        best_n = max(novelties)
        mean_n = sum(novelties) / len(novelties)
        reached = sum(1 for r in reached_flags if r)

        best_novels.append(best_n)
        mean_novels.append(mean_n)
        reached_per_gen.append(reached)

        print(f"  best novelty={best_n:.3f} | mean novelty={mean_n:.3f} | reached={reached}/{POP_SIZE} | archive={len(archive)}")

        best_idx_gen = max(range(len(population)), key=lambda i: hybrid_scores[i])
        if hybrid_scores[best_idx_gen] > best_overall_hybrid:
            best_overall_hybrid = hybrid_scores[best_idx_gen]
            best_overall_genome = population[best_idx_gen][:]

        # archive update: top-N novelty
        sorted_idx_novel = sorted(range(len(population)), key=lambda i: novelties[i], reverse=True)
        for i in sorted_idx_novel[:ARCHIVE_ADD_TOP]:
            archive.append(behaviours[i])

        # selection by hybrid
        sorted_idx = sorted(range(len(population)), key=lambda i: hybrid_scores[i], reverse=True)
        parents = [population[i] for i in sorted_idx[:PARENTS]]

        new_population = [population[i][:] for i in sorted_idx[:ELITE]]
        while len(new_population) < POP_SIZE:
            p = random.choice(parents)
            new_population.append(mutate(p))

        population = new_population

    with open(C.FAROL_GENOME, "w") as f:
        f.write(",".join(str(x) for x in best_overall_genome))

    print(f"\n✅ Saved best genome to: {C.FAROL_GENOME}")
    return best_novels, mean_novels, archive, reached_per_gen, C.FAROL_GENOME


def plot_novelty(best, mean, reached_per_gen):
    plt.figure(figsize=(10, 5))
    plt.plot(best, label="Best novelty", linewidth=2)
    plt.plot(mean, label="Mean novelty", linestyle="--")
    plt.plot(reached_per_gen, label="Reached goal", linestyle=":", linewidth=2)
    plt.xlabel("Generation")
    plt.ylabel("Value")
    plt.title("Farol – Novelty Search + Reached Goal")
    plt.grid(True)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    best, mean, archive, reached, _ = train_evolution_farol(C.FAROL_MAP)
    plot_novelty(best, mean, reached)
