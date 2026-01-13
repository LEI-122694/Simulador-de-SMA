# TrainEvolutionMaze.py

import random
from Learning.Brains.GenomeBrain import GenomeBrain
from Learning.Adapters.MazeAdapter import MazeAdapter
from Agents.LearningAgent import LearningAgent
from Environments.Maze import setup_maze

POP_SIZE = 30
GENERATIONS = 80
MAX_STEPS = 200

def evaluate(env_template, genome):
    env = env_template.clone()
    adapter = MazeAdapter()
    brain = GenomeBrain(genome)

    agent = LearningAgent("EVO", env, start_pos=(0, 0), adapter=adapter, brain=brain)
    env.agents = [agent]

    total_reward = 0

    for step in range(MAX_STEPS):
        obs = env.observacaoPara(agent)
        agent.observacao(obs)

        action = agent.age()
        env.agir(action, agent)
        env.atualizacao()

        obs2 = env.observacaoPara(agent)
        reward = adapter.reward(
            agent,
            agent.prev_state,
            agent.prev_action,
            agent.state,
            obs2,
            step,
            MAX_STEPS
        )

        total_reward += reward

        if agent.reached_goal:
            break

    return total_reward


def train_evolution_maze():
    env, _ = setup_maze(agent_type="learning", map_type="fixed")

    genome_size = GenomeBrain.genome_size()
    population = [
        [random.uniform(-1, 1) for _ in range(genome_size)]
        for _ in range(POP_SIZE)
    ]

    for gen in range(GENERATIONS):
        scored = [(evaluate(env, g), g) for g in population]
        scored.sort(reverse=True, key=lambda x: x[0])

        print(f"[GEN {gen}] best={scored[0][0]:.2f}")

        elites = [g for _, g in scored[:5]]
        population = elites[:]

        while len(population) < POP_SIZE:
            parent = random.choice(elites)
            child = parent[:]
            i = random.randrange(len(child))
            child[i] += random.uniform(-0.2, 0.2)
            population.append(child)


if __name__ == "__main__":
    train_evolution_maze()
