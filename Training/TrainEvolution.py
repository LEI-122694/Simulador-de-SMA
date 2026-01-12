# TrainEvolution

import random
from Learning.brains.GenomeBrain import GenomeBrain
from Learning.adapters.MazeAdapter import MazeAdapter
from Agents.LearningAgent import LearningAgent
from Environments.Maze import Maze

POP_SIZE = 30
GENERATIONS = 100
MAX_STEPS = 300

env = Maze()
adapter = MazeAdapter()

def evaluate(genome):
    brain = GenomeBrain(genome)
    agent = LearningAgent("EvoAgent", env, env.start_pos, adapter, brain)
    env.reset()

    total_reward = 0
    for step in range(MAX_STEPS):
        obs = env.observe(agent)
        agent.observacao(obs)

        move = agent.age()
        if move:
            env.move_agent(agent, move)

        obs2 = env.observe(agent)
        r = adapter.reward(
            agent,
            agent.prev_state,
            agent.prev_action,
            agent.state,
            obs2,
            step,
            MAX_STEPS
        )
        total_reward += r

        if agent.reached_goal:
            break

    return total_reward

population = [GenomeBrain().genome for _ in range(POP_SIZE)]

for g in range(GENERATIONS):
    scored = [(evaluate(gen), gen) for gen in population]
    scored.sort(reverse=True, key=lambda x: x[0])

    print(f"[GEN {g}] best fitness = {scored[0][0]:.2f}")

    elites = [gen for _, gen in scored[:5]]
    population = elites[:]

    while len(population) < POP_SIZE:
        parent = random.choice(elites)
        child = parent[:]
        i = random.randrange(len(child))
        child[i] += random.uniform(-0.2, 0.2)
        population.append(child)
