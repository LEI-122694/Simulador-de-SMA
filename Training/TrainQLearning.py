# TrainQlearning.py

from Learning.brains.QLearningBrain import QLearningBrain
from Learning.adapters.FarolAdapter import FarolAdapter
from Agents.LearningAgent import LearningAgent
from Environments.Lighthouse import Lighthouse

MAX_STEPS = 500
EPISODES = 1000

env = Lighthouse()
adapter = FarolAdapter()
brain = QLearningBrain()

agent = LearningAgent("QL-Agent", env, env.start_pos, adapter, brain)
agent.set_mode("train")

for ep in range(EPISODES):
    env.reset()
    agent.reached_goal = False
    agent.visited_positions = set()

    for step in range(MAX_STEPS):
        obs = env.observe(agent)
        agent.observacao(obs)

        move = agent.age()
        if move:
            env.move_agent(agent, move)

        obs2 = env.observe(agent)
        reward = adapter.reward(
            agent,
            agent.prev_state,
            agent.prev_action,
            agent.state,
            obs2,
            step,
            MAX_STEPS
        )

        agent.avaliacaoEstadoAtual(reward)

        if agent.reached_goal:
            break

    if ep % 100 == 0:
        print(f"[EP {ep}] steps={step}")

brain.save("policy_farol.json")
