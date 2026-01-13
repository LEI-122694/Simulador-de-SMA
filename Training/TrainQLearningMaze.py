# TrainQLearningMaze.py

from Learning.brains.QLearningBrain import QLearningBrain
from Learning.adapters.MazeAdapter import MazeAdapter
from Agents.LearningAgent import LearningAgent
from Environments.Maze import setup_maze

EPISODES = 500
MAX_STEPS = 300

def train_qlearning_maze():
    env, agents = setup_maze(agent_type="learning", map_type="fixed")

    adapter = MazeAdapter()
    brain = QLearningBrain()

    agent = agents[0]
    agent.brain = brain
    agent.adapter = adapter
    agent.set_mode("train")

    env.agents = [agent]

    for ep in range(EPISODES):
        env.step_count = 0
        agent.reached_goal = False

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

            agent.avaliacaoEstadoAtual(reward)

            if agent.reached_goal:
                break

        if ep % 50 == 0:
            print(f"[EP {ep}] reached={agent.reached_goal}")

    brain.save("policy_maze.json")


if __name__ == "__main__":
    train_qlearning_maze()
