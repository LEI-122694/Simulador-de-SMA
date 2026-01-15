# Training/TrainQLearningMaze.py
import matplotlib.pyplot as plt
import Config as C

from Learning.Brains.QLearningBrain import QLearningBrain
from Learning.Adapters.MazeAdapter import MazeAdapter
from Agents.LearningAgent import LearningAgent
from Environments.Maze import load_fixed_map

EPISODES  = C.Q_EPISODES
MAX_STEPS = C.Q_MAX_STEPS

ALPHA   = C.Q_ALPHA
GAMMA   = C.Q_GAMMA
EPSILON = C.Q_EPSILON


def plot_learning_curve(rewards, title="Learning Curve — Maze (Q-learning)"):
    plt.figure(figsize=(10, 4))
    plt.plot(rewards)
    plt.xlabel("Episode")
    plt.ylabel("Total reward")
    plt.title(title)
    plt.grid(True)
    plt.show()


def train_qlearning_maze(map_file: str, out_policy: str = None, plot: bool = True):
    # IMPORTANT: include position for Q-learning (avoids state aliasing)
    adapter = MazeAdapter(include_position=True)
    brain = QLearningBrain(alpha=ALPHA, gamma=GAMMA, epsilon=EPSILON)

    episode_rewards = []

    for ep in range(EPISODES):
        env, start_positions, _, _ = load_fixed_map(map_file)
        start_pos = tuple(start_positions["A"])

        agent = LearningAgent("QL", env, start_pos, adapter, brain)
        agent.set_mode("train")
        env.agents = [agent]

        total_reward = 0.0

        for step in range(1, MAX_STEPS + 1):
            obs = env.observacaoPara(agent)
            agent.observacao(obs)

            valid = adapter.valid_actions(agent, env, obs)
            if not valid:
                break

            move = agent.age()
            env.agir(move, agent)
            env.atualizacao()

            obs2 = env.observacaoPara(agent)
            agent.observacao(obs2)

            r = adapter.reward(
                agent,
                agent.prev_state,
                agent.prev_action,
                agent.state,
                obs2,
                step,
                MAX_STEPS
            )
            total_reward += float(r)

            next_valid = adapter.valid_actions(agent, env, obs2)
            brain.update(
                agent.prev_state,
                agent.prev_action,
                r,
                agent.state,
                agent.reached_goal,
                next_valid_actions=next_valid
            )

            if agent.reached_goal:
                break

        episode_rewards.append(total_reward)

        if ep % 50 == 0:
            print(f"[MAZE Q] EP {ep} reached={agent.reached_goal} | total_reward={total_reward:.2f}")

    # save policy
    save_path = C.MAZE_POLICY if out_policy is None else out_policy
    brain.save(save_path)
    print(f"✅ Saved policy to: {save_path}")

    # plot curve
    if plot:
        plot_learning_curve(episode_rewards)

    return save_path, episode_rewards


if __name__ == "__main__":
    train_qlearning_maze(C.MAZE_MAP, plot=True)
