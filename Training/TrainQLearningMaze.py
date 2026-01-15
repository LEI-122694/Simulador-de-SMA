# Training/TrainQLearningMaze.py
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


def train_qlearning_maze(map_file: str, out_policy: str = None):
    # IMPORTANT: include position for Q-learning (avoids state aliasing)
    adapter = MazeAdapter(include_position=True)
    brain = QLearningBrain(alpha=ALPHA, gamma=GAMMA, epsilon=EPSILON)

    for ep in range(EPISODES):
        env, start_positions, _, _ = load_fixed_map(map_file)
        start_pos = tuple(start_positions["A"])

        agent = LearningAgent("QL", env, start_pos, adapter, brain)
        agent.set_mode("train")
        env.agents = [agent]

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

        if ep % 50 == 0:
            print(f"[MAZE Q] EP {ep} reached={agent.reached_goal}")

    # ✅ correct save path: policy file (NOT the map!)
    save_path = C.MAZE_POLICY if out_policy is None else out_policy
    brain.save(save_path)
    print(f"✅ Saved policy to: {save_path}")
    return save_path


if __name__ == "__main__":
    train_qlearning_maze(C.MAZE_MAP)
