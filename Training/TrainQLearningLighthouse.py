# Training/TrainQLearningLighthouse.py
import os

from Learning.Brains.QLearningBrain import QLearningBrain
from Learning.Adapters.FarolAdapter import FarolAdapter
from Agents.LearningAgent import LearningAgent
from Environments.Lighthouse import load_fixed_map

EPISODES = 500
MAX_STEPS = 300

ALPHA = 0.3
GAMMA = 0.95
EPSILON = 0.2


def train_qlearning_lighthouse(map_file: str, out_policy: str = "policy_farol.json"):
    adapter = FarolAdapter()
    brain = QLearningBrain(alpha=ALPHA, gamma=GAMMA, epsilon=EPSILON)

    for ep in range(EPISODES):
        # Fresh env per episode (same fixed map, but resets agent positions cleanly)
        env, start_positions, _, _ = load_fixed_map(map_file)
        start_pos = tuple(start_positions["A"])

        agent = LearningAgent("QL", env, start_pos, adapter, brain)
        agent.set_mode("train")
        env.agents = [agent]

        for step in range(1, MAX_STEPS + 1):
            # S
            obs = env.observacaoPara(agent)
            agent.observacao(obs)

            valid = adapter.valid_actions(agent, env, obs)
            if not valid:
                break

            # A -> move
            move = agent.age()
            env.agir(move, agent)
            env.atualizacao()

            # S' (must update agent.state!)
            obs2 = env.observacaoPara(agent)
            agent.observacao(obs2)

            # reward(S,A,S')
            r = adapter.reward(
                agent,
                agent.prev_state,
                agent.prev_action,
                agent.state,
                obs2,
                step,
                MAX_STEPS
            )

            # Update with next state's valid actions (more stable)
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
            print(f"[FAROL Q] EP {ep} reached={agent.reached_goal}")

    base_dir = os.path.dirname(os.path.dirname(__file__))
    save_path = os.path.join(base_dir, out_policy)
    brain.save(save_path)
    print(f"✅ Saved policy to: {save_path}")
    return save_path


if __name__ == "__main__":
    BASE = os.path.dirname(os.path.dirname(__file__))
    MAP_FILE = os.path.join(BASE, "Resources", "farol_map_2.json")
    train_qlearning_lighthouse(MAP_FILE)
