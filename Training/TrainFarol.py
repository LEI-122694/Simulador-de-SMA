# Training/TrainFarol.py

import matplotlib.pyplot as plt
import os

from Environments.Lighthouse import setup_lighthouse
from Agents.LighthouseQLearningAgent import LighthouseQLearningAgent
from Main import MotorDeSimulacao   # we reuse your existing simulator engine


# --------------------------
# CONFIG
# --------------------------

# Default map for training + testing (SAME MAP)
# You can change this to any other JSON map later.

# Calculate absolute path to project root
BASE_DIR = os.path.dirname(os.path.dirname(__file__))   # goes up one folder

MAP_FILE = os.path.join(BASE_DIR, "Resources", "farol_map_1.json")

EPISODES = 80
MAX_STEPS = 200

ALPHA = 0.3
GAMMA = 0.95
EPSILON_START = 0.3
EPSILON_DECAY = 0.98
MIN_EPSILON = 0.05


class FarolRLEngine:
    """
    Minimal RL training engine for Farol (single agent).
    Does NOT use MotorDeSimulacao, so the IF-based code remains untouched.
    """

    def __init__(self, env, agent, max_steps=200):
        self.env = env
        self.agent = agent
        self.max_steps = max_steps

    def run_episode(self):
        """
        Run one episode WITHOUT printing the whole map.
        Returns total reward (float).
        """
        total_reward = 0.0
        goal = next(iter(self.env.goals))

        for _ in range(self.max_steps):

            # 1) observation
            obs = self.env.observacaoPara(self.agent)
            self.agent.observacao(obs)

            old_pos = (self.agent.x, self.agent.y)
            dist_old = abs(goal[0] - old_pos[0]) + abs(goal[1] - old_pos[1])

            # 2) choose action
            action = self.agent.age()

            # 3) apply action
            self.env.agir(action, self.agent)

            new_pos = (self.agent.x, self.agent.y)
            dist_new = abs(goal[0] - new_pos[0]) + abs(goal[1] - new_pos[1])

            # -----------------------------
            # Reward Function B (dense shaping)
            # -----------------------------
            if self.agent.reached_goal:
                reward = 100.0
            else:
                if action is None:
                    reward = -5.0   # stuck or no movement
                else:
                    reward = -1.0   # time penalty

                if dist_new < dist_old:
                    reward += 2.0   # moved closer
                elif dist_new > dist_old:
                    reward -= 2.0   # moved away
            # -----------------------------

            self.agent.avaliacaoEstadoAtual(reward)
            total_reward += reward

            self.env.atualizacao()

            if self.agent.reached_goal:
                break

        return total_reward


def train_farol(map_file=MAP_FILE):
    """
    Multi-episode training on ONE fixed map (map_file).
    Training is map-specific but works for ANY map you give it.
    """

    episode_rewards = []
    shared_Q = {}   # shared Q-table across episodes

    epsilon = EPSILON_START

    for ep in range(EPISODES):
        print(f"\n====== EPISÓDIO {ep+1}/{EPISODES} ======")

        # Load SAME map each episode – deterministic layout
        env, rule_agents = setup_lighthouse(
            mode="test",
            json_file=map_file
        )

        # We train ONLY Agent A (first in JSON)
        agent_A = rule_agents[0]
        start_pos = (agent_A.x, agent_A.y)

        # remove rule-based agents from env; we'll insert RL agent only
        env.agents = []

        # RL agent with shared Q-table across episodes
        rl_agent = LighthouseQLearningAgent(
            name="RL",
            env=env,
            start_pos=start_pos,
            alpha=ALPHA,
            gamma=GAMMA,
            epsilon=epsilon,
            q_table=shared_Q,
        )
        rl_agent.set_mode("train")

        engine = FarolRLEngine(env, rl_agent, max_steps=MAX_STEPS)
        total_reward = engine.run_episode()
        episode_rewards.append(total_reward)

        print(f"Recompensa total episódio {ep+1}: {total_reward:.1f}")
        print(f"Epsilon atual: {epsilon:.3f}")

        # Epsilon decay for exploration
        epsilon = max(MIN_EPSILON, epsilon * EPSILON_DECAY)

        # Save policy after each episode (overwrites file)
        policy_file = _policy_filename_for_map(map_file)
        rl_agent.save_policy(policy_file)

    return episode_rewards


def _policy_filename_for_map(map_file):
    """
    Simple helper to give different policy filenames per map if you want.
    Right now it just uses a generic name; you can get fancy if needed.
    """
    return "policy_farol.json"


def plot_learning_curve(rewards):
    plt.figure(figsize=(9, 5))
    plt.plot(rewards)
    plt.title("Curva de Aprendizagem — Farol (Q-learning)")
    plt.xlabel("Episódio")
    plt.ylabel("Recompensa total")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def test_trained_policy(map_file=MAP_FILE):
    """
    After training + plotting, run ONE visual episode using your original
    MotorDeSimulacao, but with the trained RL agent instead of IF logic.

    This is what your prof will see AFTER the graph.
    """

    # Same map as training
    env, rule_agents = setup_lighthouse(
        mode="test",
        json_file=map_file
    )

    # Only Agent A / first start position
    agent_A = rule_agents[0]
    start_pos = (agent_A.x, agent_A.y)

    # remove rule-based agents
    env.agents = []

    rl_agent = LighthouseQLearningAgent("RL", env, start_pos)
    rl_agent.load_policy(_policy_filename_for_map(map_file))
    rl_agent.set_mode("test")   # greedy, no learning

    motor = MotorDeSimulacao(env, [rl_agent])
    motor.executa()


if __name__ == "__main__":
    # 1) Train on MAP_FILE, collect episode rewards
    rewards = train_farol(MAP_FILE)

    # 2) Show learning curve FIRST
    plot_learning_curve(rewards)

    # 3) Then show console map with the trained RL agent on the SAME MAP
    test_trained_policy(MAP_FILE)

