# Training/TrainFarol.py

import os
import matplotlib.pyplot as plt

from Environments.Lighthouse import load_fixed_map
from Agents.LighthouseLearningAgent import LighthouseQLearningAgent

# ============================================================
# ABSOLUTE MAP PATH
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
MAP_FILE = os.path.join(BASE_DIR, "Resources", "farol_map_1.json")

# ============================================================
# RL HYPERPARAMETERS
# ============================================================

EPISODES = 200
MAX_STEPS = 250

ALPHA = 0.3
GAMMA = 0.95

EPSILON_START = 0.4
EPSILON_DECAY = 0.90
MIN_EPSILON = 0.01


# ============================================================
# RL TRAINING ENGINE (MULTI-AGENT)
# ============================================================

class FarolRLEngine:
    def __init__(self, env, agents, max_steps=200):
        self.env = env
        self.agents = agents
        self.max_steps = max_steps

    def run_episode(self):
        total_reward = 0.0

        for step in range(1, self.max_steps + 1):
            for agent in self.agents:
                # 1. Observation
                obs = self.env.observacaoPara(agent)
                agent.observacao(obs)

                # 2. Action
                action = agent.age()

                # 3. Apply action
                self.env.agir(action, agent)

                # 4. Reward
                reward = agent.calcula_recompensa(
                    estado_anterior=agent.last_state,
                    acao=agent.last_action,
                    estado_atual=agent.current_state,
                    passo_atual=step,
                    max_steps=self.max_steps
                )

                # Q-update
                agent.avaliacaoEstadoAtual(reward)
                total_reward += reward

            self.env.atualizacao()

            # Stop episode if any agent reaches the lighthouse
            if any(agent.reached_goal for agent in self.agents):
                break

        return total_reward


# ============================================================
# TRAINING LOOP
# ============================================================

def train_farol(map_file=MAP_FILE):
    episode_rewards = []
    shared_Q = {}   # <<<<<< SHARED Q-TABLE

    epsilon = EPSILON_START

    for ep in range(EPISODES):
        print(f"\n===== EPISÓDIO {ep + 1}/{EPISODES} =====")

        # -------------------------------------------
        # Load fixed map
        # -------------------------------------------
        env, start_positions, goal, obstacles = load_fixed_map(map_file)

        start_A = tuple(start_positions["A"])
        start_B = tuple(start_positions["B"])

        # -------------------------------------------
        # Two cooperative learning agents
        # -------------------------------------------
        agent_A = LighthouseQLearningAgent(
            name="A",
            env=env,
            start_pos=start_A,
            alpha=ALPHA,
            gamma=GAMMA,
            epsilon=epsilon,
            q_table=shared_Q
        )

        agent_B = LighthouseQLearningAgent(
            name="B",
            env=env,
            start_pos=start_B,
            alpha=ALPHA,
            gamma=GAMMA,
            epsilon=epsilon,
            q_table=shared_Q
        )

        agent_A.set_mode("train")
        agent_B.set_mode("train")

        env.agents = [agent_A, agent_B]

        # -------------------------------------------
        # Execute training episode
        # -------------------------------------------
        engine = FarolRLEngine(env, env.agents, MAX_STEPS)
        total_reward = engine.run_episode()
        episode_rewards.append(total_reward)

        print(f"Recompensa total: {total_reward:.1f} | epsilon: {epsilon:.2f}")

        # -------------------------------------------
        # Epsilon decay
        # -------------------------------------------
        epsilon = max(MIN_EPSILON, epsilon * EPSILON_DECAY)

        # -------------------------------------------
        # Reset agents internal state
        # -------------------------------------------
        for agent in env.agents:
            agent.last_state = None
            agent.last_action = None
            agent.reached_goal = False

        # -------------------------------------------
        # Save learned policy
        # -------------------------------------------
        agent_A.save_policy(os.path.join(BASE_DIR, "policy_farol.json"))

    return episode_rewards


# ============================================================
# LEARNING CURVE PLOT
# ============================================================

def plot_learning_curve(rewards):
    plt.figure(figsize=(9, 5))
    plt.plot(rewards)
    plt.title("Curva de Aprendizagem — Farol (Q-learning, 2 agentes)")
    plt.xlabel("Episódio")
    plt.ylabel("Recompensa total")
    plt.grid(True)

    plt.show(block=False)
    plt.pause(2)
    plt.close()


# ============================================================
# LOAD TRAINED AGENT (UNCHANGED)
# ============================================================

def load_trained_agent(map_file=MAP_FILE):
    env, start_positions, goal, obstacles = load_fixed_map(map_file)
    start_A = tuple(start_positions["A"])

    rl = LighthouseQLearningAgent("RL", env, start_A)
    rl.load_policy(os.path.join(BASE_DIR, "policy_farol.json"))
    rl.set_mode("test")

    env.agents = [rl]
    return env, [rl]


# ============================================================
# EXECUTE
# ============================================================

if __name__ == "__main__":
    rewards = train_farol(MAP_FILE)
    plot_learning_curve(rewards)
    load_trained_agent(MAP_FILE)
