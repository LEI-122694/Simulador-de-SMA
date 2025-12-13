# Training/TrainFarol.py

import os
import matplotlib.pyplot as plt

from Environments.Lighthouse import load_fixed_map
from Agents.LighthouseLearningAgent import LighthouseQLearningAgent

# ============================================================
# ABSOLUTE MAP PATH (NO MORE FILE NOT FOUND)
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
# RL TRAINING ENGINE
# ============================================================

class FarolRLEngine:
    def __init__(self, env, agent, max_steps=200):
        self.env = env
        self.agent = agent
        self.max_steps = max_steps

    def run_episode(self):
        total_reward = 0.0
        goal = next(iter(self.env.goals))

        for step in range(1, self.max_steps + 1):
            # 1. Observation
            obs = self.env.observacaoPara(self.agent)
            self.agent.observacao(obs)

            # 2. Action
            action = self.agent.age()

            # 3. Apply action
            self.env.agir(action, self.agent)

            # 4. Reward
            reward = self.agent.calcula_recompensa(
                estado_anterior=self.agent.last_state,
                acao=self.agent.last_action,
                estado_atual=self.agent.current_state,
                passo_atual=step,
                max_steps=self.max_steps
            )

            # Q-update
            self.agent.avaliacaoEstadoAtual(reward)
            total_reward += reward

            self.env.atualizacao()

            if self.agent.reached_goal:
                break

        return total_reward


# ============================================================
# TRAINING LOOP
# ============================================================

def train_farol(map_file=MAP_FILE):
    episode_rewards = []
    shared_Q = {}

    epsilon = EPSILON_START

    for ep in range(EPISODES):
        print(f"\n===== EPISÓDIO {ep + 1}/{EPISODES} =====")

        # -------------------------------------------
        # Load map (using colleague's load_fixed_map)
        # -------------------------------------------
        env, start_positions, goal, obstacles = load_fixed_map(map_file)

        start_A = tuple(start_positions["A"])

        # Insert RL agent manually
        rl = LighthouseQLearningAgent(
            name="RL",
            env=env,
            start_pos=start_A,
            alpha=ALPHA,
            gamma=GAMMA,
            epsilon=epsilon,
            q_table=shared_Q
        )
        rl.set_mode("train")

        env.agents = [rl]

        # Execute training episode
        engine = FarolRLEngine(env, rl, MAX_STEPS)
        total_reward = engine.run_episode()
        episode_rewards.append(total_reward)

        print(f"Recompensa: {total_reward:.1f} | epsilon: {epsilon:.2f}")

        # Epsilon decay
        epsilon = max(MIN_EPSILON, epsilon * EPSILON_DECAY)

        # Reset internal RL state
        rl.last_state = None
        rl.last_action = None
        rl.reached_goal = False

        # Save learned Q-table
        rl.save_policy(os.path.join(BASE_DIR, "policy_farol.json"))

    return episode_rewards


# ============================================================
# LEARNING CURVE PLOT
# ============================================================

def plot_learning_curve(rewards):
    plt.figure(figsize=(9, 5))
    plt.plot(rewards)
    plt.title("Curva de Aprendizagem — Farol (Q-learning)")
    plt.xlabel("Episódio")
    plt.ylabel("Recompensa total")
    plt.grid(True)

    plt.show(block=False)  # <<<<<<<<<< FIX
    plt.pause(2)            # show for 2 seconds
    plt.close()             # then auto-close


# ============================================================
# TEST TRAINED AGENT ON MAP
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
