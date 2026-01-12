import os
import numpy as np
import matplotlib.pyplot as plt

from Environments.Lighthouse import load_fixed_map
from Agents.Fixed.LighthouseFixedAgent import LighthouseFixedAgent
from Agents.LighthouseLearningAgent import LighthouseQLearningAgent

# ==================================================
# CONFIGURATION
# ==================================================
FIXED_RUNS = 30          # stochastic baseline
MAX_STEPS = 200

BASE_DIR  = os.path.dirname(os.path.dirname(__file__))
MAP_FILE = os.path.join(BASE_DIR, "Resources", "farol_map_2.json")
POLICY   = os.path.join(BASE_DIR, "policy_farol.json")


# ==================================================
def run_episode(agent):
    env = agent.env

    for step in range(1, MAX_STEPS + 1):
        obs = env.observacaoPara(agent)
        agent.observacao(obs)

        action = agent.age()
        env.agir(action, agent)
        env.atualizacao()

        if agent.reached_goal:
            return step

    return MAX_STEPS


# ==================================================
def evaluate_fixed():
    steps = []

    for _ in range(FIXED_RUNS):
        env, starts, _, _ = load_fixed_map(MAP_FILE)
        agent = LighthouseFixedAgent("FIXED", env, tuple(starts["A"]))
        agent.set_mode("test")
        env.agents = [agent]

        steps.append(run_episode(agent))

    return steps


# ==================================================
def evaluate_learned_once():
    env, starts, _, _ = load_fixed_map(MAP_FILE)
    agent = LighthouseQLearningAgent("RL", env, tuple(starts["A"]))
    agent.load_policy(POLICY)
    agent.set_mode("test")
    env.agents = [agent]

    return run_episode(agent)


# ==================================================
if __name__ == "__main__":

    fixed_steps   = evaluate_fixed()
    learned_steps = evaluate_learned_once()

    fixed_mean = float(np.mean(fixed_steps))
    fixed_std  = float(np.std(fixed_steps))

    print("\n===== FAROL FINAL EVALUATION =====")
    print(f"Fixed agent   → average steps = {fixed_mean:.1f} ({FIXED_RUNS} runs)")
    print(f"Learned agent → steps = {learned_steps}")

    # ------------------------------------------------
    # BAR GRAPH ONLY (NO BOXPLOT)
    # ------------------------------------------------
    labels = ["Fixed (avg)", "Learned (1 run)"]
    values = [fixed_mean, learned_steps]
    errors = [fixed_std, 0.0]

    plt.figure()
    plt.bar(labels, values)
    plt.ylabel("Steps to reach lighthouse")
    plt.title("Farol — Fixed Baseline vs Learned Policy (Test Mode)")
    plt.grid(True, axis="y")
    plt.show()
