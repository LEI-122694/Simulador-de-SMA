import os
import numpy as np
import matplotlib.pyplot as plt

from Environments.Maze import load_fixed_map
from Agents.Fixed.MazeFixedAgent import MazeFixedAgent
from Agents.MazeLearningAgent import MazeLearningAgent

# ==================================================
# CONFIG (match training budget!)
# ==================================================
FIXED_RUNS = 30
STEPS_PER_AGENT = 100  # must match TrainMaze.py

BASE_DIR   = os.path.dirname(os.path.dirname(__file__))
MAP_FILE   = os.path.join(BASE_DIR, "Resources", "maze_map_1.json")
GENOME_FILE = os.path.join(BASE_DIR, "maze_best_genome.txt")


# ==================================================
def run_episode(agent):
    env = agent.env
    for step in range(1, STEPS_PER_AGENT + 1):
        obs = env.observacaoPara(agent)
        agent.observacao(obs)

        action = agent.age()
        env.agir(action, agent)
        env.atualizacao()

        if agent.reached_goal:
            return step
    return None  # did not reach within budget


# ==================================================
def evaluate_fixed_many():
    """
    Fixed baseline — run many times because it may vary (random.shuffle).
    Returns:
        steps_list: list of ints or STEPS_PER_AGENT for failures
        success_count: how many runs reached the goal
    """
    steps_list = []
    success_count = 0

    for _ in range(FIXED_RUNS):
        env, starts, _, _ = load_fixed_map(MAP_FILE)
        agent = MazeFixedAgent("FIXED", env, tuple(starts["A"]))
        env.agents = [agent]

        steps = run_episode(agent)
        if steps is None:
            steps_list.append(STEPS_PER_AGENT)  # treat fail as max steps
        else:
            steps_list.append(steps)
            success_count += 1

    return steps_list, success_count


# ==================================================
def evaluate_evolved_once():
    """
    Best evolved agent — single run (genome is fixed).
    Returns steps or None if fail.
    """
    if not os.path.exists(GENOME_FILE):
        raise FileNotFoundError(
            "maze_best_genome.txt not found. Run TrainMaze first to generate it."
        )

    with open(GENOME_FILE, "r") as f:
        genome = [float(x) for x in f.read().split(",")]

    env, starts, _, _ = load_fixed_map(MAP_FILE)
    agent = MazeLearningAgent("EVO", env, tuple(starts["A"]), genome)
    env.agents = [agent]

    return run_episode(agent)


# ==================================================
if __name__ == "__main__":

    fixed_steps, fixed_success = evaluate_fixed_many()
    evo_steps = evaluate_evolved_once()

    fixed_avg = float(np.mean(fixed_steps))

    print("\n===== MAZE FINAL EVALUATION =====")
    print(f"Fixed agent   → avg steps = {fixed_avg:.1f} ({FIXED_RUNS} runs), success = {fixed_success}/{FIXED_RUNS}")
    if evo_steps is None:
        print("Evolved agent → did not reach the goal within step budget")
        evo_value = STEPS_PER_AGENT
    else:
        print(f"Evolved agent → steps = {evo_steps}")
        evo_value = evo_steps

    # ------------------------------------------------
    # Simple bar graph (baseline avg vs evolved single)
    # ------------------------------------------------
    plt.figure()
    plt.bar(
        ["Fixed (avg)", "Evolved (1 run)"],
        [fixed_avg, evo_value]
    )
    plt.ylabel("Steps to reach goal (lower is better)")
    plt.title("Maze — Fixed Baseline vs Best Evolved Agent")
    plt.grid(True, axis="y")
    plt.show()
