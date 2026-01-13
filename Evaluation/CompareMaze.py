# Evaluation/CompareMaze.py
import os
import numpy as np
import matplotlib.pyplot as plt

from Environments.Maze import load_fixed_map
from Agents.Fixed.MazeFixedAgent import MazeFixedAgent
from Agents.LearningAgent import LearningAgent

from Learning.Brains.QLearningBrain import QLearningBrain
from Learning.Brains.GenomeBrain import GenomeBrain
from Learning.Adapters.MazeAdapter import MazeAdapter

# -----------------------------
RUNS = 30
MAX_STEPS = 200

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
MAP_FILE = os.path.join(BASE_DIR, "Resources", "maze_map_2.json")
POLICY_FILE = os.path.join(BASE_DIR, "policy_maze.json")
GENOME_FILE = os.path.join(BASE_DIR, "maze_best_genome.txt")


def run_episode(env, agent, max_steps=MAX_STEPS):
    env.agents = [agent]
    if hasattr(agent, "episode_reset"):
        agent.episode_reset()

    for step in range(1, max_steps + 1):
        obs = env.observacaoPara(agent)
        agent.observacao(obs)

        move = agent.age()
        env.agir(move, agent)
        env.atualizacao()

        obs2 = env.observacaoPara(agent)
        agent.observacao(obs2)

        if agent.reached_goal:
            return True, step

    return False, max_steps


def eval_fixed():
    steps = []
    success = 0
    for _ in range(RUNS):
        env, starts, _, _ = load_fixed_map(MAP_FILE)
        agent = MazeFixedAgent("FIXED", env, tuple(starts["A"]))
        agent.set_mode("test")
        ok, st = run_episode(env, agent)
        success += int(ok)
        steps.append(st)
    return success, steps


def eval_q():
    adapter = MazeAdapter()
    brain = QLearningBrain()
    brain.load(POLICY_FILE)

    steps = []
    success = 0
    for _ in range(RUNS):
        env, starts, _, _ = load_fixed_map(MAP_FILE)
        agent = LearningAgent("Q", env, tuple(starts["A"]), adapter, brain)
        agent.set_mode("test")
        ok, st = run_episode(env, agent)
        success += int(ok)
        steps.append(st)
    return success, steps


def eval_evo():
    adapter = MazeAdapter()

    if not os.path.exists(GENOME_FILE):
        raise FileNotFoundError(f"Missing {GENOME_FILE}. Train evolution first.")

    with open(GENOME_FILE, "r") as f:
        genome = [float(x) for x in f.read().strip().split(",")]

    steps = []
    success = 0
    for _ in range(RUNS):
        env, starts, _, _ = load_fixed_map(MAP_FILE)

        brain = GenomeBrain(
            genome=genome,
            inputs=adapter.observation_size(),
            hidden=6,
            outputs=adapter.action_size(),
            action_order=adapter.ACTIONS
        )

        agent = LearningAgent("EVO", env, tuple(starts["A"]), adapter, brain)
        agent.set_mode("test")
        ok, st = run_episode(env, agent)
        success += int(ok)
        steps.append(st)
    return success, steps


def summarize(label, success, steps):
    arr = np.array(steps, dtype=float)
    print(f"\n=== MAZE {label} ===")
    print(f"Success: {success}/{RUNS} ({100*success/RUNS:.1f}%)")
    print(f"Avg steps (fail=max): {arr.mean():.1f}  | std: {arr.std():.1f}")


if __name__ == "__main__":
    sF, stF = eval_fixed()
    sQ, stQ = eval_q()
    sE, stE = eval_evo()

    summarize("Fixed", sF, stF)
    summarize("Q-learning", sQ, stQ)
    summarize("Evolution", sE, stE)

    labels = ["Fixed", "Q", "Evo"]
    means = [np.mean(stF), np.mean(stQ), np.mean(stE)]
    succs = [sF/RUNS, sQ/RUNS, sE/RUNS]

    plt.figure()
    plt.bar(labels, means)
    plt.ylabel("Avg steps (fail=max)")
    plt.title("Maze — Avg Steps (lower is better)")
    plt.grid(True, axis="y")
    plt.show()


