# Evaluation/CompareAll.py
import os
import numpy as np
import matplotlib.pyplot as plt

from Environments.Lighthouse import load_fixed_map as load_farol
from Environments.Maze import load_fixed_map as load_maze

from Agents.Fixed.LighthouseFixedAgent import LighthouseFixedAgent
from Agents.Fixed.MazeFixedAgent import MazeFixedAgent
from Agents.LearningAgent import LearningAgent

from Learning.Brains.QLearningBrain import QLearningBrain
from Learning.Brains.GenomeBrain import GenomeBrain
from Learning.Adapters.FarolAdapter import FarolAdapter
from Learning.Adapters.MazeAdapter import MazeAdapter

from Config import (
    FAROL_MAP, MAZE_MAP,
    FAROL_POLICY, MAZE_POLICY,
    FAROL_GENOME, MAZE_GENOME,
    RUNS, MAX_STEPS_FAROL, MAX_STEPS_MAZE,
    EVO_HIDDEN
)


def run_episode(env, agent, max_steps):
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


# ---------------- FAROL ----------------
def eval_farol_fixed():
    steps, succ = [], 0
    for _ in range(RUNS):
        env, starts, _, _ = load_farol(FAROL_MAP)
        agent = LighthouseFixedAgent("FIXED", env, tuple(starts["A"]))
        agent.set_mode("test")
        ok, st = run_episode(env, agent, MAX_STEPS_FAROL)
        succ += int(ok)
        steps.append(st)
    return succ, steps


def eval_farol_q():
    adapter = FarolAdapter()
    brain = QLearningBrain()
    brain.load(FAROL_POLICY)

    steps, succ = [], 0
    for _ in range(RUNS):
        env, starts, _, _ = load_farol(FAROL_MAP)
        agent = LearningAgent("Q", env, tuple(starts["A"]), adapter, brain)
        agent.set_mode("test")
        ok, st = run_episode(env, agent, MAX_STEPS_FAROL)
        succ += int(ok)
        steps.append(st)
    return succ, steps


def eval_farol_evo():
    adapter = FarolAdapter()
    if not os.path.exists(FAROL_GENOME):
        raise FileNotFoundError(f"Missing {FAROL_GENOME}. Train evolution farol first.")

    with open(FAROL_GENOME, "r") as f:
        genome = [float(x) for x in f.read().strip().split(",")]

    steps, succ = [], 0
    for _ in range(RUNS):
        env, starts, _, _ = load_farol(FAROL_MAP)

        brain = GenomeBrain(
            genome=genome,
            inputs=adapter.observation_size(),
            hidden=EVO_HIDDEN,
            outputs=adapter.action_size(),
            action_order=adapter.ACTIONS
        )

        agent = LearningAgent("EVO", env, tuple(starts["A"]), adapter, brain)
        agent.set_mode("test")
        ok, st = run_episode(env, agent, MAX_STEPS_FAROL)
        succ += int(ok)
        steps.append(st)
    return succ, steps


# ---------------- MAZE ----------------
def eval_maze_fixed():
    steps, succ = [], 0
    for _ in range(RUNS):
        env, starts, _, _ = load_maze(MAZE_MAP)
        agent = MazeFixedAgent("FIXED", env, tuple(starts["A"]))
        agent.set_mode("test")
        ok, st = run_episode(env, agent, MAX_STEPS_MAZE)
        succ += int(ok)
        steps.append(st)
    return succ, steps


def eval_maze_q():
    # IMPORTANT: must match training config
    adapter = MazeAdapter(include_position=True)
    brain = QLearningBrain()
    brain.load(MAZE_POLICY)

    steps, succ = [], 0
    for _ in range(RUNS):
        env, starts, _, _ = load_maze(MAZE_MAP)
        agent = LearningAgent("Q", env, tuple(starts["A"]), adapter, brain)
        agent.set_mode("test")
        ok, st = run_episode(env, agent, MAX_STEPS_MAZE)
        succ += int(ok)
        steps.append(st)
    return succ, steps


def eval_maze_evo():
    adapter = MazeAdapter(include_position=False)
    if not os.path.exists(MAZE_GENOME):
        raise FileNotFoundError(f"Missing {MAZE_GENOME}. Train evolution maze first.")

    with open(MAZE_GENOME, "r") as f:
        genome = [float(x) for x in f.read().strip().split(",")]

    steps, succ = [], 0
    for _ in range(RUNS):
        env, starts, _, _ = load_maze(MAZE_MAP)

        brain = GenomeBrain(
            genome=genome,
            inputs=adapter.observation_size(),
            hidden=EVO_HIDDEN,
            outputs=adapter.action_size(),
            action_order=adapter.ACTIONS
        )

        agent = LearningAgent("EVO", env, tuple(starts["A"]), adapter, brain)
        agent.set_mode("test")
        ok, st = run_episode(env, agent, MAX_STEPS_MAZE)
        succ += int(ok)
        steps.append(st)
    return succ, steps


def summarize(env_name, label, success, steps):
    arr = np.array(steps, dtype=float)
    return {
        "env": env_name,
        "agent": label,
        "success_rate": success / RUNS,
        "avg_steps": float(arr.mean()),
        "std_steps": float(arr.std()),
    }


if __name__ == "__main__":
    results = []

    sF, stF = eval_farol_fixed()
    sQ, stQ = eval_farol_q()
    sE, stE = eval_farol_evo()
    results.append(summarize("Farol", "Fixed", sF, stF))
    results.append(summarize("Farol", "Q",     sQ, stQ))
    results.append(summarize("Farol", "Evo",   sE, stE))

    sF2, stF2 = eval_maze_fixed()
    sQ2, stQ2 = eval_maze_q()
    sE2, stE2 = eval_maze_evo()
    results.append(summarize("Maze", "Fixed", sF2, stF2))
    results.append(summarize("Maze", "Q",     sQ2, stQ2))
    results.append(summarize("Maze", "Evo",   sE2, stE2))

    print("\n================= COMPARISON SUMMARY =================")
    for r in results:
        print(
            f"{r['env']:5s} | {r['agent']:5s} | "
            f"success={100 * r['success_rate']:.1f}% | "
            f"avg_steps={r['avg_steps']:.1f} ± {r['std_steps']:.1f}"
        )

    envs = ["Farol", "Maze"]
    agents = ["Fixed", "Q", "Evo"]

    def get(env, agent, key):
        for r in results:
            if r["env"] == env and r["agent"] == agent:
                return r[key]
        return None

    x = np.arange(len(agents))
    width = 0.35

    # Avg steps
    farol_means = [get("Farol", a, "avg_steps") for a in agents]
    maze_means  = [get("Maze",  a, "avg_steps") for a in agents]

    plt.figure()
    plt.bar(x - width / 2, farol_means, width, label="Farol")
    plt.bar(x + width / 2, maze_means,  width, label="Maze")
    plt.xticks(x, agents)
    plt.ylabel("Avg steps (fail = max)")
    plt.title("Average Steps — Farol vs Maze")
    plt.grid(True, axis="y")
    plt.legend()
    plt.show()


