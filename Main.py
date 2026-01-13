# Main.py
import os
import time

from Environments.Lighthouse import setup_lighthouse, load_fixed_map as load_farol
from Environments.Maze import setup_maze, load_fixed_map as load_maze

from Agents.LearningAgent import LearningAgent

from Learning.Adapters.FarolAdapter import FarolAdapter
from Learning.Adapters.MazeAdapter import MazeAdapter
from Learning.Brains.QLearningBrain import QLearningBrain
from Learning.Brains.GenomeBrain import GenomeBrain

from Training.TrainQLearningLighthouse import train_qlearning_lighthouse
from Training.TrainQLearningMaze import train_qlearning_maze
from Training.TrainEvolutionLighthouse import train_evolution_farol
from Training.TrainEvolutionMaze import train_evolution_maze


class MotorDeSimulacao:
    def __init__(self, env, agents, delay=0.4, max_steps=250):
        self.env = env
        self.agents = agents
        self.delay = delay
        self.max_steps = max_steps

    def executa(self):
        for step in range(self.max_steps):
            print(f"\n--- Step {step+1} ---")
            self.env.display()

            for agent in self.agents:
                obs = self.env.observacaoPara(agent)
                agent.observacao(obs)

                accao = agent.age()
                self.env.agir(accao, agent)

                if agent.reached_goal:
                    print(f"🎯 Agente {agent.name} atingiu o objetivo!")

            self.env.atualizacao()

            if all(a.reached_goal for a in self.agents):
                print("🎉 Todos os agentes atingiram o objetivo!")
                return

            time.sleep(self.delay)

        print("⏹ Limite de passos atingido.")


# ---------------------------------------------------
def build_learning_agent_farol(env, start_pos, metodo):
    adapter = FarolAdapter()
    base_dir = os.path.dirname(__file__)

    if metodo == "qlearning":
        brain = QLearningBrain()
        brain.load(os.path.join(base_dir, "policy_farol.json"))
        agent = LearningAgent("Q", env, start_pos, adapter, brain)
        agent.set_mode("test")
        return agent

    if metodo == "evolution":
        genome_path = os.path.join(base_dir, "farol_best_genome.txt")
        with open(genome_path, "r") as f:
            genome = [float(x) for x in f.read().strip().split(",")]

        brain = GenomeBrain(
            genome=genome,
            inputs=adapter.observation_size(),
            hidden=6,
            outputs=adapter.action_size(),
            action_order=adapter.ACTIONS
        )
        agent = LearningAgent("EVO", env, start_pos, adapter, brain)
        agent.set_mode("test")
        return agent

    raise ValueError("metodo_aprendizagem deve ser 'qlearning' ou 'evolution'")


def build_learning_agent_maze(env, start_pos, metodo):
    adapter = MazeAdapter()
    base_dir = os.path.dirname(__file__)

    if metodo == "qlearning":
        brain = QLearningBrain()
        brain.load(os.path.join(base_dir, "policy_maze.json"))
        agent = LearningAgent("Q", env, start_pos, adapter, brain)
        agent.set_mode("test")
        return agent

    if metodo == "evolution":
        genome_path = os.path.join(base_dir, "maze_best_genome.txt")
        with open(genome_path, "r") as f:
            genome = [float(x) for x in f.read().strip().split(",")]

        brain = GenomeBrain(
            genome=genome,
            inputs=adapter.observation_size(),
            hidden=6,
            outputs=adapter.action_size(),
            action_order=adapter.ACTIONS
        )
        agent = LearningAgent("EVO", env, start_pos, adapter, brain)
        agent.set_mode("test")
        return agent

    raise ValueError("metodo_aprendizagem deve ser 'qlearning' ou 'evolution'")


# ---------------------------------------------------
if __name__ == "__main__":

    # USER CONFIG
    tipo_agente = "learning"          # "fixed" | "learning"
    tipo_mapa = "fixed"               # "fixed" | "random"
    ambiente = "farol"                 # "farol" | "maze"

    metodo_aprendizagem = "qlearning" # "qlearning" | "evolution"
    treinar_antes = True              # True = train then test

    # Validation
    if tipo_agente == "learning" and tipo_mapa == "random":
        raise ValueError("LEARNING só pode ser usado com MAPA FIXO.")

    if tipo_agente == "fixed" and metodo_aprendizagem not in ("qlearning", "evolution"):
        raise ValueError("metodo_aprendizagem inválido.")

    base_dir = os.path.dirname(__file__)

    # FAROL
    if ambiente == "farol":
        map_path = os.path.join(base_dir, "Resources", "farol_map_2.json")

        if tipo_agente == "learning":
            if treinar_antes:
                if metodo_aprendizagem == "qlearning":
                    print("\n🔵 TREINO Q-LEARNING (FAROL)\n")
                    train_qlearning_lighthouse(map_path)
                else:
                    print("\n🔵 TREINO EVOLUTION (FAROL)\n")
                    train_evolution_farol(map_path)

            env, starts, _, _ = load_farol(map_path)
            start_pos = tuple(starts["A"])
            agent = build_learning_agent_farol(env, start_pos, metodo_aprendizagem)
            agents = [agent]
        else:
            json_file = "Resources/farol_map_2.json" if tipo_mapa == "fixed" else None
            env, agents = setup_lighthouse(agent_type="fixed", map_type=tipo_mapa, json_file=json_file)

    # MAZE
    elif ambiente == "maze":
        map_path = os.path.join(base_dir, "Resources", "maze_map_2.json")

        if tipo_agente == "learning":
            if treinar_antes:
                if metodo_aprendizagem == "qlearning":
                    print("\n🔵 TREINO Q-LEARNING (MAZE)\n")
                    train_qlearning_maze(map_path)
                else:
                    print("\n🔵 TREINO EVOLUTION (MAZE)\n")
                    train_evolution_maze(map_path)

            env, starts, _, _ = load_maze(map_path)
            start_pos = tuple(starts["A"])
            agent = build_learning_agent_maze(env, start_pos, metodo_aprendizagem)
            agents = [agent]
        else:
            json_file = "Resources/maze_map_2.json" if tipo_mapa == "fixed" else None
            env, agents = setup_maze(agent_type="fixed", map_type=tipo_mapa, json_file=json_file)

    else:
        raise ValueError("Ambiente inválido! Escolher 'farol' ou 'maze'.")

    motor = MotorDeSimulacao(env, agents)
    motor.executa()
