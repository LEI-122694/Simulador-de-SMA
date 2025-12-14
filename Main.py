# Main.py
import time

from Environments.Lighthouse import setup_lighthouse
from Environments.Maze import setup_maze
from Training.TrainFarol import train_farol, plot_learning_curve, load_trained_agent
from Environments.Lighthouse import load_fixed_map
from Training.TrainMaze import train_maze, plot_novelty as plot_maze_novelty, load_trained_maze_agent

import os


class MotorDeSimulacao:
    def __init__(self, env, agents, delay=0.2, max_steps=250):
        self.env = env
        self.agents = agents
        self.delay = delay
        self.max_steps = max_steps

    def listaAgentes(self):
        return self.agents

    def executa(self):
        for step in range(self.max_steps):
            print(f"\n--- Step {step+1} ---")
            self.env.display()

            # Cada agente age
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
# NOVA EXECUÇÃO CONFIGURÁVEL
# ---------------------------------------------------
if __name__ == "__main__":

    # ---------------------------------------------------
    # CONFIGURAÇÃO DO UTILIZADOR
    # ---------------------------------------------------
    # Tipo de agente: "fixed" ou "learning"
    tipo_agente = "learning"

    # Tipo de mapa: "fixed" ou "random"
    tipo_mapa = "fixed"

    # Ambiente: "farol" ou "maze"
    ambiente = "farol"

    # ---------------------------------------------------
    # REGRAS DE VALIDAÇÃO
    # ---------------------------------------------------
    if tipo_agente == "learning" and tipo_mapa == "random":
        print("ERRO: O modo LEARNING só pode ser usado com MAPA FIXO!")
        print("    Mude para: tipo_mapa = 'fixed'")
        exit(1)


    # ---------------------------------------------------
    # SELEÇÃO DO AMBIENTE E MAPA
    # ---------------------------------------------------
    if ambiente == "farol":

        # ==========================================
        # SPECIAL BRANCH: FAROL Q-LEARNING
        # ==========================================
        if tipo_agente == "learning" and tipo_mapa == "fixed":
            print("\n🔵 Iniciando TREINO Q-LEARNING para FAROL...\n")

            BASE = os.path.dirname(__file__)
            map_path = os.path.join(BASE, "Resources", "farol_map_2.json")

            # 1️⃣ TRAIN
            rewards = train_farol(map_path)

            # 2️⃣ SHOW LEARNING CURVE
            plot_learning_curve(rewards)

            print("\n🟢 Testando POLÍTICA TREINADA no ambiente FAROL...\n")

            # 3️⃣ LOAD TRAINED AGENT
            env, agents = load_trained_agent(map_path)

            # 4️⃣ RUN SIMULATION using your normal engine
            motor = MotorDeSimulacao(env, agents)
            motor.executa()

            exit(0)

        # NORMAL FIXED / RANDOM FAROL (unchanged)
        if tipo_mapa == "fixed":
            json_file = "Resources/farol_map_2.json"
        else:
            json_file = None

        env, agents = setup_lighthouse(agent_type=tipo_agente,
                                       map_type=tipo_mapa,
                                       json_file=json_file)


    elif ambiente == "maze":

        # ==========================================

        # SPECIAL BRANCH: NOVELTY SEARCH EVOLUTION

        # ==========================================

        if tipo_agente == "learning" and tipo_mapa == "fixed":
            print("\n🔵 Iniciando EVOLUÇÃO (NOVELTY SEARCH) para MAZE...\n")

            BASE = os.path.dirname(__file__)

            map_path = os.path.join(BASE, "Resources", "maze_map_1.json")

            # Train
            best, mean, archive, goals, reached_list = train_maze(map_path)

            # Plot
            plot_maze_novelty(best, mean, archive, goals, reached_list)

            print("\n🟢 Testando o MELHOR AGENTE evoluído no MAZE...\n")

            # Load best individual

            env, agents = load_trained_maze_agent(map_path)

            motor = MotorDeSimulacao(env, agents)

            motor.executa()

            exit(0)

        # ==================================================

        # NORMAL FIXED / RANDOM MAZE (unchanged)

        # ==================================================

        if tipo_mapa == "fixed":

            json_file = "Resources/maze_map_1.json"

        else:

            json_file = None

        env, agents = setup_maze(agent_type=tipo_agente,

                                 map_type=tipo_mapa,

                                 json_file=json_file)

    else:
        raise ValueError("Ambiente inválido! Escolher 'farol' ou 'maze'.")


    # ---------------------------------------------------
    # APLICAR MODO AOS AGENTES
    # ---------------------------------------------------
    if tipo_agente != "learning":  # RL already sets its own mode
        for agent in agents:
            agent.set_mode("train" if tipo_agente == "learning" else "test")

    # ---------------------------------------------------
    # EXECUTAR
    # ---------------------------------------------------
    motor = MotorDeSimulacao(env, agents)
    motor.executa()
