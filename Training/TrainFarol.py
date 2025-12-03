# Main.py
import time

from Environments.Lighthouse import setup_lighthouse
from Environments.Maze import setup_maze


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
    tipo_agente = "fixed"

    # Tipo de mapa: "fixed" ou "random"
    tipo_mapa = "random"

    # Ambiente: "farol" ou "maze"
    ambiente = "maze"

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
        if tipo_mapa == "fixed":
            json_file = "Resources/farol_map_1.json"
        else:
            json_file = None  # aleatório

        env, agents = setup_lighthouse(agent_type=tipo_agente,
                                       map_type=tipo_mapa,
                                       json_file=json_file)

    elif ambiente == "maze":
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
    for agent in agents:
        agent.set_mode("train" if tipo_agente == "learning" else "test")


    # ---------------------------------------------------
    # EXECUTAR
    # ---------------------------------------------------
    motor = MotorDeSimulacao(env, agents)
    motor.executa()
