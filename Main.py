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


# ---------------------------
# EXECUÇÃO
# ---------------------------
if __name__ == "__main__":
    # ---------------------------
    # CONFIGURAÇÃO
    # ---------------------------
    # Escolher o ambiente: "farol" ou "maze"
    ambiente = "farol"

    # Escolher o modo: "test" ou "train"
    modo = "test"

    # ---------------------------
    # INICIALIZAÇÃO
    # ---------------------------
    if ambiente == "farol":
        # Se modo de teste, usar JSON fixo; se treino, mapa aleatório
        json_file = "Resources/farol_map_1.json" if modo == "test" else None
        env, agents = setup_lighthouse(mode=modo, json_file=json_file)

    elif ambiente == "maze":
        # Se modo de teste, usar JSON fixo; se treino, mapa aleatório
        json_file = "Resources/maze_map_1.json" if modo == "test" else None
        env, agents = setup_maze(mode=modo, json_file=json_file)

    else:
        raise ValueError("Ambiente inválido! Escolher 'farol' ou 'maze'.")

    # Definir modo para cada agente
    for agent in agents:
        agent.set_mode(modo)

    # ---------------------------
    # EXECUÇÃO DO MOTOR
    # ---------------------------
    motor = MotorDeSimulacao(env, agents)
    motor.executa()
