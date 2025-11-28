import time

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
# EXECUÇÃO EXEMPLO
# ---------------------------
if __name__ == "__main__":
    # Escolher o ambiente
    # Exemplo Labirinto:
    from Environments.Maze import setup_maze
    env, agents = setup_maze()

    # Ou Farol:
    # from Environments.Lighthouse import setup_lighthouse
    # env, agents = setup_lighthouse()

    # Opcional: definir modo train/test
    for agent in agents:
        agent.set_mode("train")  # ou "test"

    motor = MotorDeSimulacao(env, agents)
    motor.executa()
