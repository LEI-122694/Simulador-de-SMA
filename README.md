# Simulador de Sistemas Multi-Agente com Aprendizagem

Projeto prático desenvolvido no âmbito da disciplina **Agentes Autónomos**, cujo objetivo é a implementação de um simulador flexível em Python para sistemas multi-agente, com suporte a agentes de comportamento fixo e agentes com capacidade de aprendizagem.

Foram implementados dois problemas clássicos:
- **Farol (Lighthouse)**
- **Labirinto (Maze)**

---

## Autores

- **Daniel Gutierrez** — Nº 122694  
- **Ahmad Hussein** — Nº 111641  

---

## Requisitos

- Python **3.8** ou superior  
- Não são necessárias bibliotecas externas (apenas bibliotecas standard do Python)

---

## Estrutura do Projeto (resumo)

Agents/
Environments/
Learning/
Training/
Evaluation/
Resources/
Main.py

yaml
Copiar código

O ficheiro **`Main.py`** é o ponto de entrada principal do simulador.

---

## Como Executar

A execução do simulador é feita através do ficheiro `Main.py`.

### Executar o programa

Na raiz do projeto, executar:

```bash
python Main.py
Configuração do Simulador
A configuração é feita diretamente no bloco final do ficheiro Main.py:

python
Copiar código
if __name__ == "__main__":

    # USER CONFIG
    tipo_agente = "learning"          # "fixed" | "learning"
    tipo_mapa = "fixed"               # "fixed" | "random"
    ambiente = "farol"                # "farol" | "maze"

    metodo_aprendizagem = "evolution" # "qlearning" | "evolution"
    treinar_antes = True              # True = treina antes de testar
Parâmetros Disponíveis
Ambiente
"farol" — Problema do Farol

"maze" — Problema do Labirinto

Tipo de Agente
"fixed" — Agentes com comportamento pré-programado

"learning" — Agente com aprendizagem (modo teste após treino)

Tipo de Mapa
"fixed" — Mapa definido em ficheiro JSON

"random" — Mapa gerado aleatoriamente
⚠️ Disponível apenas para agentes fixos

Método de Aprendizagem
"qlearning" — Aprendizagem por reforço (Q-Learning)

"evolution" — Estratégia evolucionária baseada em genoma

Treino
True — Executa treino antes do teste

False — Usa políticas ou genomas previamente guardados

Exemplos de Configuração
Farol com Q-Learning
python
Copiar código
ambiente = "farol"
tipo_agente = "learning"
metodo_aprendizagem = "qlearning"
treinar_antes = True
Labirinto com Estratégia Evolucionária
python
Copiar código
ambiente = "maze"
tipo_agente = "learning"
metodo_aprendizagem = "evolution"
treinar_antes = True
Agentes Fixos com Mapa Aleatório
python
Copiar código
ambiente = "maze"
tipo_agente = "fixed"
tipo_mapa = "random"
