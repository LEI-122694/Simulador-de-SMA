# Simulador de Sistemas Multi-Agente com Aprendizagem

Projeto prático desenvolvido no âmbito da disciplina **Agentes Autónomos**, cujo objetivo é a implementação de um simulador flexível em Python para sistemas multi-agente, com suporte a agentes de comportamento fixo e agentes com capacidade de aprendizagem.

Foram implementados dois problemas clássicos:
- **Farol (Lighthouse)**
- **Labirinto (Maze)**

---

## Autores

- **Ahmad Hussein** — Nº 111641  
- **Daniel Gutierrez** — Nº 122694  

---

## Requisitos

- Python **3.8** ou superior  
- Não são necessárias bibliotecas externas (apenas bibliotecas padrão do Python)

---

## Estrutura do Projeto (resumo)

- Agents/
- Environments/
- Learning/
- Training/
- Evaluation/
- Resources/
- Main.py

> O ficheiro **`Main.py`** é o ponto de entrada principal do simulador.

---

## Como Executar

A execução do simulador é feita através do ficheiro `Main.py`.

### Executar o programa

A configuração é feita diretamente no bloco final do ficheiro Main.py:

```python
if __name__ == "__main__":

    # USER CONFIG
    tipo_agente = "learning"          # "fixed" | "learning"
    tipo_mapa = "fixed"               # "fixed" | "random"
    ambiente = "farol"                # "farol" | "maze"

    metodo_aprendizagem = "evolution" # "qlearning" | "evolution"
    treinar_antes = True              # True = treina antes de testar

```

## Parâmetros Disponíveis

### Ambiente
- `"farol"` — Problema do Farol
- `"maze"` — Problema do Labirinto

### Tipo de Agente
- `"fixed"` — Agentes com comportamento pré-programado
- `"learning"` — Agente com aprendizagem (modo teste após treino)

### Tipo de Mapa
- `"fixed"` — Mapa definido em ficheiro JSON
- `"random"` — Mapa gerado aleatoriamente  (Disponível apenas para agentes fixos)

### Método de Aprendizagem
- `"qlearning"` — Aprendizagem por reforço (Q-Learning)
- `"evolution"` — Estratégia evolucionária baseada em genoma

### Treinamento
- `True` — Executa treino antes do teste
- `False` — Usa políticas ou genomas previamente guardados

## Exemplos de Configuração

### Farol com Q-Learning
```python
ambiente = "farol"
tipo_agente = "learning"
metodo_aprendizagem = "qlearning"
treinar_antes = True
```

### Labirinto com Estratégia Evolucionária
```python
ambiente = "maze"
tipo_agente = "learning"
metodo_aprendizagem = "evolution"
treinar_antes = True
```

### Agentes Fixos com Mapa Aleatório
```python
ambiente = "maze"
tipo_agente = "fixed"
tipo_mapa = "random"
```

## Comparação e Avaliação de Modelos
Para comparar e avaliar o desempenho dos diferentes modelos implementados, existem três scripts específicos:

### 1. Comparação no Problema do Farol
```bash
python CompareFarol.py
```
Este script executa e compara:

 - Agentes fixos vs agentes com aprendizagem

 - Diferentes métodos de aprendizagem (Q-Learning vs Evolution) no ambiente Farol

### 2. Comparação no Problema do Labirinto
```bash
python CompareMaze.py
```
Este script executa e compara:

 - Agentes fixos vs agentes com aprendizagem no labirinto

 - Diferentes métodos de aprendizagem no ambiente Maze

### 3. Comparação Completa de Todos os Modelos
```bash
python CompareAll.py
```
Este script executa uma comparação abrangente de todos os modelos implementados:

 - Ambos os ambientes (Farol e Maze)

## Configuração Global (Config.py)

O ficheiro **`Config.py`** centraliza todas as configurações e parâmetros do simulador, incluindo mapas, caminhos de saída e hiperparâmetros dos métodos de aprendizagem. Isto permite modificar rapidamente o comportamento do simulador sem alterar o código principal.  

### Mapas
-FAROL_MAP
-MAZE_MAP

### Ficheiros de saída
-FAROL_POLICY, MAZE_POLICY
-FAROL_GENOME, MAZE_GENOME

### Avaliação
-RUNS
-MAX_STEPS_FAROL, MAX_STEPS_MAZE

### Q-Learning
-Q_EPISODES, Q_MAX_STEPS
-Q_ALPHA, Q_GAMMA, Q_EPSILON

### Evolução
-EVO_POP_SIZE, EVO_GENERATIONS, EVO_STEPS_PER_AGENT
-EVO_MUTATION_RATE, EVO_MUTATION_STD

### Novelty Search / Híbrido
-EVO_HIDDEN, K_NEIGHBORS
-ARCHIVE_ADD_TOP, NOVELTY_ALPHA

### Seleção
-EVO_PARENTS, EVO_ELITE

 - Todos os métodos de aprendizagem
