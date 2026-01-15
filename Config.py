# Config.py
import os

# Root folder (where Main.py is)
BASE_DIR = os.path.dirname(__file__)

# ----------------------------
# Choose maps ONCE here
# ----------------------------
FAROL_MAP_NAME = "farol_map_2.json"
MAZE_MAP_NAME  = "maze_map_1.json"

FAROL_MAP = os.path.join(BASE_DIR, "Resources", FAROL_MAP_NAME)
MAZE_MAP  = os.path.join(BASE_DIR, "Resources", MAZE_MAP_NAME)

# ----------------------------
# Output files (keep consistent)
# ----------------------------
FAROL_POLICY = os.path.join(BASE_DIR, "policy_farol.json")
MAZE_POLICY  = os.path.join(BASE_DIR, "policy_maze.json")

FAROL_GENOME = os.path.join(BASE_DIR, "farol_best_genome.txt")
MAZE_GENOME  = os.path.join(BASE_DIR, "maze_best_genome.txt")

# ----------------------------
# Evaluation budgets
# ----------------------------
RUNS = 30
MAX_STEPS_FAROL = 250
MAX_STEPS_MAZE  = 200

# ----------------------------
# Q-learning hyperparameters
# ----------------------------
Q_EPISODES  = 500
Q_MAX_STEPS = 300
Q_ALPHA     = 0.3
Q_GAMMA     = 0.95
Q_EPSILON   = 0.2

# ----------------------------
# Evolution hyperparameters (generic)
# ----------------------------
EVO_POP_SIZE        = 40
EVO_GENERATIONS     = 80
EVO_STEPS_PER_AGENT = 200
EVO_MUTATION_RATE   = 0.15
EVO_MUTATION_STD    = 0.5

# ----------------------------
# Novelty / Hybrid Evolution (new)
# ----------------------------
EVO_HIDDEN = 6

# novelty distance
K_NEIGHBORS = 10

# archive update rule: add top-N novelty behaviours each generation
ARCHIVE_ADD_TOP = 5

# hybrid weight: ALPHA*novelty + (1-ALPHA)*fitness
NOVELTY_ALPHA = 0.05

# selection
EVO_PARENTS = 10
EVO_ELITE   = 10
