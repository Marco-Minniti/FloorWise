"""
Configuration for Natural Language Interface
"""

# Ollama Configuration
OLLAMA_HOST = "131.114.51.41:11434"
OLLAMA_MODEL = "llama2"

# Paths
HOUSE_DATA_PATH = "../5. ENGINE/house-planner/data/3_graph_updated_with_walls.json"
OUTPUT_DIR = "./outputs"

# Search parameters
MAX_TIME_SECONDS = 3600.0  # 60 minutes
MAX_SOLUTIONS = 3
MAX_DEPTH = 8
ADAPTIVE_DELTAS = True
POOL_EXPANSION = True
MAX_POOL_LEVEL = 3
VERBOSE = True

# final_script.py parameters (onion algorithm)
FINAL_SCRIPT_PATH = "../5. ENGINE/house-planner/mytest/manual_scripts/final_script.py"
ONION_ALGORITHM_DEFAULTS = {
    "TOLERANCE": 0.30,  # Tolleranza sulle aree in m²
    "MIN_ROOM_AREA": 5.0,  # Area minima stanze
    "MAX_ITERATIONS": 10,  # Max iterazioni algoritmo
    "NUM_SOLUTIONS": 1,  # Numero soluzioni (default: 1)
}




