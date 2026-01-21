"""
Quick Configuration for Natural Language Interface (for faster testing)
Use this for quick tests by doing: import config_quick as config
"""

# Ollama Configuration
OLLAMA_HOST = "131.114.51.41:11434"
OLLAMA_MODEL = "llama2"

# Paths
HOUSE_DATA_PATH = "../5. ENGINE/house-planner/data/3_graph_updated_with_walls.json"
OUTPUT_DIR = "./outputs"

# Search parameters - REDUCED FOR QUICK TESTING
MAX_TIME_SECONDS = 300.0  # 5 minutes instead of 60
MAX_SOLUTIONS = 3
MAX_DEPTH = 4  # Reduced from 8 for faster search
ADAPTIVE_DELTAS = True
POOL_EXPANSION = True
MAX_POOL_LEVEL = 2  # Reduced from 3
VERBOSE = True




