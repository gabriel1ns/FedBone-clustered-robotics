from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / "data" / "raw"
RESULTS_DIR = ROOT_DIR / "results"

for d in [DATA_DIR, RESULTS_DIR / "plots", RESULTS_DIR / "models", RESULTS_DIR / "logs"]:
    d.mkdir(parents=True, exist_ok=True)

# ==================== DATASET ====================
NUM_ROBOTS = 10
SEQUENCE_LENGTH = 128
NUM_FEATURES = 9  # HAR dataset features
NUM_CLASSES = 6   # HAR dataset: walking, walking_upstairs, walking_downstairs, sitting, standing, laying
ALPHA = 0.5       # Dirichlet alpha for non-IID (lower = more heterogeneous)

# ==================== MODEL ====================
# LSTM baseline
HIDDEN_SIZE = 128
NUM_LAYERS = 2
DROPOUT = 0.3

# FedBone specific
EMBED_DIM = 64           # Embedding dimension (client-side)
GENERAL_HIDDEN = 128     # General model hidden size (server-side)
ADAPTATION_HEADS = 4     # Multi-head attention heads in task adaptation

# ==================== TRAINING ====================
BATCH_SIZE = 32
LOCAL_EPOCHS = 5
LEARNING_RATE = 0.001

# ==================== FEDERATED LEARNING ====================
NUM_ROUNDS = 50
CLIENTS_PER_ROUND = 5

# ==================== CLUSTERING (for Clustered FL) ====================
NUM_CLUSTERS = 3
CLUSTERING_METHOD = "kmeans"  # "kmeans" or "hierarchical"
RECLUSTERING_INTERVAL = 10    # Re-cluster every N rounds (0 = no reclustering)

# ==================== FEDBONE ====================
USE_GP_AGGREGATION = True     # Use Gradient Projection aggregation
USE_TASK_ADAPTATION = True    # Use task adaptation module

# ==================== MULTI-TASK ====================
NUM_TASKS = 3                 # Number of different tasks
TASK_DISTRIBUTION = "mixed"   # "uniform", "specialized", or "mixed"

# ==================== SYSTEM ====================
SEED = 42
DEVICE = "cuda"  # "cuda" or "cpu"

# ==================== LOGGING ====================
LOG_INTERVAL = 5              # Log metrics every N rounds
SAVE_MODEL_INTERVAL = 10      # Save model every N rounds
VERBOSE = True                # Print detailed logs