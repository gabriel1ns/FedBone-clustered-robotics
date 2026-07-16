from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent
ROBOMIMIC_DATA_DIR = ROOT_DIR / "data" / "robomimic"
RESULTS_DIR = ROOT_DIR / "results"

for d in [ROBOMIMIC_DATA_DIR, RESULTS_DIR / "plots", RESULTS_DIR / "models", RESULTS_DIR / "logs"]:
    d.mkdir(parents=True, exist_ok=True)


NUM_ROBOTS = 10
SEQUENCE_LENGTH = 10
POLICY_TYPE = "deterministic"  # "deterministic" or "gaussian"
ROBOMIMIC_TASK_FILES = []     # Empty discovers local *.hdf5 under data/robomimic, excluding .cache
ROBOMIMIC_OBS_KEYS = []       # Empty auto-detects low-dimensional obs keys
ROBOMIMIC_TEST_RATIO = 0.2
ROBOMIMIC_MAX_DEMOS_PER_TASK = 0  # 0 means all demos
ROBOMIMIC_SUCCESS_THRESHOLD = 0.75
# LSTM baselines
HIDDEN_SIZE = 128
NUM_LAYERS = 2
DROPOUT = 0.3

# FedBone specific
EMBED_DIM = 64           # Embedding dimension (client-side)
GENERAL_HIDDEN = 128     # General model hidden size (server-side)
ADAPTATION_HEADS = 4     # Multi-head attention heads in task adaptation

BATCH_SIZE = 64
LOCAL_EPOCHS = 2
LEARNING_RATE = 0.0005

NUM_ROUNDS = 50
CLIENTS_PER_ROUND = 10

NUM_CLUSTERS = 3
CLUSTERING_METHOD = "kmeans"  # "kmeans" or "hierarchical"
RECLUSTERING_INTERVAL = 10    # Re-cluster every N rounds (0 = no reclustering)


USE_GP_AGGREGATION = True     # Use Gradient Projection aggregation
USE_TASK_ADAPTATION = True    # Use task adaptation module
RUN_FEDBONE_ABLATIONS = True  # Compare FedBone simple averaging vs GP Aggregation


SEED = 42
DEVICE = "cuda"  # "cuda" or "cpu"

LOG_INTERVAL = 5              # Log metrics every N rounds
SAVE_MODEL_INTERVAL = 10      # Save model every N rounds
VERBOSE = True                # Print detailed logs
