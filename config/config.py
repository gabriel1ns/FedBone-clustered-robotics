from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / "data" / "raw"
RESULTS_DIR = ROOT_DIR / "results"

for d in [DATA_DIR, RESULTS_DIR / "plots", RESULTS_DIR / "models", RESULTS_DIR / "logs"]:
    d.mkdir(parents=True, exist_ok=True)

#dataset
NUM_ROBOTS = 10
SEQUENCE_LENGTH = 50
NUM_FEATURES = 6
NUM_CLASSES = 5

#model
HIDDEN_SIZE = 128
NUM_LAYERS = 2
DROPOUT = 0.3

#training
BATCH_SIZE = 32
LOCAL_EPOCHS = 5
LEARNING_RATE = 0.001

#FL
NUM_ROUNDS = 50
CLIENTS_PER_ROUND = 5

#clusters
NUM_CLUSTERS = 3
CLUSTERING_METHOD = "kmeans"
RECLUSTERING_INTERVAL = 10


SEED = 42
DEVICE = "cuda"