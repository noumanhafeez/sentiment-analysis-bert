# config.py

MODEL_NAME = "bert-base-uncased"

MAX_LENGTH = 256
NUM_LABELS = 2

TRAIN_SIZE = 5000
TEST_SIZE = 1000

BATCH_SIZE = 8
EPOCHS = 2
LEARNING_RATE = 2e-5

OUTPUT_DIR = "./saved_model"
LOG_DIR = "./logs"