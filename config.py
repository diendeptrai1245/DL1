# E:\code\DL1\config.py

from pathlib import Path

# --- General Paths ---
# Project base directory
BASE_DIR = Path(__file__).resolve().parent

# --- Dataset Config ---
# Auto-join dataset paths
DATASET_DIR = BASE_DIR / 'DATASET' # <-- Edit 'DATASET' if your folder name differs
TRAIN_DIR = DATASET_DIR / 'TRAIN'
TEST_DIR = DATASET_DIR / 'TEST'

# --- Model Config ---
MODEL_SAVE_PATH = BASE_DIR / 'models' / 'cnn_rubbish_classifier.h5'
CLASS_NAMES_PATH = BASE_DIR / 'class_names.json' # Path to save class names
HISTORY_SAVE_PATH = BASE_DIR / 'training_history.json' # Path to save training history

# --- Training Parameters ---
IMAGE_WIDTH = 128
IMAGE_HEIGHT = 128
BATCH_SIZE = 32  # Tuneable parameter
EPOCHS = 10      # Tuneable parameter