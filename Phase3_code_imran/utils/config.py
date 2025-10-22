# utils/config.py
import torch

# Kaggle dataset root (same as teammate)
DATA_ROOT = "/kaggle/input/balanced-raf-db-dataset-7575-grayscale"

# Training
BATCH_SIZE = 32
EPOCHS = 5
LR = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = 2

# Classes
CLASSES = ["Angry","Disgust","Fear","Happy","Neutral","Sad","Surprise"]
NUM_CLASSES = len(CLASSES)

# Image sizes
IMG_SIZE = 224

# Text (placeholder until you have real text)
MAX_TOKENS = 32      # used by dummy tokenizer & text net
VOCAB_SIZE = 30522   # keep BERT-like size to swap later if needed
