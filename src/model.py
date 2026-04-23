# model.py

from transformers import AutoModelForSequenceClassification
from config.config import MODEL_NAME, NUM_LABELS

def get_model():
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_LABELS
    )
    return model