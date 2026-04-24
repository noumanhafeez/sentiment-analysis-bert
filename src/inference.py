# inference.py

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from config.config import OUTPUT_DIR

class SentimentPredictor:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(OUTPUT_DIR)
        self.model = AutoModelForSequenceClassification.from_pretrained(OUTPUT_DIR)

        self.model.eval()

    def predict(self, text: str):
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True
        )

        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)

        label = torch.argmax(probs).item()
        confidence = probs[0][label].item()

        return {
            "text": text,
            "label": "POSITIVE" if label == 1 else "NEGATIVE",
            "confidence": round(confidence, 4)
        }

