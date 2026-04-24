# dataset.py

from datasets import load_dataset
from config.config import TRAIN_SIZE, TEST_SIZE, MODEL_NAME, MAX_LENGTH
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


def load_and_prepare_dataset():
    dataset = load_dataset("imdb")
    dataset = dataset.shuffle(seed=42)

    train_dataset = dataset["train"].select(range(TRAIN_SIZE))
    test_dataset = dataset["test"].select(range(TEST_SIZE))

    def preprocess(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            padding=True,
            max_length=MAX_LENGTH
        )

    train_dataset = train_dataset.map(preprocess, batched=True)
    test_dataset = test_dataset.map(preprocess, batched=True)

    train_dataset = train_dataset.remove_columns(["text"])
    test_dataset = test_dataset.remove_columns(["text"])

    train_dataset.set_format("torch")
    test_dataset.set_format("torch")

    return train_dataset, test_dataset, tokenizer