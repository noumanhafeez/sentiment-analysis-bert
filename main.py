# train.py

from src.dataset import load_and_prepare_dataset
from src.model import get_model
from src.trainer import train_model
from utils.logger import setup_logger


def main():
    logger = setup_logger()

    logger.info("Loading dataset...")
    train_dataset, test_dataset, tokenizer = load_and_prepare_dataset()

    logger.info("Loading model...")
    model = get_model()

    train_model(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        logger=logger
    )


if __name__ == "__main__":
    main()