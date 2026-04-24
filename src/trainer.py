# trainer.py

from transformers import TrainingArguments, Trainer, DataCollatorWithPadding
from config.config import (
    OUTPUT_DIR,
    LOG_DIR,
    BATCH_SIZE,
    EPOCHS,
    LEARNING_RATE
)
from utils.logger import compute_metrics


def train_model(model, tokenizer, train_dataset, test_dataset, logger):

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        logging_dir=LOG_DIR,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=EPOCHS,
        weight_decay=0.01,
        logging_steps=50,
        load_best_model_at_end=True,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )

    logger.info("Starting training...")
    trainer.train()

    logger.info("Training completed. Saving model...")

    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    logger.info(f"Model saved at {OUTPUT_DIR}")