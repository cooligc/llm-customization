import os
from transformers import (
    GPTNeoForCausalLM,
    GPT2Tokenizer,
    Trainer,
    TrainingArguments,
    TextDataset,
    DataCollatorForLanguageModeling,
)


MODEL_NAME = "EleutherAI/gpt-neo-125M"
OUTPUT_DIR = "models/gpt-neo-ipl"
TRAIN_FILE = "data/ipl_docs.txt"


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading tokenizer and model")
    tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
    model = GPTNeoForCausalLM.from_pretrained(MODEL_NAME)

    print("Preparing dataset")
    train_dataset = TextDataset(tokenizer=tokenizer, file_path=TRAIN_FILE, block_size=128)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        overwrite_output_dir=True,
        num_train_epochs=1,
        per_device_train_batch_size=4,
        save_steps=500,
        save_total_limit=2,
        logging_steps=100,
        fp16=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
    )

    print("Starting training")
    trainer.train()
    print("Saving final checkpoint")
    trainer.save_model(OUTPUT_DIR)


if __name__ == "__main__":
    main()
