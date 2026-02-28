"""Example script to fine-tune a RAG-style model using question/answer pairs.

This is a skeleton; for a real project you'd prepare a `datasets.Dataset`
containing `input_text` (question) and `target_text` (answer) fields and then
use `Trainer`/`Seq2SeqTrainer` with a `RagSequenceForGeneration` model.
"""
from transformers import (
    RagTokenizer,
    RagRetriever,
    RagSequenceForGeneration,
    Trainer,
    TrainingArguments,
)
import pickle

PASSAGES_FILE = "data/ipl_docs.pkl"
INDEX_FILE = "data/ipl_index.faiss"
QA_TSV = "data/qa_pairs.tsv"  # expected tab-separated question<tab>answer
OUTPUT_DIR = "models/rag-ipl"


def load_qa_dataset(path: str):
    import pandas as pd
    df = pd.read_csv(path, sep="\t", names=["question", "answer"])
    return df


def main():
    # load retriever
    retriever = RagRetriever.from_pretrained(
        "facebook/rag-token-base",
        index_name="custom",
        passages_path=PASSAGES_FILE,
        index_path=INDEX_FILE,
    )
    tokenizer = RagTokenizer.from_pretrained("facebook/rag-token-base", retriever=retriever)

    model = RagSequenceForGeneration.from_pretrained("facebook/rag-sequence-base", retriever=retriever)

    # prepare dataset
    df = load_qa_dataset(QA_TSV)
    # rename for HF dataset compatibility
    dataset = {
        "train": df.to_dict(orient="list")
    }
    # note: you would convert this to a proper Dataset object and
    # tokenize it in a preprocessing function.

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=4,
        num_train_epochs=1,
        logging_steps=50,
        save_steps=200,
        fp16=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset['train'],  # placeholder
        tokenizer=tokenizer,
    )
    # for a real run you'd implement data collator, data preprocessing etc.

    trainer.train()
    trainer.save_model(OUTPUT_DIR)


if __name__ == '__main__':
    main()
