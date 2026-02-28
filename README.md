# IPL GPT-Neo RAG Demo

This repository shows how to build a simple Retrieval-Augmented Generation (RAG) system using a GPT‑Neo model and an IPL ball-by-ball dataset (`data/IPL.csv`). The goal is to train a custom language model on the cricket data and use a vector store to retrieve relevant facts when answering questions.

The workflow is broken down step‑wise:

1. **Preprocess** the CSV into plain-text passages.
2. **Embed and index** the passages using a sentence-transformer and FAISS.
3. **Fine‑tune** a GPT‑Neo model on the IPL text for language modelling.
4. **Query** the model with retrieval context (RAG-style) to produce answers.

All code examples are in Python and use the [🤗 Transformers](https://huggingface.co/docs/transformers/) library.

---

## Dataset

The data lives at `data/IPL.csv` and contains ball-by-ball information such as match id, teams, batter/bowler names, runs, wickets, venue, etc. We convert each row to a short string representing the delivery.

## Setup

1. Clone or open this workspace.
2. Create a virtual environment (recommended) and install the dependencies:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` includes:

```text
pandas
transformers
sentence-transformers
faiss-cpu      # or faiss-gpu if you have a CUDA device
```

3. Make sure you have at least a few gigabytes of disk space for model checkpoints. The example uses the 125M GPT‑Neo variant to keep resources reasonable.

## Step 1 – Preprocessing

Convert the CSV to a line-per-document text file. Run:

```bash
python preprocess.py
```

This creates `data/ipl_docs.txt` containing one passage per delivery and a pickled version used later.

> **Note:** you can easily alter `preprocess.py` to aggregate by match, innings, or player.

## Step 2 – Build the Retriever

Create vector embeddings for each passage and store them in a FAISS index:

```bash
python index.py
```

This script saves:

* `data/ipl_index.faiss` – the Faiss index with 384‑dim embeddings
* `data/ipl_docs.pkl` – a Python list of the original strings used for lookup

You can query the index interactively by importing `index.retrieve`.

## Step 3 – Fine‑tune GPT‑Neo

Train a causal language model on the IPL text. This is not strictly required for RAG, but it helps the generator ``understand`` the domain language.

```bash
python train.py
```

The checkpoint is written to `models/gpt-neo-ipl`. Feel free to adjust the model size or training parameters in `train.py`.

## Step 4 – Retrieval‑augmented generation

The `rag_query.py` script shows how to combine a user question, the retrieved passages, and GPT‑Neo to produce an answer.

```bash
python rag_query.py --question "Who took the first wicket in match 335982?"
```

The script uses the `index.retrieve` helper to get the top‑k documents and then prefixes them to the prompt.

You can use this pattern to build a simple Q&A bot, where retrieval is transparent to GPT‑Neo.

## Optional utilities

### Evaluation script

If you collect a set of question/answer pairs (CSV with `question,answer` headers),
run `evaluate.py` to get a rough measure of how often the generated reply matches
the gold answer and to compute BLEU if `nltk` is installed:

```bash
python evaluate.py data/qa_pairs.csv --k 5 --sample 100
```

This is useful for quick regressions when you modify the index, the prompt, or the
generator model.

### Web UI

Start a lightweight Flask application by running:

```bash
python web_app.py
```

Point your browser to `http://localhost:5000` to type questions or POST to `/api/ask`.
The source file contains both a human-facing HTML page and a JSON API.

### Supervised RAG fine‑tuning

`fine_tune_rag.py` is a skeleton showing how you could take a TSV file of QA pairs
and run a full `RagSequenceForGeneration` training loop using 🤗 Transformers.
The script needs to be fleshed out with proper dataset preprocessing and tokenization
but demonstrates the key components: retriever creation, tokenizer, and `Trainer`.

## Extending the pipeline

* Replace FAISS with another vector store (e.g. Chroma, Milvus).
* Use a larger GPT‑Neo or GPT‑J model; modify `train.py` accordingly.
* Convert the job to a formal RAG model by leveraging `RagRetriever` and `RagSequenceForGeneration` from 🤗 Transformers – see the inline comments in `rag_pipeline.py`.
* Add evaluation, caching, streaming, or a web front‑end.
* **Evaluation:** use `evaluate.py` to load QA pairs, run queries, and compute metrics such as accuracy or BLEU.
* **Web UI:** spin up a simple Flask/FastAPI server (`web_app.py`) to expose the RAG system via HTTP or a browser.
* **Supervised RAG training:** `fine_tune_rag.py` contains an example loop that fine‑tunes on question/answer pairs using HuggingFace `Trainer`.

---

This README is part of a minimal demo; see the Python files for full implementation details.
