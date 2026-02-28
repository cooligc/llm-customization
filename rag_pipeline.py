"""Example of building a HuggingFace RAG model backed by our IPL index.

RAG is a two‑stage architecture: a retriever that searches a knowledge base and a generator
that conditions on the retrieved passages. The code below is illustrative – the built-in
`RagRetriever` expects either a HuggingFace dataset or precomputed index.

We use our custom FAISS index and passages. At training time you could create a
`Dataset` of (question, answer) pairs and fine‑tune the whole RAG model with
`.train()` in a similar way to `train.py`.

Note: RAG normally uses a BART/T5-style generator; here we show how you could
plug in GPT-Neo by passing the generator config manually.

"""
from transformers import (
    RagTokenizer,
    RagRetriever,
    RagSequenceForGeneration,
    GPTNeoForCausalLM,
    GPT2Tokenizer,
)
import pickle
import faiss

PASSAGES_FILE = "data/ipl_docs.pkl"
INDEX_FILE = "data/ipl_index.faiss"

def make_rag_model():
    # load passages for the retriever (they must be UTF-8 strings)
    with open(PASSAGES_FILE, "rb") as f:
        passages = pickle.load(f)

    # HuggingFace rag retriever can be initialized with our index
    retriever = RagRetriever.from_pretrained(
        "facebook/rag-token-base",   # tokenizer/embedding defaults
        index_name="custom",
        passages_path=PASSAGES_FILE,
        index_path=INDEX_FILE,
    )

    # load a standard RAG generator and override it if desired
    # generator = RagSequenceForGeneration.from_pretrained("facebook/rag-sequence-base", retriever=retriever)
    # # You could replace `generator.question_encoder` or `generator.generator`
    # # with GPT-Neo if you want to experiment, but this requires careful config.
    #
    # return generator
    return retriever


if __name__ == "__main__":
    print("This file is a demonstration; run `rag_query.py` for a working example.")
