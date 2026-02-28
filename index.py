import pickle
import faiss
from sentence_transformers import SentenceTransformer

DOC_PICKLE = "data/ipl_docs.pkl"
FAISS_INDEX = "data/ipl_index.faiss"
MODEL_NAME = "all-MiniLM-L6-v2"  # small embedding model


def build_index():
    # load existing docs
    with open(DOC_PICKLE, "rb") as f:
        docs = pickle.load(f)
    print(f"Loaded {len(docs)} passages")

    print("Creating embedding model")
    model = SentenceTransformer(MODEL_NAME)
    print("Computing embeddings (this may take a while)...")
    embeddings = model.encode(docs, show_progress_bar=True, convert_to_numpy=True)

    dim = embeddings.shape[1]
    print(f"Embedding dimension = {dim}")

    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    print(f"Index contains {index.ntotal} vectors")

    faiss.write_index(index, FAISS_INDEX)
    print(f"Faiss index saved to {FAISS_INDEX}")


def retrieve(query: str, k: int = 5) -> list[str]:
    """Return the top-k passages for the query"""
    # rebuild model and index each time for simplicity; in production you'd cache
    model = SentenceTransformer(MODEL_NAME)
    with open(DOC_PICKLE, "rb") as f:
        docs = pickle.load(f)
    index = faiss.read_index(FAISS_INDEX)

    q_vec = model.encode([query], convert_to_numpy=True)
    D, I = index.search(q_vec, k)
    hits = []
    for idx in I[0]:
        if idx < len(docs):
            hits.append(docs[idx])
    return hits


if __name__ == "__main__":
    # simple command‑line interface for testing
    build_index()
    while True:
        q = input("query> ")
        if q.strip() == "":
            break
        results = retrieve(q, k=3)
        print("--- retrieved")
        for r in results:
            print(r)
