"""
retriever.py — Local semantic retriever over the FAISS index built by ingest.py
"""
import pickle
from pathlib import Path

import faiss
from sentence_transformers import SentenceTransformer

INDEX_DIR = Path(__file__).parent / "data" / "index"
EMBED_MODEL = "all-MiniLM-L6-v2"


class Retriever:
    def __init__(self):
        index_path = INDEX_DIR / "visa_index.faiss"
        chunks_path = INDEX_DIR / "chunks.pkl"
        if not index_path.exists():
            raise FileNotFoundError(
                "No index found. Run `python ingest.py` first to build the vector store."
            )
        self.index = faiss.read_index(str(index_path))
        with open(chunks_path, "rb") as f:
            self.chunks = pickle.load(f)
        self.model = SentenceTransformer(EMBED_MODEL)

    def retrieve(self, query: str, k: int = 4):
        q_emb = self.model.encode([query], normalize_embeddings=True)
        scores, idxs = self.index.search(q_emb, k)
        results = []
        for score, idx in zip(scores[0], idxs[0]):
            if idx == -1:
                continue
            chunk = self.chunks[idx]
            results.append({
                "text": chunk["text"],
                "source": chunk["source"],
                "score": float(score),
            })
        return results


_retriever_singleton = None


def get_retriever() -> Retriever:
    global _retriever_singleton
    if _retriever_singleton is None:
        _retriever_singleton = Retriever()
    return _retriever_singleton
