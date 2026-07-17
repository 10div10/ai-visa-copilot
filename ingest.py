"""
ingest.py — Build a local FAISS vector index over embassy rules, company policy, and FAQs.

Runs fully local (sentence-transformers on CPU) so it's fine on 8GB RAM.
Usage:
    python ingest.py
"""
import os
import glob
import json
import pickle
from pathlib import Path

from sentence_transformers import SentenceTransformer
import faiss

DATA_DIR = Path(__file__).parent / "data" / "policies"
INDEX_DIR = Path(__file__).parent / "data" / "index"
INDEX_DIR.mkdir(parents=True, exist_ok=True)

# Small, fast, local embedding model (~80MB, CPU-friendly)
EMBED_MODEL = "all-MiniLM-L6-v2"
CHUNK_SIZE = 500     # characters
CHUNK_OVERLAP = 80


def chunk_text(text: str, source: str):
    chunks = []
    start = 0
    while start < len(text):
        end = start + CHUNK_SIZE
        chunk = text[start:end].strip()
        if chunk:
            chunks.append({"text": chunk, "source": source})
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks


def build_index():
    print(f"Loading embedding model '{EMBED_MODEL}' (local, CPU)...")
    model = SentenceTransformer(EMBED_MODEL)

    all_chunks = []
    for filepath in glob.glob(str(DATA_DIR / "*.md")):
        source = os.path.basename(filepath)
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()
        chunks = chunk_text(text, source)
        all_chunks.extend(chunks)
        print(f"  {source}: {len(chunks)} chunks")

    if not all_chunks:
        raise SystemExit("No policy documents found in data/policies/. Add .md files first.")

    texts = [c["text"] for c in all_chunks]
    print(f"Embedding {len(texts)} chunks...")
    embeddings = model.encode(texts, show_progress_bar=True, normalize_embeddings=True)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # cosine sim via inner product on normalized vecs
    index.add(embeddings)

    faiss.write_index(index, str(INDEX_DIR / "visa_index.faiss"))
    with open(INDEX_DIR / "chunks.pkl", "wb") as f:
        pickle.dump(all_chunks, f)

    print(f"Index built: {index.ntotal} vectors -> {INDEX_DIR}")


if __name__ == "__main__":
    build_index()
