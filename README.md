# AI Visa Assistant

A conversational visa assistant that answers questions over embassy rules, company
sponsorship policy, and FAQs — and can explain *why* a specific document is required
(not just that it's missing) by grounding its answer in retrieved policy text.

## Architecture

```
User query
   │
   ▼
FastAPI  /chat
   │
   ▼
LangGraph pipeline
   ├─ classify_intent      → general question vs. "what's missing / why" question
   ├─ retrieve              → local FAISS search (sentence-transformers, all-MiniLM-L6-v2)
   ├─ general_rag           → answers from retrieved chunks, cites sources
   └─ missing_doc_explain   → diffs user's documents against a visa checklist,
                              explains each gap using retrieved policy context
   │
   ▼
Groq (Llama 3.3 70B) — generation only. Everything else (embeddings, indexing,
retrieval, checklist logic) runs 100% locally, so it's cheap and fast on 8GB RAM.
```

## Why this design

- **Local-first**: embeddings + vector search run on CPU with `sentence-transformers`
  + `faiss-cpu`. Only the final generation call hits an API (Groq, free tier, fast).
- **Explainability over generic answers**: the `missing_doc` path doesn't just say
  "you're missing X" — it pulls the *reason* from the checklist + policy text
  (e.g. "I-797 proves petition approval; without it the officer can't verify status").
- **Multi-language**: the `language` field in each request is threaded straight into
  the generation prompt.
- **Multi-source RAG**: embassy rules and internal company policy live as separate
  markdown files but are retrieved from the same index, each answer cites which file
  it drew from.

## Setup

```bash
cd visa-assistant
pip install -r requirements.txt

export GROQ_API_KEY=your_key_here   # free at console.groq.com

python ingest.py                     # builds the local FAISS index (data/index/)
uvicorn app.main:app --reload --port 8000
```

Optional demo UI:
```bash
streamlit run streamlit_app.py
```

## Example request

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Why is my visa missing something?",
    "language": "English",
    "visa_type": "us_h1b",
    "user_documents": ["ds160", "passport", "photo"]
  }'
```

This will detect the gap-analysis intent, see that `i797`, `employment_letter`, and
`education` are missing from `us_h1b`'s checklist, and generate an explanation for
each — grounded in the retrieved embassy-rules chunks.

## Extending it

- Swap in real embassy/company docs by dropping more `.md` files into
  `data/policies/` and re-running `python ingest.py`.
- Add new visa types by adding entries to `data/checklists.json`.
- Swap Groq for any other LangChain-compatible chat model by editing `rag_graph.py`.
- Add conversation memory by extending `VisaState` with a message history list and
  wiring it into the prompts.

## Known limitation (documented, not hidden)

The `classify_intent` node uses a keyword heuristic rather than a learned classifier —
good enough for a demo, but a real system would train/prompt a small classifier or add
few-shot intent examples for more robust routing on ambiguous queries.
