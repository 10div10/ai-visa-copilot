# AI Visa Copilot

A conversational visa assistant that answers questions over embassy rules, company
sponsorship policy, and FAQs — explains *why* a specific document is required (not
just that it's missing) — and estimates rejection risk with an explainable ML model,
grounding every answer in retrieved policy text rather than generic responses.

## Architecture

```
User query
   │
   ▼
FastAPI  /chat, /risk-score
   │
   ▼
LangGraph pipeline
   ├─ classify_intent      → general question vs. "what's missing" vs. "what's my risk"
   ├─ retrieve              → local FAISS search (sentence-transformers, all-MiniLM-L6-v2)
   ├─ general_rag           → answers from retrieved chunks, cites sources
   ├─ missing_doc_explain   → diffs user's documents against a visa checklist,
   │                          explains each gap using retrieved policy context
   └─ risk_score            → scores the application with a trained XGBoost model,
                              explains the top SHAP-driven risk factors in plain language
   │
   ▼
Groq (Llama 3.3 70B) — generation only. Everything else (embeddings, indexing,
retrieval, checklist logic, and risk scoring) runs 100% locally, so it's cheap
and fast on 8GB RAM.
```

## Why this design

- **Local-first**: embeddings + vector search run on CPU with `sentence-transformers`
  + `faiss-cpu`. Risk scoring runs on a local XGBoost model. Only the final generation
  call hits an API (Groq, free tier, fast).
- **Explainability over generic answers**: the `missing_doc` path pulls the *reason*
  from the checklist + policy text (e.g. "I-797 proves petition approval; without it
  the officer can't verify status"), and the `risk_score` path explains predictions
  using SHAP values rather than a black-box probability.
- **Multi-language**: the `language` field in each request is threaded straight into
  the generation prompt.
- **Multi-source RAG**: embassy rules and internal company policy live as separate
  markdown files but are retrieved from the same index; each answer cites which file
  it drew from.

## Rejection risk scoring

`/risk-score` (and the `risk_score` intent in `/chat`) predicts the probability an
application gets rejected, using:

- Document completeness
- Passport validity buffer
- Financial proof strength
- Ties-to-home-country strength
- Application timing relative to travel/start date
- Prior rejection history

The model is trained on a synthetic dataset (`generate_synthetic_data.py`) built from
the rejection patterns documented in `data/policies/*.md` (real embassy rejection data
isn't public). It's an XGBoost classifier (`train_risk_model.py`), and predictions are
explained with SHAP (`risk_model.py`) — every score comes with the top 3 contributing
factors, not just a bare number.

## Setup

```bash
git clone https://github.com/10div10/ai-visa-copilot.git
cd ai-visa-copilot
pip install -r requirements.txt --break-system-packages   # macOS system Python

export GROQ_API_KEY=your_key_here   # free at console.groq.com

python3 ingest.py                    # builds the local FAISS index (data/index/)
python3 generate_synthetic_data.py   # builds the synthetic training set
python3 train_risk_model.py          # trains the XGBoost risk model

uvicorn app.main:app --port 8000
```

Optional demo UI (in a second terminal):
```bash
streamlit run streamlit_app.py
```

## Example requests

**General / document-gap question:**
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
This detects the gap-analysis intent, sees that `i797`, `employment_letter`, and
`education` are missing from `us_h1b`'s checklist, and explains each gap — grounded
in the retrieved embassy-rules chunks.

**Rejection risk score (pure model, no LLM call):**
```bash
curl -X POST http://localhost:8000/risk-score \
  -H "Content-Type: application/json" \
  -d '{
    "visa_type": "us_h1b",
    "doc_completeness": 0.5,
    "passport_buffer_days": 30,
    "financial_ratio": 0.8,
    "ties_strength": 0.4,
    "days_to_travel": 20,
    "prior_rejection": 1
  }'
```

**Risk score with a natural-language explanation (via `/chat`):**
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "query": "How likely am I to get rejected and why?",
    "visa_type": "us_h1b",
    "application_details": {
      "visa_type": "us_h1b",
      "doc_completeness": 0.5,
      "passport_buffer_days": 30,
      "financial_ratio": 0.8,
      "ties_strength": 0.4,
      "days_to_travel": 20,
      "prior_rejection": 1
    }
  }'
```

## Extending it

- Swap in real embassy/company docs by dropping more `.md` files into
  `data/policies/` and re-running `python3 ingest.py`.
- Add new visa types by adding entries to `data/checklists.json` (and adding matching
  rows to the synthetic data generator if you want risk scoring for them too).
- Swap Groq for any other LangChain-compatible chat model by editing `rag_graph.py`.
- Add conversation memory by extending `VisaState` with a message history list and
  wiring it into the prompts.
- Replace the synthetic risk-training data with a real, labeled dataset when available.

## Known limitations (documented, not hidden)

- The `classify_intent` node uses a keyword heuristic rather than a learned
  classifier — good enough for a demo, but a real system would train/prompt a small
  classifier or add few-shot intent examples for more robust routing.
- The risk model is trained on **synthetic** data with hand-specified feature
  relationships, not real embassy outcomes — it demonstrates the ML system design
  (feature engineering, explainable model, integration into a RAG pipeline) rather
  than making claims about real-world rejection rates.
t routing on ambiguous queries.
