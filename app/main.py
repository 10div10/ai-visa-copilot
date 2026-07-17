"""
app/main.py — FastAPI server for the AI Visa Assistant.

Run:
    uvicorn app.main:app --reload --port 8000

Endpoints:
    POST /chat            -> general or gap-analysis Q&A (LangGraph pipeline)
    GET  /visa-types       -> list supported visa types + their checklists
    GET  /health
"""
import json
import sys
from pathlib import Path
from typing import List, Optional

sys.path.append(str(Path(__file__).parent.parent))  # allow importing sibling modules

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from rag_graph import get_graph, CHECKLISTS

app = FastAPI(title="AI Visa Assistant", version="0.1.0")


class ChatRequest(BaseModel):
    query: str
    language: Optional[str] = "English"
    visa_type: Optional[str] = None            # e.g. "us_h1b"
    user_documents: Optional[List[str]] = None  # doc ids the applicant already has
    application_details: Optional[dict] = None  # for risk-score questions


class ChatResponse(BaseModel):
    answer: str
    intent: str
    sources: List[str]
    risk_result: Optional[dict] = None


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/visa-types")
def visa_types():
    return {
        vid: {"label": v["label"], "documents": [d["id"] for d in v["required_documents"]]}
        for vid, v in CHECKLISTS.items()
    }


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="query must not be empty")

    graph = get_graph()
    result = graph.invoke({
        "query": req.query,
        "language": req.language,
        "visa_type": req.visa_type,
        "user_documents": req.user_documents or [],
        "application_details": req.application_details,
    })

    sources = sorted({c["source"] for c in result.get("retrieved", [])})
    return ChatResponse(
        answer=result["answer"],
        intent=result["intent"],
        sources=sources,
        risk_result=result.get("risk_result"),
    )


class RiskScoreRequest(BaseModel):
    visa_type: str
    doc_completeness: float = 1.0
    passport_buffer_days: int = 180
    financial_ratio: float = 1.5
    ties_strength: float = 0.7
    days_to_travel: int = 45
    prior_rejection: int = 0


@app.post("/risk-score")
def risk_score(req: RiskScoreRequest):
    """Direct numeric risk-scoring endpoint (bypasses the LLM/chat layer)."""
    from risk_model import get_risk_model
    model = get_risk_model()
    return model.score(req.dict())
