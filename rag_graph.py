"""
rag_graph.py — LangGraph pipeline for the AI Visa Assistant.

Nodes:
  classify_intent -> route to either:
    - general_rag        (answers FAQs / policy questions using retrieved chunks)
    - missing_doc_explain (checks user's documents against the checklist, explains gaps
                            with reasons pulled from the retrieved policy text)
  Both paths converge to a final language-aware response.

Uses Groq (free/fast Llama 3.3 70B) as the LLM so the whole thing is cheap and
mostly-local: only the generation call leaves the machine, embeddings + retrieval + FAISS
all run on-device.
"""
import os
import json
from pathlib import Path
from typing import TypedDict, List, Optional

from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq

from retriever import get_retriever
from risk_model import get_risk_model, ApplicationInput

CHECKLIST_PATH = Path(__file__).parent / "data" / "checklists.json"
with open(CHECKLIST_PATH, "r", encoding="utf-8") as f:
    CHECKLISTS = json.load(f)

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.2,
    api_key=os.environ.get("GROQ_API_KEY"),
)


class VisaState(TypedDict, total=False):
    query: str
    language: str
    visa_type: Optional[str]           # e.g. "us_h1b", "schengen_tourist"
    user_documents: Optional[List[str]]  # doc ids the user says they already have
    application_details: Optional[dict]  # ApplicationInput fields for risk scoring
    intent: str                         # "general_rag" | "missing_doc" | "risk_score"
    retrieved: List[dict]
    risk_result: Optional[dict]
    answer: str


def classify_intent(state: VisaState) -> VisaState:
    """Cheap heuristic + LLM fallback to decide the question type."""
    q = state["query"].lower()
    risk_signals = ["risk", "chance", "likely to be rejected", "probability",
                    "odds", "how likely", "will i get rejected", "approval chance"]
    gap_signals = ["missing", "why do i need", "don't have", "do not have",
                   "rejected", "what documents", "checklist", "which documents"]

    if state.get("application_details") and any(s in q for s in risk_signals):
        state["intent"] = "risk_score"
    elif state.get("visa_type") and any(s in q for s in gap_signals):
        state["intent"] = "missing_doc"
    else:
        state["intent"] = "general_rag"
    return state


def retrieve_node(state: VisaState) -> VisaState:
    retriever = get_retriever()
    state["retrieved"] = retriever.retrieve(state["query"], k=4)
    return state


def general_rag_node(state: VisaState) -> VisaState:
    context = "\n\n".join(
        f"[{c['source']}] {c['text']}" for c in state["retrieved"]
    )
    lang = state.get("language", "English")
    prompt = f"""You are a visa assistant. Answer the user's question ONLY using the context below.
If the context doesn't contain the answer, say so honestly instead of guessing.
Respond in {lang}. Cite the source file name in brackets when you use a fact.

Context:
{context}

Question: {state['query']}

Answer:"""
    resp = llm.invoke(prompt)
    state["answer"] = resp.content
    return state


def missing_doc_node(state: VisaState) -> VisaState:
    visa_type = state.get("visa_type")
    checklist = CHECKLISTS.get(visa_type)
    lang = state.get("language", "English")

    if not checklist:
        state["answer"] = (
            f"I don't have a document checklist for visa type '{visa_type}' yet. "
            f"Available types: {', '.join(CHECKLISTS.keys())}."
        )
        return state

    have = set(state.get("user_documents") or [])
    required = checklist["required_documents"]
    missing = [d for d in required if d["id"] not in have]

    context = "\n\n".join(f"[{c['source']}] {c['text']}" for c in state["retrieved"])

    if not missing:
        gap_summary = "The user appears to have all required documents for this visa type."
    else:
        gap_summary = "Missing documents:\n" + "\n".join(
            f"- {d['name']}: {d['why']}" for d in missing
        )

    prompt = f"""You are a visa assistant explaining document gaps to an applicant, for {checklist['label']}.

{gap_summary}

Supporting policy context (for grounding, cite source file names when relevant):
{context}

User's question: {state['query']}

Write a clear, empathetic explanation in {lang}. For each missing document, explain
WHY it's required (not just that it's missing) and what specifically could go wrong
without it. Keep it concise and actionable — end with a short checklist of next steps."""
    resp = llm.invoke(prompt)
    state["answer"] = resp.content
    return state


def risk_score_node(state: VisaState) -> VisaState:
    lang = state.get("language", "English")
    app_details: ApplicationInput = state.get("application_details") or {}
    app_details.setdefault("visa_type", state.get("visa_type"))

    risk_model = get_risk_model()
    result = risk_model.score(app_details)
    state["risk_result"] = result

    context = "\n\n".join(f"[{c['source']}] {c['text']}" for c in state.get("retrieved", []))
    factors_text = "\n".join(
        f"- {f['factor']} (value: {f['value']}) {f['direction']} rejection risk"
        for f in result["top_factors"]
    )

    prompt = f"""You are a visa assistant giving a risk assessment. A model estimated this
application's rejection probability at {result['rejection_probability']:.0%} ({result['risk_level']} risk).

Top contributing factors (from the model, most important first):
{factors_text}

Supporting policy context (for grounding specific advice, cite source file names when relevant):
{context}

User's question: {state['query']}

Write a clear, honest, non-alarmist explanation in {lang}. State the risk level and
percentage plainly, explain what's driving it in plain language, and give 2-3 concrete,
actionable steps to reduce the risk. Make clear this is a data-driven estimate, not a
guarantee or an official decision."""
    resp = llm.invoke(prompt)
    state["answer"] = resp.content
    return state


def route_intent(state: VisaState) -> str:
    return state["intent"]


def build_graph():
    graph = StateGraph(VisaState)
    graph.add_node("classify_intent", classify_intent)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("general_rag", general_rag_node)
    graph.add_node("missing_doc", missing_doc_node)
    graph.add_node("risk_score", risk_score_node)

    graph.set_entry_point("classify_intent")
    graph.add_edge("classify_intent", "retrieve")
    graph.add_conditional_edges(
        "retrieve",
        route_intent,
        {"general_rag": "general_rag", "missing_doc": "missing_doc", "risk_score": "risk_score"},
    )
    graph.add_edge("general_rag", END)
    graph.add_edge("missing_doc", END)
    graph.add_edge("risk_score", END)
    return graph.compile()


_graph_singleton = None


def get_graph():
    global _graph_singleton
    if _graph_singleton is None:
        _graph_singleton = build_graph()
    return _graph_singleton
