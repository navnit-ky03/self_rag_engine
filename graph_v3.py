"""Isolated Self-RAG v3 LangGraph: opt-in only; production v2 graph stays intact.

V3 retains a groundedness evaluator and one bounded repair, while dropping two
redundant sequential LLM judges (batch relevance + usefulness). Retrieval uses
the validated v3 staging namespace. No external calls at import besides LLM
client construction; Pinecone is connected on first retrieval.
"""
from __future__ import annotations

import time
from typing import Literal, TypedDict

from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from config_v3 import (
    GROQ_MODEL, GROQ_API_KEY, GROQ_EVAL_MODEL, GROQ_EVAL_API_KEY,
    V3_MAX_REVISIONS,
)
from schemas import State, issup, retrieval
from retriever_v3 import retrieve
from routing_v3 import requires_portfolio_retrieval, needs_context_rewrite
from prompts_v3 import DIRECT, GENERATION, REWRITE, GROUNDING, REVISION
from medical_safety_v3 import ensure_safe_medical_claims

MAX_HISTORY_MESSAGES = 12


class V3State(State, total=False):
    retrieval_profile: str
    retrieval_stats: dict
    grounding_status: str
    revisions: int
    medical_safety_intervened: bool


def _history_text(history: list[dict]) -> str:
    if not history:
        return "No previous conversation."
    return "\n".join(
        ("User" if m.get("role") == "user" else "Assistant")
        + ": " + str(m.get("content", ""))
        for m in history[-MAX_HISTORY_MESSAGES:]
    )


def _store_turn(state: V3State) -> dict:
    history = list(state.get("chat_history") or [])
    history.extend([
        {"role": "user", "content": state["question"]},
        {"role": "assistant", "content": state["answer"]},
    ])
    return {"chat_history": history[-MAX_HISTORY_MESSAGES:]}


def _timing(label: str, start: float, info: str = "") -> None:
    print(f"[V3 TIMING] {label}: {(time.perf_counter()-start)*1000:.0f}ms {info}")


llm = ChatGroq(model=GROQ_MODEL, api_key=GROQ_API_KEY, temperature=0.2)
eval_llm = ChatGroq(
    model=GROQ_EVAL_MODEL,
    api_key=GROQ_EVAL_API_KEY or GROQ_API_KEY,
    temperature=0,
)
rewrite_llm = eval_llm.with_structured_output(retrieval, method="json_mode")
support_llm = eval_llm.with_structured_output(issup)


def route_question(state: V3State) -> dict:
    question = state["question"]
    return {
        "need_retrival": requires_portfolio_retrieval(question),
        "retrieval_query": question,
        "docs": [],
        "relevant_docs": [],
        "context": "",
        "answer": "",
        "revisions": 0,
        "grounding_status": "not_checked",
        "retrieval_profile": "",
        "retrieval_stats": {},
    }


def question_route(state: V3State) -> Literal["direct_ans", "resolve_context"]:
    return "resolve_context" if state.get("need_retrival", True) else "direct_ans"


def direct_ans(state: V3State) -> dict:
    t0 = time.perf_counter()
    answer = llm.invoke(DIRECT.format_messages(
        question=state["question"],
        chat_history=_history_text(state.get("chat_history") or []),
    )).content
    _timing("direct_ans", t0)
    return {"answer": answer}


def resolve_context(state: V3State) -> dict:
    t0 = time.perf_counter()
    question = state["question"]
    history = state.get("chat_history") or []
    if not needs_context_rewrite(question, history):
        return {"retrieval_query": question}
    result = rewrite_llm.invoke(REWRITE.format_messages(
        question=question,
        chat_history=_history_text(history),
    ))
    rewritten = result.retrieval_query.strip() or question
    _timing("resolve_context", t0, f"rewritten={rewritten != question}")
    return {"retrieval_query": rewritten}


def retrive(state: V3State) -> dict:
    t0 = time.perf_counter()
    query = state.get("retrieval_query") or state["question"]
    docs, diagnostics = retrieve(query)
    _timing("retrive", t0, f"profile={diagnostics['profile']} docs={len(docs)}")
    return {
        "docs": docs,
        "relevant_docs": docs,
        "retrieval_profile": diagnostics["profile"],
        "retrieval_stats": diagnostics,
    }


def evidence_route(state: V3State) -> Literal["context_ans", "no_context"]:
    return "context_ans" if state.get("relevant_docs") else "no_context"


def context_ans(state: V3State) -> dict:
    t0 = time.perf_counter()
    sections = []
    for d in state.get("relevant_docs") or []:
        meta = d.metadata
        source = meta.get("source", "KB")
        section = meta.get("section_id", "unknown")
        project = meta.get("project_id") or "general"
        sections.append(
            f"[Source: {source}, section {section}, project {project}]\n"
            f"{d.page_content}"
        )
    context = "\n\n---\n\n".join(sections).strip()
    if not context:
        return {"context": "", "answer": "I couldn't verify that information from Navnit's available project documentation."}
    answer = llm.invoke(GENERATION.format_messages(
        question=state["question"], context=context,
        chat_history=_history_text(state.get("chat_history") or []),
    )).content
    _timing("context_ans", t0, f"chars={len(answer)}")
    return {"answer": answer, "context": context}


def no_context(state: V3State) -> dict:
    return {
        "answer": "I couldn't verify the requested details in Navnit's available project documentation.",
        "context": "",
        "grounding_status": "not_checked",
    }


def issup_node(state: V3State) -> dict:
    t0 = time.perf_counter()
    decision = support_llm.invoke(GROUNDING.format_messages(
        question=state["question"], answer=state["answer"],
        context=state.get("context") or "",
    ))
    _timing("issup", t0, decision.issup)
    return {"grounding_status": decision.issup, "evidence": decision.evidence}


def support_route(state: V3State) -> Literal["finalize", "revise_ans"]:
    if state.get("grounding_status") == "fully_supported":
        return "finalize"
    if (state.get("revisions") or 0) < V3_MAX_REVISIONS:
        return "revise_ans"
    return "finalize"


def revise_ans(state: V3State) -> dict:
    t0 = time.perf_counter()
    answer = llm.invoke(REVISION.format_messages(
        question=state["question"], answer=state["answer"],
        context=state.get("context") or "",
        evidence="\n".join(state.get("evidence") or []) or "No excerpts verified.",
    )).content
    _timing("revise_ans", t0)
    return {
        "answer": answer,
        "revisions": (state.get("revisions") or 0) + 1,
        "grounding_status": "not_checked",
    }


def finalize(state: V3State) -> dict:
    status = state.get("grounding_status")
    if state.get("need_retrival", True) and status != "fully_supported" and state.get("context"):
        # Do not leak an unverified draft when checking exhausts its budget.
        answer = (
            "I couldn't verify enough evidence to answer that reliably. "
            "You can ask about a specific project or its implementation."
        )
    else:
        answer = state["answer"]

    # Explicit clinical safety gate for the model-generated final answer.
    # Run AFTER grounding, so an overly permissive model judge cannot approve
    # claims that the project is clinically validated or ready for patient use.
    answer, guarded = ensure_safe_medical_claims(answer)
    if guarded:
        print("[V3 SAFETY] Unsupported NeuroScan clinical claim replaced.")
    final_state = {**state, "answer": answer}
    return {
        "answer": answer,
        "chat_history": _store_turn(final_state)["chat_history"],
        "medical_safety_intervened": guarded,
    }


graph = StateGraph(V3State)
graph.add_node("route_question", route_question)
graph.add_node("direct_ans", direct_ans)
graph.add_node("resolve_context", resolve_context)
graph.add_node("retrive", retrive)
graph.add_node("context_ans", context_ans)
graph.add_node("no_context", no_context)
graph.add_node("issup", issup_node)
graph.add_node("revise_ans", revise_ans)
graph.add_node("finalize", finalize)

graph.add_edge(START, "route_question")
graph.add_conditional_edges("route_question", question_route, {
    "direct_ans": "direct_ans", "resolve_context": "resolve_context",
})
graph.add_edge("direct_ans", "finalize")
graph.add_edge("resolve_context", "retrive")
graph.add_conditional_edges("retrive", evidence_route, {
    "context_ans": "context_ans", "no_context": "no_context",
})
graph.add_edge("context_ans", "issup")
graph.add_conditional_edges("issup", support_route, {
    "finalize": "finalize", "revise_ans": "revise_ans",
})
graph.add_edge("revise_ans", "issup")
graph.add_edge("no_context", "finalize")
graph.add_edge("finalize", END)

app = graph.compile(checkpointer=MemorySaver())
