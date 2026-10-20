import time
from typing import Literal
from langchain_groq import ChatGroq

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from config import (
    GROQ_MODEL,
    GROQ_API_KEY,
    GROQ_EVAL_MODEL,
    GROQ_EVAL_API_KEY
)
from schemas import State, DecisionState, relevantstate, issup, isUse, retrieval,BatchRelevanceState
from retriever import retriver
from prompts import (
    Decision_Prompt,
    direct_generation_prompt,
    is_relevant_prompt,
    rag_generation_prompt,
    issup_prompt,
    revise_prompt,
    isuse_prompt,
    rewrite_for_retrieval_prompt,
    batch_relevance_prompt,
)

memory = MemorySaver()

# Primary Groq model: Answer generation and revision
llm = ChatGroq(
    model=GROQ_MODEL,
    api_key=GROQ_API_KEY,
    temperature=0.2
)

# Secondary Groq model: Evaluation and routing
evaluation_llm = ChatGroq(
    model=GROQ_EVAL_MODEL,
    api_key=GROQ_EVAL_API_KEY,
    temperature=0
)

# Structured evaluation models
decision_llm = evaluation_llm.with_structured_output(DecisionState)

relevant_llm = evaluation_llm.with_structured_output(
    BatchRelevanceState,
    method="json_mode"
)

issup_llm = evaluation_llm.with_structured_output(issup)

isuse_llm = evaluation_llm.with_structured_output(isUse)

rewrite_llm = evaluation_llm.with_structured_output(
    retrieval,
    method="json_mode"
)


def log_node_timing(node_name: str, start_time: float, details: str = ""):
    """Helper function to print structured node execution latency to terminal."""
    elapsed = time.perf_counter() - start_time
    detail_str = f" | {details}" if details else ""
    print(f"[TIMING: NODE] {node_name:<18} | {elapsed:.3f}s ({elapsed * 1000:.1f}ms){detail_str}")


MAX_HISTORY_MESSAGES = 12

def format_history(history: list[dict]) -> str:
    if not history:
        return "No prior conversation history."
    formatted = []
    for msg in history[-MAX_HISTORY_MESSAGES:]:
        role = "User" if msg["role"] == "user" else "Assistant"
        formatted.append(f"{role}: {msg['content']}")
    return "\n".join(formatted)

def updated_history(history,question,answer):
    updated = history + [
        {"role":"user","content":question},
        {"role":"assistant","content":answer}
    ]
    return updated[-MAX_HISTORY_MESSAGES:]


# NODE FUNCTIONS WITH TIMING INSTRUMENTATION
def Decision_Node(state: State):
    t0 = time.perf_counter()
    history_str = format_history(state.get("chat_history", []))
    decision = decision_llm.invoke(Decision_Prompt.format_messages(
        question=state['question'],
        chat_history=history_str
    ))
    log_node_timing("Decision_Node", t0, f"need_retrieval={decision.need_retrival}")
    return {
    "need_retrival": decision.need_retrival,

    # Reset retry counters
    "retries": 0,
    "retrieval_tries": 0,

    # Reset retrieval state
    "retrieval_query": "",
    "docs": [],
    "relevant_docs": [],

    # Reset answer-generation state
    "context": "",
    "answer": "",
    "evidence": [],
    "reason": "",
    "rewrite_unchanged": False,
    }


def Direct_Answer(state: State):
    t0 = time.perf_counter()
    history_str = format_history(state.get("chat_history", []))
    ans = llm.invoke(
        direct_generation_prompt.format_messages(
            question=state['question'],
            chat_history=history_str
        )
    )
    update_history = updated_history(
        state.get("chat_history",[]),
        state["question"],
        ans.content
    )

    log_node_timing("Direct_Answer", t0, f"Generated {len(ans.content)} chars")
    return {"answer": ans.content, "chat_history": update_history}

def resolve_context(state: State):
    history = state.get("chat_history") or []

    if not history:
        return {"retrieval_query": state["question"]}

    context = rewrite_llm.invoke(
        rewrite_for_retrieval_prompt.format_messages(
            question=state["question"],
            retrieval_query="",
            answer="",
            chat_history=format_history(history)
        )
    )

    return {"retrieval_query": context.retrieval_query}

def retrive(state: State):
    t0 = time.perf_counter()
    query = state.get("retrieval_query") or state["question"]
    docs = retriver.invoke(query)
    log_node_timing("retrive", t0, f"Fetched {len(docs)} docs for: '{query[:30]}...'")
    return {"docs": docs}


def relevant(state: State):
    t0 = time.perf_counter()
    docs = state.get("docs", [])

    if not docs:
        log_node_timing("relevant", t0, "0 docs to grade")
        return {"relevant_docs": []}

    # Combine retrieved documents into one prompt
    documents_text = "\n\n".join(
        f"Document {i + 1}:\n{doc.page_content}"
        for i, doc in enumerate(docs)
    )

    # Single Groq evaluation request
    decision = relevant_llm.invoke(
        batch_relevance_prompt.format_messages(
            question=state.get("retrieval_query") or state["question"],
            documents=documents_text
        )
    )

    # Validate the number of returned decisions
    if len(decision.relevant) != len(docs):
        raise ValueError(
            f"Expected {len(docs)} relevance decisions, "
            f"received {len(decision.relevant)}"
        )

    # Keep only relevant documents
    relevant_docs = [
        doc
        for doc, is_relevant in zip(docs, decision.relevant)
        if is_relevant
    ]

    log_node_timing(
        "relevant",
        t0,
        f"Kept {len(relevant_docs)}/{len(docs)} relevant docs"
    )

    return {"relevant_docs": relevant_docs}


def context_ans(state: State):
    t0 = time.perf_counter()
    context = "\n\n---\n\n".join(
        [d.page_content for d in state.get("relevant_docs", [])]
    ).strip()

    if not context:
        fallback_msg = "I couldn't find specific details in Navnit's records to answer that accurately. Feel free to ask about his featured projects!"
        history = updated_history(
    state.get("chat_history") or [],
    state["question"],
    fallback_msg
)
        log_node_timing("context_ans", t0, "Empty context -> fallback triggered")
        return {
            "answer": fallback_msg,
            "context": "",
            "chat_history": history
        }

    history_str = format_history(state.get("chat_history", []))
    ans = llm.invoke(rag_generation_prompt.format_messages(
        question=state["question"],
        context=context,
        chat_history=history_str
    ))

    log_node_timing("context_ans", t0, f"Generated {len(ans.content)} chars")
    return {
        "answer": ans.content,
        "context": context
    }


def no_context(state: State):
    t0 = time.perf_counter()

    fallback_msg = (
        "I couldn't verify the requested details from "
        "Navnit's available knowledge base, so I can't "
        "provide a reliable answer to that question."
    )

    update_history = updated_history(
            state.get("chat_history",[]),
            state["question"],
            fallback_msg
        )

    log_node_timing("no_context", t0, "Fallback triggered")

    return {
        "answer": fallback_msg,
        "context": "",
        "chat_history": update_history,
    }


def issup_node(state: State):
    t0 = time.perf_counter()
    question = state["question"]
    answer = state["answer"]
    context = state["context"]
    decision = issup_llm.invoke(
        issup_prompt.format_messages(
            question=question,
            answer=answer,
            context=context,
        )
    )
    log_node_timing("issup_node", t0, f"Status: {decision.issup}")
    return {"issup": decision.issup, "evidence": decision.evidence}


def revise_ans(state: State):
    t0 = time.perf_counter()
    question = state["question"]
    answer = state["answer"]
    context = state["context"]
    evidence_str = "\n".join(state.get("evidence", [])) or "None extracted yet."

    decision = llm.invoke(
        revise_prompt.format_messages(
            question=question,
            answer=answer,
            context=context,
            evidence=evidence_str,
        )
    )

    log_node_timing("revise_ans", t0, f"Retry count: {state.get('retries', 0) + 1}")
    return {"answer": decision.content, "retries": state.get("retries", 0) + 1}


def isuse_node(state: State):
    t0 = time.perf_counter()
    decision = isuse_llm.invoke(
    isuse_prompt.format_messages(
        question=state.get("retrieval_query") or state["question"],
        answer=state.get("answer", ""),
    )
)

    res = {
        "isuse": decision.isuse,
        "reason": decision.reason
    }
    print(f"[USEFULNESS] Question: {state.get('retrieval_query') or state['question']}")
    print(f"[USEFULNESS] Answer: {state.get('answer', '')[:500]}")
    print(f"[USEFULNESS] Decision: {decision.isuse}")
    print(f"[USEFULNESS] Reason: {decision.reason}")

    if decision.isuse == "useful":
        update_history = updated_history(
        state.get("chat_history", []),
        state["question"],
        state["answer"]
        )
        res["chat_history"] = update_history

    log_node_timing("isuse_node", t0, f"Usefulness: {decision.isuse}")
    return res


def rewrite(state: State):
    t0 = time.perf_counter()

    history_str = format_history(state.get("chat_history") or [])

    out = rewrite_llm.invoke(
        rewrite_for_retrieval_prompt.format_messages(
            question=state["question"],
            retrieval_query=state.get("retrieval_query", ""),
            answer=state.get("answer", ""),
            chat_history=history_str
        )
    )

    previous_query = state.get("retrieval_query") or state["question"]
    new_query = out.retrieval_query.strip()

    if new_query.casefold() == previous_query.strip().casefold():
        print("[REWRITE] Query unchanged — skipping repeated retrieval")

        return {
            "retrieval_query": previous_query,
            "rewrite_unchanged": True
        }

    log_node_timing(
        "rewrite",
        t0,
        f"New Query: '{new_query[:30]}...' | "
        f"Try: {state.get('retrieval_tries', 0) + 1}"
    )

    return {
        "retrieval_query": new_query,
        "retrieval_tries": state.get("retrieval_tries", 0) + 1,
        "rewrite_unchanged": False
    }


# ROUTER FUNCTIONS
def decision_router(state: State) -> Literal["Direct_Answer", "resolve_context"]:
    if state['need_retrival']:
        return "resolve_context"
    else:
        return "Direct_Answer"


def context_router(state: State) -> Literal["context_ans", "no_context"]:
    if len(state["relevant_docs"]) > 0:
        return "context_ans"
    return "no_context"


max_retries = 2


def issup_router(state: State) -> Literal["isuse", "revise_ans", "no_context"]:
    status = state.get("issup")
    if status == "fully_supported":
        return "isuse"

    count = state.get("retries", 0)

    if count >= max_retries:
        return "no_context"

    return "revise_ans"


max_rewrites = 2



def isuse_router(state: State) -> Literal[
    "END", "rewrite", "no_context"
]:
    if state.get("isuse") == "useful":
        return "END"

    if state.get("retrieval_tries", 0) >= max_rewrites:
        return "no_context"

    return "rewrite"

def rewrite_router(state: State) -> Literal["retrive", "no_context"]:

    if state.get("rewrite_unchanged", False):
        return "no_context"

    return "retrive"

# GRAPH CONSTRUCTION
graph = StateGraph(State)

graph.add_node("decision_node", Decision_Node)
graph.add_node("resolve_context",resolve_context)
graph.add_node("direct_ans", Direct_Answer)
graph.add_node("retrive", retrive)
graph.add_node("relevant", relevant)
graph.add_node("context_ans", context_ans)
graph.add_node("no_context", no_context)
graph.add_node("issup", issup_node)
graph.add_node("revise_ans", revise_ans)
graph.add_node("isuse", isuse_node)
graph.add_node("rewrite", rewrite)

graph.add_edge(START, "decision_node")
graph.add_conditional_edges("decision_node", decision_router, {
    "Direct_Answer": "direct_ans",
    "resolve_context": "resolve_context",
})

graph.add_edge("direct_ans", END)
graph.add_edge("resolve_context","retrive" )
graph.add_edge("retrive", "relevant")
graph.add_conditional_edges("relevant", context_router, {
    "context_ans": "context_ans",
    "no_context": "no_context",
})

graph.add_edge("context_ans", "issup")
graph.add_conditional_edges("issup", issup_router, {
    "isuse": "isuse",
    "revise_ans": "revise_ans",
    "no_context":"no_context"
})
graph.add_edge("revise_ans", "issup")
graph.add_conditional_edges("isuse", isuse_router, {
    "END": END,
    "rewrite": "rewrite",
    "no_context": "no_context"
})
graph.add_conditional_edges(
    "rewrite",
    rewrite_router,
    {
        "retrive": "retrive",
        "no_context": "no_context"
    }
)
graph.add_edge("no_context", END)

app = graph.compile(checkpointer=memory)