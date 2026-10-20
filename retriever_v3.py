"""Isolated Self-RAG v3 retrieval strategies; v2 retriever remains unchanged.

No network calls at import time. A store is created only when build_retriever()
is explicitly invoked. All defaults target the v3 staging namespace.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import re
import time
from typing import Any, Literal

V3_NAMESPACE = "portfolio-kb-v3-staging"
RetrievalProfile = Literal["focused", "detailed", "comparative"]

@dataclass(frozen=True)
class ProfileSettings:
    k: int
    search_type: Literal["similarity", "mmr"]
    fetch_k: int = 0
    lambda_mult: float = 0.5

PROFILES: dict[str, ProfileSettings] = {
    "focused": ProfileSettings(k=4, search_type="similarity"),
    "detailed": ProfileSettings(k=6, search_type="similarity"),
    "comparative": ProfileSettings(k=8, search_type="similarity", fetch_k=18, lambda_mult=0.55),
}

# Match explicit entities only. Pronouns and elliptical follow-ups require
# graph-level context resolution before using the retriever.
PROJECT_ALIASES: dict[str, tuple[str, ...]] = {
    "self-corrective-rag": (
        "self-corrective rag", "self correcting rag", "self-rag",
        "self rag", "portfolio master rag", "self_rag_engine", "rag engine",
    ),
    "neuroscan-ai": ("neuroscan", "brain tumor segmentation", "3d brain mri"),
    "voice-booking": ("voice booking", "booking assistant", "aura health", "n8n booking"),
    "lecture-rag": ("lecture rag", "lecture assistant", "sigma course", "teaching rag"),
    "customer-value": ("customer clv", "customer lifetime value", "churn and clv", "churn model"),
    "fraud-detection": ("fraud detection", "credit card scam", "anti-fraud"),
    "delivery-analytics": ("delivery analytics", "delivery time prediction", "quick-commerce"),
    "portfolio-frontend": ("portfolio frontend", "self_rag_frontend"),
}

COMPARATIVE = re.compile(
    r"\b(compare|comparison|versus|vs\.?|which projects?|what projects?|"
    r"best (?:\w+\s+){0,3}projects?|top (?:\w+\s+){0,3}projects?|strongest|most advanced|most complex|"
    r"most technical|rank|ranking|across projects?|among projects?|"
    r"multiple projects?|recommend projects?|"
    r"demonstrate (?:his |navnit'?s )?(?:skills|experience|capabilit))\b",
    flags=re.IGNORECASE,
)
DETAILED = re.compile(
    r"\b(architecture|pipeline|trade-?offs?|implementation|internals|"
    r"challenges?|design decisions?|how does|how did|why does|"
    r"deployment|end.to.end|step.by.step|explain in detail)\b",
    flags=re.IGNORECASE,
)


def _contains_phrase(text: str, phrase: str) -> bool:
    return re.search(r"(?<!\w)" + re.escape(phrase) + r"(?!\w)", text) is not None


def explicit_project_ids(query: str) -> list[str]:
    q = query.casefold()
    return [project for project, aliases in PROJECT_ALIASES.items()
            if any(_contains_phrase(q, alias) for alias in aliases)]


def choose_profile(query: str) -> RetrievalProfile:
    """Transparent conservative baseline classifier; evaluation may replace it."""
    if not query.strip():
        raise ValueError("Retrieval query must be nonempty")
    if COMPARATIVE.search(query) or len(explicit_project_ids(query)) > 1:
        return "comparative"
    if DETAILED.search(query):
        return "detailed"
    return "focused"


def make_search_kwargs(
    query: str,
    profile: RetrievalProfile | None = None,
    *,
    project_id: str | None = None,
) -> tuple[str, dict[str, Any]]:
    """Produce LangChain search kwargs, including safe project scoping.

    project_id is an explicit override from a trustworthy upstream router;
    no auto-filter is applied for broad/comparative requests.
    """
    selected = profile or choose_profile(query)
    if selected not in PROFILES:
        raise ValueError(f"Unknown retrieval profile: {selected}")
    if project_id is not None and project_id not in PROJECT_ALIASES:
        raise ValueError(f"Unknown project_id: {project_id}")
    settings = PROFILES[selected]
    # Comparative fetches a broader candidate pool in ONE search; evidence_v3
    # selects the final eight locally (no second embedding or LLM request).
    candidate_k = settings.fetch_k if selected == "comparative" and settings.search_type == "similarity" else settings.k
    kwargs: dict[str, Any] = {"k": candidate_k}
    if settings.search_type == "mmr":
        kwargs.update(fetch_k=settings.fetch_k, lambda_mult=settings.lambda_mult)

    # For focused/detailed questions we may scope only when exactly one
    # project is mentioned explicitly, or a validated project_id is provided.
    # Broad questions MUST preserve multiple project evidence.
    if selected != "comparative":
        ids = explicit_project_ids(query)
        scoped = project_id if project_id is not None else (ids[0] if len(ids) == 1 else None)
        if scoped:
            kwargs["filter"] = {"project_id": {"$eq": scoped}}
    return settings.search_type, kwargs


@lru_cache(maxsize=2)
def _cached_store(index_name: str, namespace: str) -> Any:
    """Reuse the client and embedding object across searches in one process."""
    from langchain_google_genai import GoogleGenerativeAIEmbeddings
    from langchain_pinecone import PineconeVectorStore
    from config import EMBEDDING_MODEL
    return PineconeVectorStore.from_existing_index(
        index_name=index_name,
        embedding=GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL),
        namespace=namespace,
    )


def _default_index_name() -> str:
    from config import PINECONE_INDEX_NAME
    return PINECONE_INDEX_NAME


def build_retriever(
    query: str,
    *,
    profile: RetrievalProfile | None = None,
    project_id: str | None = None,
    store: Any = None,
    index_name: str | None = None,
    namespace: str = V3_NAMESPACE,
) -> tuple[Any, dict[str, Any]]:
    """Construct retriever. Store injection enables offline mock tests."""
    if namespace != V3_NAMESPACE:
        raise ValueError("v3 retriever may only target its staging namespace")
    selected = profile or choose_profile(query)
    search_type, search_kwargs = make_search_kwargs(
        query, selected, project_id=project_id
    )
    if store is None:
        store = _cached_store(index_name or _default_index_name(), namespace)
    retriever = store.as_retriever(
        search_type=search_type, search_kwargs=search_kwargs
    )
    diagnostics = {
        "profile": selected, "search_type": search_type,
        "search_kwargs": search_kwargs, "namespace": namespace,
    }
    return retriever, diagnostics


def retrieve(
    query: str, *, profile: RetrievalProfile | None = None,
    project_id: str | None = None, store: Any = None,
    index_name: str | None = None,
) -> tuple[list[Any], dict[str, Any]]:
    """Execute one search and report metadata coverage; no answer generation."""
    t0 = time.perf_counter()
    retriever, diagnostics = build_retriever(
        query, profile=profile, project_id=project_id,
        store=store, index_name=index_name,
    )
    ready_ms = round((time.perf_counter() - t0) * 1000, 2)
    query_start = time.perf_counter()
    raw_docs = list(retriever.invoke(query))
    search_ms = round((time.perf_counter() - query_start) * 1000, 2)
    from evidence_v3 import choose_evidence
    selected = diagnostics["profile"]
    docs = choose_evidence(
        raw_docs, query,
        limit=PROFILES[selected].k,
        required_projects=tuple(explicit_project_ids(query)) if selected == "comparative" else (),
    )
    projects = sorted({d.metadata.get("project_id", "") for d in docs
                       if d.metadata.get("project_id", "")})
    diagnostics.update(
        setup_ms=ready_ms,
        search_ms=search_ms,
        candidate_count=len(raw_docs),
        result_count=len(docs),
        distinct_projects=projects,
        section_types=sorted({d.metadata.get("section_type", "")
                              for d in docs if d.metadata.get("section_type", "")}),
        elapsed_ms=round((time.perf_counter() - t0) * 1000, 2),
    )
    return docs, diagnostics
