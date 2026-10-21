"""Deterministic evidence selection for v3 retrieved candidates.

No LLM calls. One broad vector search supplies candidates; optional fallback for
explicitly named missing projects is handled by retriever_v3.
"""
from __future__ import annotations

from collections import Counter
import re
from typing import Any

STOPWORDS = {"what", "which", "who", "with", "about", "best", "projects", "project",
             "tell", "explain", "does", "have", "more", "than", "into", "this",
             "that", "from", "they", "their", "navnit", "engineering", "demonstrate"}


def _terms(text: str) -> set[str]:
    return {term for term in re.findall(r"[a-z0-9]+", text.casefold())
            if len(term) > 2 and term not in STOPWORDS}


def _evidence_score(doc: Any, query: str, rank: int) -> float:
    meta = doc.metadata
    body = doc.page_content or ""
    kind = meta.get("section_type", "")
    lower = body.casefold()
    # Preserve similarity relevance through rank; modest evidence quality boosts.
    score = 8.0 / (rank + 4)
    if kind == "project_knowledge":
        score += 2.5
    elif kind == "capability_comparison":
        score += 1.6
    elif kind == "professional_profile":
        score += 0.5
    elif kind == "conversation_reference":
        score += 0.1
    elif kind == "integrity_and_operations":
        score -= 1.8

    # Headings help distinguish implementation facts from suggested questions.
    if "**implementation.**" in lower or "**architecture" in lower:
        score += 2.0
    if "**engineering strengths.**" in lower:
        score += 0.8
    if "**good related questions:**" in lower:
        score -= 2.5
    if "**evidence and metrics.**" in lower and not (
        {"metric", "dice", "accuracy", "performance", "result"} & _terms(query)
    ):
        score -= 0.6

    overlap = len(_terms(query) & _terms(body))
    score += min(overlap, 5) * 0.22
    return score


def choose_evidence(docs: list[Any], query: str, *, limit: int = 8,
                    required_projects: tuple[str, ...] = ()) -> list[Any]:
    """Return evidence-rich, multi-project results preserving original doc objects.

    Favors named required projects if present in the candidate set. Missing
    projects cannot be invented by local reranking.
    """
    if limit <= 0:
        raise ValueError("limit must be positive")
    ranked = sorted(enumerate(docs),
                    key=lambda pair: (-_evidence_score(pair[1], query, pair[0]), pair[0]))
    chosen: list[Any] = []
    selected_idx: set[int] = set()
    project_counts: Counter[str] = Counter()
    general_count = 0

    def add(idx: int, doc: Any) -> None:
        nonlocal general_count
        chosen.append(doc)
        selected_idx.add(idx)
        project = doc.metadata.get("project_id") or ""
        if project:
            project_counts[project] += 1
        else:
            general_count += 1

    # Required means explicit mentions in the question, not unconditional
    # favorites. Reserve at most one project evidence chunk for each.
    for needed in required_projects[:limit]:
        match = next(((i, d) for i, d in ranked
                      if i not in selected_idx
                      and d.metadata.get("project_id") == needed), None)
        if match:
            add(*match)

    # First pass: cover distinct evidence-bearing projects and useful guidance.
    for i, doc in ranked:
        if len(chosen) >= limit:
            break
        if i in selected_idx:
            continue
        project = doc.metadata.get("project_id") or ""
        if project and project_counts[project] == 0:
            add(i, doc)
        elif not project and general_count < 2 and doc.metadata.get("section_type") in {
            "capability_comparison", "professional_profile",
        }:
            add(i, doc)

    # Second pass: fill gaps with the strongest remaining evidence.
    for i, doc in ranked:
        if len(chosen) >= limit:
            break
        if i in selected_idx:
            continue
        project = doc.metadata.get("project_id") or ""
        if project and project_counts[project] >= 2:
            continue
        if not project and general_count >= 2:
            continue
        add(i, doc)

    return chosen
