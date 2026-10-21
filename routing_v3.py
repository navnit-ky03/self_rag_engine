"""Small deterministic v3 routing helpers: no API calls, safe to unit test."""
from __future__ import annotations
import re

from retriever_v3 import explicit_project_ids

_DIRECT = re.compile(
    r"^\s*(?:(?:hi|hello|hey|good morning|good evening|good afternoon)"
    r"(?:\s+(?:there|assistant))?[!.\s]*|(?:thanks|thank you|bye|goodbye)[!.\s]*)$",
    re.IGNORECASE,
)
_TECHNICAL_GENERAL = re.compile(
    r"^\s*(?:what is|what are|define|explain)\s+"
    r"(?:an?\s+)?(?:rag|retrieval augmented generation|"
    r"transformers?|machine learning|deep learning|"
    r"vector embeddings?|langgraph|pinecone)\s*\??\s*$",
    re.IGNORECASE,
)
_FOLLOWUP = re.compile(
    r"\b(?:it|its|that|those|they|them|their|he|his|"
    r"the (?:first|second|third|last|previous|other) one|"
    r"this project|that project|both projects|"
    r"the (?:former|latter))\b",
    re.IGNORECASE,
)


def requires_portfolio_retrieval(question: str) -> bool:
    """Conservative: only self-contained greetings and generic definitions bypass KB."""
    if not question.strip():
        raise ValueError("Question must not be empty")
    return not (_DIRECT.fullmatch(question) or _TECHNICAL_GENERAL.fullmatch(question))


def needs_context_rewrite(question: str, history: list[dict] | None) -> bool:
    """Only pay for a context-rewrite call for likely pronoun/ellipsis follow-ups.

    An explicitly named project is authoritative over earlier thread topics.
    """
    if not history or explicit_project_ids(question):
        return False
    return bool(_FOLLOWUP.search(question))
