"""Feature-gated v3 runtime configuration; never changes v2 deployment defaults."""
import os
from config import (
    GROQ_MODEL, GROQ_API_KEY, GROQ_EVAL_MODEL, GROQ_EVAL_API_KEY,
    EMBEDDING_MODEL, PINECONE_INDEX_NAME,
)

V3_ENABLED = os.getenv("SELF_RAG_VERSION", "2").strip() == "3"
V3_NAMESPACE = "portfolio-kb-v3-staging"
V3_MAX_REVISIONS = int(os.getenv("V3_MAX_REVISIONS", "1"))
V3_MAX_REWRITES = int(os.getenv("V3_MAX_REWRITES", "1"))
# Independent evaluator calls are expensive. Keep grounding, skip redundant
# batch relevance and usefulness LLM calls by default; can enable for A/B.
V3_GRADE_RELEVANCE = os.getenv("V3_GRADE_RELEVANCE", "false").lower() == "true"
V3_GRADE_USEFULNESS = os.getenv("V3_GRADE_USEFULNESS", "false").lower() == "true"
