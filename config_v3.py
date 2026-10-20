"""Feature-gated v3 runtime configuration; never changes v2 deployment defaults."""
import os
from config import (
    GROQ_MODEL, GROQ_API_KEY, GROQ_EVAL_MODEL, GROQ_EVAL_API_KEY,
    EMBEDDING_MODEL, PINECONE_INDEX_NAME,
)

V3_ENABLED = os.getenv("SELF_RAG_VERSION", "2").strip() == "3"
V3_NAMESPACE = "portfolio-kb-v3-staging"
V3_MAX_REVISIONS = int(os.getenv("V3_MAX_REVISIONS", "1"))
