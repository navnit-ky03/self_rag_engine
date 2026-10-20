import os
from dotenv import load_dotenv

load_dotenv()

# ==========================================
# 1. GROQ LLM CONFIGURATION
# ==========================================

# Primary Groq: Answer generation and revision
GROQ_MODEL = os.getenv(
    "GROQ_MODEL", "openai/gpt-oss-120b"
)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Secondary Groq: Evaluation and routing
GROQ_EVAL_MODEL = os.getenv(
    "GROQ_EVAL_MODEL", GROQ_MODEL
)

GROQ_EVAL_API_KEY = os.getenv("GROQ_EVAL_API_KEY")


# ==========================================
# 2. GEMINI CONFIGURATION
# ==========================================

# Retained for compatibility
GEMINI_MODEL = os.getenv(
    "GEMINI_MODEL", "gemini-3.6-flash"
)

# Google embeddings are still required
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL", "gemini-embedding-001"
)


# ==========================================
# 3. KNOWLEDGE BASE
# ==========================================

DATA_FILE_PATH = os.getenv(
    "DATA_FILE_PATH",
    "Navnit_Kumar_Yadav_Master_Knowledge_Base_v2.pdf"
)


# ==========================================
# 4. PINECONE CONFIGURATION
# ==========================================

PINECONE_INDEX_NAME = os.getenv(
    "PINECONE_INDEX_NAME",
    "portfolio-knowledge-base"
)

PINECONE_NAMESPACE = os.getenv(
    "PINECONE_NAMESPACE",
    "portfolio-kb-v2-staging"
)