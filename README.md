# Portfolio Master RAG Engine

A decoupled, self-correcting RAG (Retrieval-Augmented Generation) microservice powered by **LangGraph**, **FastAPI**, **Groq (GPT-OSS)**, **Google Gemini**, and **Pinecone**. Built as the production backend AI assistant for Navnit Kumar Yadav's personal portfolio.

The system enforces strict formatting guardrails (zero raw Markdown tables, zero robotic intro phrases), features automated query rewriting, and validates generated answers through dual groundedness (`is_sup`) and usefulness (`is_use`) evaluation loops before serving responses to the UI.

---

## 🏗️ System Architecture

The core logic operates as a **LangGraph State Machine** with conditional edge routing, iterative self-correction loops, and thread-level state persistence.

![Alt Text](https://example.com/your-image.png)

## 🔄 Graph Workflow Breakdown

1. **Intent Router (`decide_retrieval`):** Evaluates input and session history to classify whether query requires internal vector lookup or can be answered directly.
2. **Context Retrieval (`retrieve` & `is_relevant`):** Queries Pinecone vectors and batch-grades relevance to strip ungrounded noise.
3. **Contextual Generation (`generate_from_context`):** Produces a candidate answer following strict markdown formatting rules.
4. **Groundedness Evaluation (`is_sup` & `revise_answer`):** Verifies facts against context. Triggers query or response revision if facts lack backing.
5. **Usefulness Check (`is_use` & `rewrite_question`):** Ensures user intent is satisfied; rewrites search query if answer is not useful.

---

## ⚙️ Environment Configuration

Copy `.env.example` to `.env` and configure your API keys:

```env
GROQ_MODEL=openai/gpt-oss-120b
GEMINI_MODEL=gemini-3.6-flash
EMBEDDING_MODEL=gemini-embedding-001

GROQ_API_KEY=your_groq_api_key_here
GOOGLE_API_KEY=your_google_api_key_here
PINECONE_API_KEY=your_pinecone_api_key_here

PINECONE_INDEX_NAME= your index name
DATA_FILE_PATH= your filepath
```

---

## 🚀 Quickstart Guide

### 1. Installation
```bash
git clone repo link
cd rag_based_assistant_model

python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### 2. Knowledge Base Ingestion
```bash
python ingest.py
```

### 3. Running FastAPI Server
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```
