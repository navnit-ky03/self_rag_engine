# Self-RAG v3.0.0 — Final Release Checklist

**Status:** staging candidate only. NOT authorized to merge or deploy.

## Owner-verified evidence
- Pinecone v3 staging namespace: 50 vectors; v2 namespace retained 88.
- Previous mocked integration suite: 9/9 passed.
- Four local chat questions succeeded twice with complete UTF-8 responses. Follow-up context, recommendations, and unsupported employment were appropriate.
- Warm end-to-end latencies: architecture 6.822 s; follow-up 3.145 s; Python recommendation 4.461 s; employment 2.695 s. Average ~4.28 s. First run incurred cold-start costs. No measured controlled v2-to-v3 benchmark.

## New release safeguards (must re-test)
- Prompts disallow unsupported clinical readiness or diagnostic use assertions about NeuroScan.
- Final answer medical safety gate replaces detected affirmative readiness claims with a research-only explanation, even after a positive grounding judgment. Narrow safeguard; not proof of clinical compliance.
- New offline regression coverage: separate conversation threads, named-project switches, clinical claims, generation provider failures.
- New offline API contract checks: GET /, POST /chat JSON contract, invalid request 422, upstream service error 503, and v3 SSE final-only behavior.

## Local verification
Run these in backend PowerShell after pulling:

    git pull --ff-only origin feature/self-rag-v3.0.0
    py -m unittest discover -s tests -p "test_v3_integration.py" -v
    py -m unittest discover -s tests -p "test_v3_api_contract.py" -v
    py -m unittest discover -s tests -p "test_retriever_v3.py" -v
    py -m unittest discover -s tests -p "test_ingest_v3.py" -v

Start opt-in v3 locally:

    $env:SELF_RAG_VERSION = "3"
    py -m uvicorn main:app --host 127.0.0.1 --port 8000

In a second terminal, repeat representative /chat requests with different thread IDs; inspect the answer and server timing/error logs.

## Hosting / release gates — review only, do NOT deploy
1. Privately record current v2 commit, Render deployment/settings, and rollback. Never paste API keys.
2. Version flag must be set to 3 only after approval; default 2 preserves v2. Check hosting secret variable names (not their values): GROQ_API_KEY, GROQ_EVAL_API_KEY (optional), GOOGLE_API_KEY, PINECONE_API_KEY, EMBEDDING_MODEL and PINECONE_INDEX_NAME.
3. Confirm the hosting runtime can query correct Pinecone index and v3 namespace; no new ingestion needed.
4. Confirm embeddings dimensions match the existing index; verify cold start and warm latency in staging.
5. Assess wildcard CORS and 10/minute IP limit for a public site. Rate limiting is not authorization.
6. MemorySaver is process-local: not durable across restarts, multiple workers, or instance moves. Decide whether acceptable before production rollout.
7. v3 /chat/stream provides one verified final event; it does not stream unverified draft tokens. Main frontend currently uses non-streaming /chat.
8. Reconfirm frontend sends {question, thread_id} and reads {answer}; frontend v2.2 is frozen.
9. Check staging 503/error behavior and rate limits without exposing internal exceptions.
10. Inspect feature-branch PR. Do not merge, deploy, or alter hosting variables without separate explicit user approval. Keep v2 available for rollback.

## Release decision
- New regression tests: pending
- End-to-end safety recheck: pending
- Staging hosting behavior: pending
- Merge approval: NO
- Deployment approval: NO

## Known limits
Heuristic retrieval may omit project evidence, LLM grounding evaluation is fallible, and the medical language guard is deliberately narrow. No further standalone retriever optimization is planned.
