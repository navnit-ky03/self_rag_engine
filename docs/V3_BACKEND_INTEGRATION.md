# Self-RAG v3.0.0 — backend integration

**This is a feature-flagged staging candidate.** Existing v2 graph and retriever, main branch, and Pinecone v2 namespace are untouched.

## Runtime selection
- Default SELF_RAG_VERSION=2: main.py imports existing graph.py.
- Opt in locally with SELF_RAG_VERSION=3: main.py imports graph_v3.py, which uses retriever_v3.py and the v3 staging namespace.
- Existing GET /, POST /chat request {question, thread_id} and answer response remain unchanged.
- Existing POST /chat/stream in v3 emits only the **verified final answer**, as a single SSE token event rather than provisional model tokens.
- LangGraph's MemorySaver is process-local; history does not survive server restarts or multi-worker instance switches.

## V3 behavior
1. Conservative routing allows direct greetings / generic definitions; questions about Navnit go to retrieval.
2. Pronoun follow-ups with existing chat history use one context-rewrite LLM call.
3. One vector search with v3 frozen retrieval profiles and local evidence selection.
4. Answer from retrieved evidence using natural, professionally grounded language.
5. One grounding evaluation; if unsupported, make at most one revision and re-check. If still unsupported, return a safe fallback.
6. Update chat history only once after the final answer.
7. Avoid the older batch relevance and usefulness evaluation LLM calls to reduce latency while retaining the grounding check.

## PowerShell: local validation

~~~powershell
git pull --ff-only origin feature/self-rag-v3.0.0
py -m unittest discover -s tests -p "test_v3_integration.py" -v
py -m unittest discover -s tests -p "test_retriever_v3.py" -v
$env:SELF_RAG_VERSION = "3"
py -m uvicorn main:app --host 127.0.0.1 --port 8000
~~~

Open a **second PowerShell** window:

~~~powershell
$body = @{question = "What framework does NeuroScan use?"; thread_id = "v3-smoke-1"} | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/chat -ContentType "application/json" -Body $body
$body = @{question = "How was it used in the pipeline?"; thread_id = "v3-smoke-1"} | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/chat -ContentType "application/json" -Body $body
~~~

Use a different thread ID to check session isolation. Verify greeting, focused project question, detailed architecture, cross-project comparison, follow-up, project switching, and unsupported biographical claims. Compare cold and warm end-to-end timings, not just retrieval speed. Never share .env contents or API keys.

## Deployment gate

This change has **not** been independently end-to-end verified. Complete local mock tests and real local /chat checks before approval. Do not merge draft PR, change production env, or deploy without owner authorization. For v3 rollout set the feature flag explicitly; rollback is achieved by restoring the original version flag and deploying the unchanged v2 graph. The process-local memory and heuristic evidence selector remain known limitations.
