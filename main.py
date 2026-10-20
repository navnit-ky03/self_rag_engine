import time
import json
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from schemas import ChatRequest, ChatResponse
from graph import app as rag_app
import traceback

app = FastAPI(
    title="Navnit's AI Assistant RAG API",
    description="Decoupled backend service powering the portfolio RAG chatbot",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

@app.get("/")
def health_check():
    return {"status": "online", "service": "Navnit's AI Assistant RAG Pipeline"}


# Standard non-streaming endpoint
@app.post("/chat", response_model=ChatResponse)
@limiter.limit("10/minute")  
def chat_endpoint(request: Request, chat_request: ChatRequest):
    config = {"configurable": {"thread_id": chat_request.thread_id}}
    t_start = time.perf_counter()

    try:
        result = rag_app.invoke({"question": chat_request.question}, config=config)
        t_elapsed = time.perf_counter() - t_start
        print(
            f"[TIMING] RAG Execution | "
            f"Thread: {chat_request.thread_id} | "
            f"Latency: {t_elapsed:.3f}s ({t_elapsed * 1000:.1f}ms) | "
            f"Query: '{chat_request.question[:40]}...'"
        )
    except Exception as e:
        traceback.print_exc()
        t_elapsed = time.perf_counter() - t_start
        print(f"[ERROR] RAG Pipeline Exception after {t_elapsed:.3f}s: {e}")
        raise HTTPException(
            status_code=503,
            detail="The assistant is temporarily unavailable, please try again in a moment."
        )

    return ChatResponse(answer=result["answer"])


# Real-time SSE Answer Streaming Endpoint
async def stream_rag_tokens(question: str, thread_id: str):
    config = {"configurable": {"thread_id": thread_id}}
    # Only stream tokens from actual answer generation nodes (ignore grading/eval LLM calls)
    answer_nodes = {"direct_ans", "context_ans", "revise_ans"}

    try:
        async for event in rag_app.astream_events({"question": question}, config=config, version="v2"):
            current_node = event.get("langgraph_node", "")
            kind = event.get("event", "")

            if current_node in answer_nodes and kind == "on_chat_model_stream":
                chunk = event.get("data", {}).get("chunk")
                if chunk and hasattr(chunk, "content") and chunk.content:
                    yield f"data: {json.dumps({'token': chunk.content})}\n\n"

    except Exception as e:
        print(f"[ERROR] Streaming Exception: {e}")
        yield f"data: {json.dumps({'token': ' An error occurred while streaming the response.'})}\n\n"

    yield "data: [DONE]\n\n"


@app.post("/chat/stream")
@limiter.limit("10/minute")
async def chat_stream_endpoint(request: Request, chat_request: ChatRequest):
    return StreamingResponse(
        stream_rag_tokens(chat_request.question, chat_request.thread_id),
        media_type="text/event-stream"
    )