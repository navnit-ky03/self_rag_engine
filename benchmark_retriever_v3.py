"""Live read-only comparison: python benchmark_retriever_v3.py.

Uses the v3 staging namespace, reuses one Pinecone client and embedding model
across tests, and prints evidence metadata without generating answers.
"""
from __future__ import annotations

from collections import Counter
import json
import time

from retriever_v3 import _cached_store, _default_index_name, V3_NAMESPACE, retrieve

QUESTIONS = (
    "What technologies does NeuroScan AI use?",
    "Explain the architecture of the Self-RAG Engine.",
    "Which projects best demonstrate Python engineering?",
    "Which projects are suitable for AI workflow automation?",
    "Compare Self-RAG Engine and NeuroScan AI.",
    "Which projects demonstrate end-to-end machine learning deployment?",
)


def concise_docs(docs: list) -> list[dict]:
    return [
        {
            "project": d.metadata.get("project_id") or "(non-project)",
            "section": d.metadata.get("section_id", ""),
            "type": d.metadata.get("section_type", ""),
            "preview": " ".join(d.page_content.split())[:170],
        }
        for d in docs
    ]


def main() -> None:
    start = time.perf_counter()
    store = _cached_store(_default_index_name(), V3_NAMESPACE)
    print("client_setup_ms:", round((time.perf_counter() - start) * 1000, 2))
    for question in QUESTIONS:
        print("\nQUESTION:", question)
        docs, diagnostics = retrieve(question, store=store)
        print("default:", json.dumps(diagnostics, ensure_ascii=False))
        print("documents:", json.dumps(concise_docs(docs), ensure_ascii=False))
        if diagnostics["profile"] != "comparative":
            continue
        # A/B check: unreranked similarity baseline (k=8) against the
        # default single-call candidate search plus local evidence selection.
        # No extra request is made in the production retrieval path.
        baseline = store.as_retriever(search_type="similarity", search_kwargs={"k": 8})
        t0 = time.perf_counter()
        baseline_docs = list(baseline.invoke(question))
        print("similarity_baseline:", json.dumps({
            "search_ms": round((time.perf_counter() - t0) * 1000, 2),
            "project_counts": dict(Counter(
                d.metadata.get("project_id") or "(non-project)"
                for d in baseline_docs
            )),
            "documents": concise_docs(baseline_docs),
        }, ensure_ascii=False))


if __name__ == "__main__":
    main()
