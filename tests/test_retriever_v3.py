"""Offline, API-free checks: py -m unittest discover -s tests -p 'test_retriever_v3.py' -v."""
import unittest
from unittest.mock import patch

from retriever_v3 import (
    V3_NAMESPACE, choose_profile, explicit_project_ids, make_search_kwargs,
    build_retriever, retrieve,
)


class FakeDoc:
    def __init__(self, project_id, section_type="project_knowledge"):
        self.metadata = {"project_id": project_id, "section_type": section_type}
        self.page_content = f"**Implementation.** Evidence about {project_id}. Python FastAPI MONAI n8n."


class FakeRetriever:
    def __init__(self, docs):
        self.docs = docs
        self.query = None

    def invoke(self, query):
        self.query = query
        return self.docs


class FakeStore:
    def __init__(self, docs=()):
        self.docs = list(docs)
        self.calls = []

    def as_retriever(self, *, search_type, search_kwargs):
        self.calls.append((search_type, search_kwargs))
        return FakeRetriever(self.docs)


class RetrievalTests(unittest.TestCase):
    def test_profiles(self):
        self.assertEqual(choose_profile("What framework does NeuroScan use?"), "focused")
        self.assertEqual(choose_profile("How does NeuroScan architecture work?"), "detailed")
        self.assertEqual(choose_profile("Compare NeuroScan with Self-RAG"), "comparative")
        self.assertEqual(choose_profile("Which projects are strongest for Python?"), "comparative")
        self.assertEqual(choose_profile("Tell me about Navnit's top Python projects"), "comparative")
        self.assertEqual(choose_profile("Best projects for AI automation"), "comparative")
        self.assertEqual(choose_profile("What is Navnit's professional profile?"), "focused")

    def test_entity_matching(self):
        self.assertEqual(explicit_project_ids("NeuroScan AI"), ["neuroscan-ai"])
        self.assertEqual(
            set(explicit_project_ids("Self-RAG vs NeuroScan")),
            {"self-corrective-rag", "neuroscan-ai"},
        )
        self.assertEqual(explicit_project_ids("Tell me about his work"), [])

    def test_project_filter_focused(self):
        kind, opts = make_search_kwargs("What technologies does NeuroScan use?")
        self.assertEqual(kind, "similarity")
        self.assertEqual(opts["k"], 4)
        self.assertEqual(opts["filter"], {"project_id": {"$eq": "neuroscan-ai"}})

    def test_detailed(self):
        kind, opts = make_search_kwargs("Explain NeuroScan architecture")
        self.assertEqual(kind, "similarity")
        self.assertEqual(opts["k"], 6)
        self.assertEqual(opts["filter"]["project_id"]["$eq"], "neuroscan-ai")

    def test_comparative_is_unfiltered(self):
        kind, opts = make_search_kwargs("Compare Self-RAG and NeuroScan")
        self.assertEqual(kind, "similarity")
        self.assertEqual(opts["k"], 18)
        self.assertNotIn("fetch_k", opts)
        self.assertNotIn("filter", opts)

    def test_no_auto_filter_on_general(self):
        _, opts = make_search_kwargs("Tell me about Navnit's skills")
        self.assertNotIn("filter", opts)

    def test_explicit_project_override(self):
        _, opts = make_search_kwargs("How is it implemented?", profile="detailed", project_id="voice-booking")
        self.assertEqual(opts["filter"], {"project_id": {"$eq": "voice-booking"}})

    def test_invalid_settings(self):
        with self.assertRaises(ValueError):
            choose_profile("")
        with self.assertRaises(ValueError):
            make_search_kwargs("Question?", profile="other")
        with self.assertRaises(ValueError):
            make_search_kwargs("Question?", project_id="fake")
        with self.assertRaises(ValueError):
            build_retriever("Question?", namespace="portfolio-kb-v2-staging", store=FakeStore())

    def test_store_caching(self):
        from retriever_v3 import _cached_store
        _cached_store.cache_clear()
        with patch("langchain_google_genai.GoogleGenerativeAIEmbeddings") as embeddings, \
             patch("langchain_pinecone.PineconeVectorStore.from_existing_index") as connect:
            first = _cached_store("test-index", V3_NAMESPACE)
            second = _cached_store("test-index", V3_NAMESPACE)
        self.assertIs(first, second)
        embeddings.assert_called_once()
        connect.assert_called_once()
        _cached_store.cache_clear()

    def test_separate_setup_and_search_timing(self):
        store = FakeStore([FakeDoc("neuroscan-ai")])
        docs, info = retrieve("What does NeuroScan use?", store=store)
        self.assertEqual(len(docs), 1)
        self.assertIn("setup_ms", info)
        self.assertIn("search_ms", info)
        self.assertLessEqual(info["setup_ms"], info["elapsed_ms"])
        self.assertLessEqual(info["search_ms"], info["elapsed_ms"])

    def test_comparative_selection_prefers_project_evidence(self):
        from evidence_v3 import choose_evidence
        candidates = [
            FakeDoc("", "conversation_reference"),
            FakeDoc("", "integrity_and_operations"),
            FakeDoc("delivery-analytics"),
            FakeDoc("fraud-detection"),
            FakeDoc("self-corrective-rag"),
            FakeDoc("neuroscan-ai"),
            FakeDoc("", "capability_comparison"),
        ]
        result = choose_evidence(
            candidates, "Compare Self-RAG and NeuroScan", limit=4,
            required_projects=("self-corrective-rag", "neuroscan-ai"),
        )
        ids = [doc.metadata["project_id"] for doc in result]
        self.assertEqual(len(result), 4)
        self.assertIn("self-corrective-rag", ids)
        self.assertIn("neuroscan-ai", ids)
        self.assertNotIn("integrity_and_operations",
                         [doc.metadata["section_type"] for doc in result])

    def test_single_search_and_evidence_cap(self):
        store = FakeStore([FakeDoc("neuroscan-ai") for _ in range(20)])
        docs, info = retrieve("What projects compare with NeuroScan?", store=store)
        self.assertEqual(len(store.calls), 1)
        self.assertLessEqual(len(docs), 8)
        self.assertEqual(info["candidate_count"], 20)

    def test_mocked_retrieval_and_diagnostics(self):
        store = FakeStore([
            FakeDoc("self-corrective-rag"),
            FakeDoc("neuroscan-ai"),
            FakeDoc("", "capability_comparison"),
        ])
        docs, report = retrieve("Compare Self-RAG and NeuroScan", store=store)
        self.assertEqual(len(docs), 3)
        self.assertEqual(report["namespace"], V3_NAMESPACE)
        self.assertEqual(report["profile"], "comparative")
        self.assertEqual(report["distinct_projects"], ["neuroscan-ai", "self-corrective-rag"])
        self.assertIn("capability_comparison", report["section_types"])
        self.assertGreaterEqual(report["elapsed_ms"], 0)
        self.assertEqual(store.calls[0][0], "similarity")
        self.assertEqual(store.calls[0][1]["k"], 18)


if __name__ == "__main__":
    unittest.main()
