"""Offline v3 routing and graph-contract tests; no Groq/Gemini/Pinecone calls."""
import importlib
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

from routing_v3 import requires_portfolio_retrieval, needs_context_rewrite


class RoutingTests(unittest.TestCase):
    def test_greeting_is_direct(self):
        self.assertFalse(requires_portfolio_retrieval("Hi"))
        self.assertFalse(requires_portfolio_retrieval("Thank you!"))

    def test_general_definition_is_direct(self):
        self.assertFalse(requires_portfolio_retrieval("What is RAG?"))

    def test_all_portfolio_questions_retrieve(self):
        for question in (
            "What projects has Navnit built?",
            "Does Navnit have professional employment experience?",
            "What technologies does NeuroScan use?",
            "What is RAG used for in Navnit's work?",
        ):
            with self.subTest(question=question):
                self.assertTrue(requires_portfolio_retrieval(question))

    def test_followups_rewrite_only_if_ambiguous(self):
        history = [
            {"role": "user", "content": "Tell me about NeuroScan"},
            {"role": "assistant", "content": "It uses MONAI."},
        ]
        self.assertTrue(needs_context_rewrite("How was it deployed?", history))
        self.assertFalse(needs_context_rewrite("How was NeuroScan deployed?", history))
        self.assertFalse(needs_context_rewrite("How was it deployed?", []))

    def test_empty_question_rejected(self):
        with self.assertRaises(ValueError):
            requires_portfolio_retrieval("   ")


class GraphContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Import graph without creating real ChatGroq clients or contacting
        # external systems; actual graph nodes are invoked using mocks.
        def fake_chat(**_):
            client = MagicMock()
            client.with_structured_output.side_effect = lambda *_a, **_k: MagicMock()
            return client

        with patch("langchain_groq.ChatGroq", side_effect=fake_chat):
            cls.graph_module = importlib.import_module("graph_v3")

    def _config(self, thread_id):
        return {"configurable": {"thread_id": thread_id}}

    def test_no_context_is_safe_and_recorded(self):
        g = self.graph_module
        with patch.object(g, "retrieve", return_value=([], {
            "profile": "focused", "result_count": 0,
        })):
            out = g.app.invoke(
                {"question": "What company hired Navnit in 2021?"},
                config=self._config("v3-no-evidence"),
            )
        self.assertIn("couldn't verify", out["answer"])
        self.assertEqual(len(out["chat_history"]), 2)

    def test_grounded_answer_is_recorded(self):
        g = self.graph_module
        from langchain_core.documents import Document
        docs = [Document(
            page_content="NeuroScan uses PyTorch and MONAI for 3D segmentation.",
            metadata={"source": "kb_v3.md", "section_id": "C2", "project_id": "neuroscan-ai"},
        )]
        with patch.object(g, "retrieve", return_value=(docs, {
            "profile": "focused", "result_count": 1,
        })), patch.object(g.llm, "invoke", return_value=SimpleNamespace(
            content="NeuroScan uses PyTorch and MONAI."
        )), patch.object(g.support_llm, "invoke", return_value=SimpleNamespace(
            issup="fully_supported", evidence=["PyTorch and MONAI"]
        )):
            out = g.app.invoke(
                {"question": "What technology does NeuroScan use?"},
                config=self._config("v3-grounded"),
            )
        self.assertEqual(out["answer"], "NeuroScan uses PyTorch and MONAI.")
        self.assertEqual(out["grounding_status"], "fully_supported")
        self.assertEqual(len(out["chat_history"]), 2)

    def test_followup_uses_context_and_stores_history_once(self):
        g = self.graph_module
        from langchain_core.documents import Document
        docs = [Document(
            page_content="NeuroScan exports ONNX for deployment.",
            metadata={"source": "kb_v3.md", "section_id": "C2", "project_id": "neuroscan-ai"},
        )]
        with patch.object(g, "retrieve", return_value=(docs, {
            "profile": "focused", "result_count": 1,
        })) as retrieval, patch.object(g.llm, "invoke", return_value=SimpleNamespace(
            content="NeuroScan uses ONNX for inference."
        )), patch.object(g.support_llm, "invoke", return_value=SimpleNamespace(
            issup="fully_supported", evidence=[]
        )), patch.object(g.rewrite_llm, "invoke", return_value=SimpleNamespace(
            retrieval_query="NeuroScan ONNX inference deployment"
        )) as rewrite:
            g.app.invoke(
                {"question": "What technology does NeuroScan use?"},
                config=self._config("v3-followup"),
            )
            out = g.app.invoke(
                {"question": "How is it deployed?"},
                config=self._config("v3-followup"),
            )
        rewrite.assert_called_once()
        self.assertEqual(retrieval.call_args.args[0], "NeuroScan ONNX inference deployment")
        self.assertEqual(len(out["chat_history"]), 4)

    def test_rejected_grounding_never_leaks_draft(self):
        g = self.graph_module
        from langchain_core.documents import Document
        docs = [Document(
            page_content="The knowledge base does not establish prior employers.",
            metadata={"source": "kb_v3.md", "section_id": "E1", "project_id": ""},
        )]
        with patch.object(g, "retrieve", return_value=(docs, {
            "profile": "focused", "result_count": 1,
        })), patch.object(g.llm, "invoke", return_value=SimpleNamespace(
            content="Navnit was employed by an imaginary company."
        )), patch.object(g.support_llm, "invoke", return_value=SimpleNamespace(
            issup="no_support", evidence=[]
        )):
            out = g.app.invoke(
                {"question": "Who employed Navnit?"},
                config=self._config("v3-hallucination"),
            )
        self.assertNotIn("imaginary", out["answer"])
        self.assertIn("couldn't verify", out["answer"])


if __name__ == "__main__":
    unittest.main()
