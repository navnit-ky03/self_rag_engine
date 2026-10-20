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


    def test_two_threads_do_not_share_conversation(self):
        g = self.graph_module
        from langchain_core.documents import Document
        docs = [Document(
            page_content="NeuroScan is a PyTorch segmentation project.",
            metadata={"section_id": "C2", "project_id": "neuroscan-ai"},
        )]
        with patch.object(g, "retrieve", return_value=(docs, {"profile": "focused"})), \
             patch.object(g.llm, "invoke", return_value=SimpleNamespace(
                 content="NeuroScan uses PyTorch."
             )), \
             patch.object(g.support_llm, "invoke", return_value=SimpleNamespace(
                 issup="fully_supported", evidence=[]
             )):
            a = g.app.invoke(
                {"question": "Describe NeuroScan."},
                config=self._config("v3-release-thread-a"),
            )
            b = g.app.invoke(
                {"question": "What does NeuroScan use?"},
                config=self._config("v3-release-thread-b"),
            )
        self.assertEqual(len(a["chat_history"]), 2)
        self.assertEqual(len(b["chat_history"]), 2)
        self.assertNotEqual(
            a["chat_history"][0]["content"],
            b["chat_history"][0]["content"],
        )

    def test_explicit_project_switch_skips_rewrite(self):
        g = self.graph_module
        from langchain_core.documents import Document
        docs = [Document(
            page_content="NeuroScan uses PyTorch; voice booking uses n8n.",
            metadata={"section_id": "B2", "project_id": ""},
        )]
        with patch.object(g, "retrieve", return_value=(docs, {"profile": "focused"})) as retrieval, \
             patch.object(g.llm, "invoke", return_value=SimpleNamespace(
                 content="A project technology summary."
             )), \
             patch.object(g.support_llm, "invoke", return_value=SimpleNamespace(
                 issup="fully_supported", evidence=[]
             )), \
             patch.object(g.rewrite_llm, "invoke") as rewrite:
            g.app.invoke(
                {"question": "Tell me about NeuroScan AI."},
                config=self._config("v3-release-switch"),
            )
            out = g.app.invoke(
                {"question": "What technologies does the Voice Booking Assistant use?"},
                config=self._config("v3-release-switch"),
            )
        rewrite.assert_not_called()
        self.assertIn("Voice Booking", retrieval.call_args.args[0])
        self.assertEqual(len(out["chat_history"]), 4)

    def test_medical_guard_even_when_judge_approves(self):
        g = self.graph_module
        from langchain_core.documents import Document
        docs = [Document(
            page_content="NeuroScan is a research 3D MRI segmentation project built using PyTorch and MONAI.",
            metadata={"section_id": "C2", "project_id": "neuroscan-ai"},
        )]
        with patch.object(g, "retrieve", return_value=(docs, {"profile": "focused"})), \
             patch.object(g.llm, "invoke", return_value=SimpleNamespace(
                 content="NeuroScan is clinically useful and suitable for patient care."
             )), \
             patch.object(g.support_llm, "invoke", return_value=SimpleNamespace(
                 issup="fully_supported", evidence=[]
             )):
            out = g.app.invoke(
                {"question": "Is NeuroScan clinically useful?"},
                config=self._config("v3-release-clinical"),
            )
        self.assertTrue(out["medical_safety_intervened"])
        self.assertNotIn("clinically useful", out["answer"])
        self.assertIn("research-oriented", out["answer"])
        self.assertEqual(out["chat_history"][-1]["content"], out["answer"])

    def test_table_answer_is_converted_before_history(self):
        g = self.graph_module
        from langchain_core.documents import Document
        docs = [Document(
            page_content="Self-RAG uses FastAPI. NeuroScan uses MONAI.",
            metadata={"section_id": "B2", "project_id": ""},
        )]
        markdown = (
            "| Project | Technology |\\n"
            "|---|---|\\n"
            "| Self-RAG | FastAPI |\\n"
            "| NeuroScan | MONAI |"
        )
        with patch.object(g, "retrieve", return_value=(docs, {
            "profile": "comparative", "result_count": 1,
        })), patch.object(g.llm, "invoke", return_value=SimpleNamespace(
            content=markdown
        )), patch.object(g.support_llm, "invoke", return_value=SimpleNamespace(
            issup="fully_supported", evidence=[]
        )):
            out = g.app.invoke(
                {"question": "Compare Self-RAG and NeuroScan."},
                config=self._config("v3-no-table"),
            )
        self.assertIn("FastAPI", out["answer"])
        self.assertIn("MONAI", out["answer"])
        self.assertNotIn("|---|", out["answer"])
        self.assertEqual(out["chat_history"][-1]["content"], out["answer"])

    def test_generation_service_failure_propagates(self):
        g = self.graph_module
        from langchain_core.documents import Document
        docs = [Document(
            page_content="NeuroScan uses PyTorch.",
            metadata={"section_id": "C2", "project_id": "neuroscan-ai"},
        )]
        with patch.object(g, "retrieve", return_value=(docs, {"profile": "focused"})), \
             patch.object(g.llm, "invoke", side_effect=RuntimeError("Groq unavailable")):
            with self.assertRaisesRegex(RuntimeError, "Groq unavailable"):
                g.app.invoke(
                    {"question": "What does NeuroScan use?"},
                    config=self._config("v3-release-provider-failure"),
                )


class ClinicalSafetyTests(unittest.TestCase):
    def test_rejects_clinical_approval_language(self):
        from medical_safety_v3 import ensure_safe_medical_claims
        answer, intervened = ensure_safe_medical_claims(
            "NeuroScan is FDA approved and clinically validated."
        )
        self.assertTrue(intervened)
        self.assertIn("does not establish clinical", answer)

    def test_allows_research_description_and_explicit_disclaimer(self):
        from medical_safety_v3 import ensure_safe_medical_claims
        for answer in (
            "NeuroScan uses PyTorch and MONAI to segment 3D MRI data.",
            "NeuroScan is not clinically validated for patient care.",
        ):
            with self.subTest(answer=answer):
                output, intervened = ensure_safe_medical_claims(answer)
                self.assertFalse(intervened)
                self.assertEqual(output, answer)

    def test_only_applies_to_neuroscan(self):
        from medical_safety_v3 import ensure_safe_medical_claims
        answer = "The phrase clinically validated appears in a dictionary example."
        self.assertEqual(ensure_safe_medical_claims(answer), (answer, False))


if __name__ == "__main__":
    unittest.main()
