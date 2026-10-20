"""Regression tests for the chat UI's deliberate lack of Markdown table support."""
import unittest
from response_format_v3 import no_markdown_tables


class ResponseFormattingTests(unittest.TestCase):
    def test_converts_project_comparison_and_preserves_each_fact(self):
        source = (
            "**Advanced Python projects**\n"
            "| Project | Evidence |\n"
            "|---|---|\n"
            "| **Self-RAG** | FastAPI and LangGraph<br>Grounded answers |\n"
            "| **NeuroScan** | PyTorch and MONAI |\n"
            "\n**Why:** Different engineering strengths."
        )
        result = no_markdown_tables(source)
        self.assertIn("**Self-RAG**", result)
        self.assertIn("- FastAPI and LangGraph", result)
        self.assertIn("- Grounded answers", result)
        self.assertIn("**NeuroScan**", result)
        self.assertIn("PyTorch and MONAI", result)
        self.assertIn("**Why:**", result)
        self.assertNotIn("|---|", result)
        self.assertNotIn("<br>", result)
        self.assertNotIn("| Project |", result)

    def test_converts_three_column_table(self):
        result = no_markdown_tables(
            "| Project | Stack | Evidence |\n"
            "|:---|:---:|---:|\n"
            "| NeuroScan | MONAI | 3D segmentation |"
        )
        self.assertIn("**NeuroScan**", result)
        self.assertIn("**Stack:** MONAI", result)
        self.assertIn("**Evidence:** 3D segmentation", result)

    def test_preserves_normal_prose_pipes(self):
        text = "Use x | y in a literal formula.\nThe API returns {\"answer\":\"ok\"}."
        self.assertEqual(no_markdown_tables(text), text)

    def test_preserves_header_without_data_rows(self):
        source = "| Column A | Column B |\n|---|---|"
        self.assertEqual(no_markdown_tables(source), source)

    def test_escaped_pipes_and_breaks(self):
        source = (
            r"\| Project \| Stack \|" + "\n"
            + r"\|---\|---\|" + "\n"
            + r"\| NeuroScan \| MONAI\<br>PyTorch \|"
        )
        result = no_markdown_tables(source)
        self.assertIn("MONAI", result)
        self.assertIn("- PyTorch", result)
        self.assertNotIn("<br>", result)

    def test_empty_and_plain_answer_unchanged(self):
        self.assertEqual(no_markdown_tables(""), "")
        self.assertEqual(no_markdown_tables("NeuroScan uses MONAI."), "NeuroScan uses MONAI.")


if __name__ == "__main__":
    unittest.main()
