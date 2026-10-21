"""Offline tests for v3 Markdown ingestion: python -m unittest discover -s tests."""
import tempfile
import unittest
from pathlib import Path

from ingest_v3 import build_chunks, make_report, read_source, _section_attributes

FIXTURE = """# NAVNIT KUMAR YADAV
## Overview
Professional profile.

# A. PROFESSIONAL PROFILE
## A1. Professional positioning
Navnit builds end-to-end AI systems.

# B. PORTFOLIO CAPABILITY AND RECOMMENDATION INDEX
## B2. Capability-to-project mapping
Self-RAG and NeuroScan both demonstrate Python engineering.

# C. VERIFIED PROJECT KNOWLEDGE
## C1. SELF-CORRECTIVE RAG ENGINE
LangGraph and FastAPI. Retrieval, relevance, grounding, usefulness.

## C2. NEUROSCAN AI
MONAI, PyTorch and 3D U-Net. ONNX inference.

# D. CONVERSATIONAL REFERENCE
## D1. Conversation principles
Answer with facts and relevant details.

# E. KNOWLEDGE INTEGRITY
## E1. Evidence and precedence
Distinguish owner-reported metrics from checked documents.
"""


class IngestionTests(unittest.TestCase):
    def test_sections_and_projects(self):
        chunks = build_chunks(FIXTURE, "knowledge_v3.md")
        self.assertTrue(any(c.metadata["project_id"] == "self-corrective-rag" for c in chunks))
        self.assertTrue(any(c.metadata["project_id"] == "neuroscan-ai" for c in chunks))
        self.assertTrue(any(c.metadata["section_type"] == "capability_comparison" for c in chunks))
        self.assertTrue(all(c.metadata["kb_version"] == "3.0.0" for c in chunks))
        self.assertTrue(all(c.metadata["evidence_status"] == "mixed_review_required" for c in chunks))

    def test_deterministic_unique_ids(self):
        one = build_chunks(FIXTURE, "knowledge_v3.md")
        two = build_chunks(FIXTURE, "knowledge_v3.md")
        self.assertEqual([c.id for c in one], [c.id for c in two])
        self.assertEqual(len(one), len({c.id for c in one}))

    def test_large_section_bounded(self):
        sample = "# C. PROJECTS\n## C1. SELF-CORRECTIVE RAG ENGINE\n" + (
            "An evidence-backed engineering note that provides details. " * 90
        )
        chunks = build_chunks(sample, "kb.md", max_chars=600)
        self.assertGreater(len(chunks), 2)
        self.assertTrue(all(len(c.text) <= 600 for c in chunks))
        self.assertTrue(all(c.metadata["project_id"] == "self-corrective-rag" for c in chunks))

    def test_project_mapping_is_explicit_only(self):
        unknown = _section_attributes("C. VERIFIED PROJECT KNOWLEDGE", "C9. UNLISTED")
        self.assertEqual(unknown["project_id"], "")
        self.assertEqual(unknown["section_type"], "project_knowledge")

    def test_reports(self):
        chunks = build_chunks(FIXTURE, "knowledge_v3.md")
        report = make_report(chunks, Path("knowledge_v3.md"), "portfolio-kb-v3-staging")
        self.assertEqual(report["chunks"], len(chunks))
        self.assertTrue(report["all_vector_ids_unique"])
        self.assertIn("neuroscan-ai", report["by_project"])

    def test_input_validation(self):
        with self.assertRaises(ValueError):
            build_chunks("", "kb.md")
        with self.assertRaises(ValueError):
            build_chunks(FIXTURE, "kb.md", max_chars=300)
        with tempfile.TemporaryDirectory() as temp:
            p = Path(temp) / "kb.md"
            p.write_text(FIXTURE, encoding="utf-8")
            self.assertEqual(read_source(p), FIXTURE)

    def test_upload_namespace_guard_before_api_import(self):
        from ingest_v3 import upload_staging
        with self.assertRaises(ValueError):
            upload_staging([], "portfolio-knowledge-base", "portfolio-kb-v2-staging")


if __name__ == "__main__":
    unittest.main()
