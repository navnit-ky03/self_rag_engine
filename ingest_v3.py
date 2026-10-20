"""Safe, section-aware Markdown ingestion for Self-RAG v3.0.0.

Dry-run requires only Python's standard library. Production upload is intentionally
append-only to a dedicated v3 namespace; this module NEVER deletes vectors.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any

KB_VERSION = "3.0.0"
DEFAULT_NAMESPACE = "portfolio-kb-v3-staging"
HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
SECTION_HEADING = re.compile(r"^([A-E])\d+\.\s+")
PROJECT_HEADINGS = {
    "C1": "self-corrective-rag",
    "C2": "neuroscan-ai",
    "C3": "voice-booking",
    "C4": "lecture-rag",
    "C5": "customer-value",
    "C6": "fraud-detection",
    "C7": "delivery-analytics",
    "C8": "portfolio-frontend",
}
MAIN_CATEGORIES = {
    "A": "professional_profile",
    "B": "capability_comparison",
    "C": "project_knowledge",
    "D": "conversation_reference",
    "E": "integrity_and_operations",
}


@dataclass(frozen=True)
class Chunk:
    id: str
    text: str
    metadata: dict[str, Any]


def _headings_and_sections(markdown: str) -> list[dict[str, str]]:
    """Group text into H2 sections, retaining their H1 parent and H2 title."""
    parent = ""
    section_title = "Introduction"
    lines: list[str] = []
    result: list[dict[str, str]] = []

    def flush() -> None:
        body = "\n".join(lines).strip()
        if body:
            result.append({"parent": parent, "title": section_title, "body": body})

    for line in markdown.splitlines():
        heading = HEADING.match(line)
        if heading and len(heading.group(1)) <= 2:
            flush()
            lines = []
            if len(heading.group(1)) == 1:
                parent = heading.group(2).strip()
                section_title = "Overview"
            else:
                section_title = heading.group(2).strip()
        else:
            lines.append(line)
    flush()
    return result


def _section_attributes(parent: str, title: str) -> dict[str, str]:
    section_match = re.match(r"^([A-E])(\d+)\.", title)
    category = section_match.group(1) if section_match else ""
    subsection = (category + section_match.group(2)) if section_match else ""
    # Metadata MUST reflect explicit section labels, not guessed LLM labels.
    project_id = PROJECT_HEADINGS.get(subsection, "")
    if not category:
        category = next(
            (key for key in MAIN_CATEGORIES if parent.startswith(key + ".")),
            "",
        )
    return {
        "section_id": subsection or category or "intro",
        "section_type": MAIN_CATEGORIES.get(category, "introduction"),
        "project_id": project_id,
        # The v3 source mixes README-backed statements, owner attestations and
        # caveats. Never claim a whole chunk is independently verified.
        "evidence_status": "mixed_review_required",
    }


def _split_body(body: str, limit: int) -> list[str]:
    """Split paragraphs first, then oversized paragraphs at safe line/word breaks.

    Text is not silently dropped. Overlap is provided by the previous paragraph
    only if it fits, reducing mid-sentence slicing and retrieval duplication.
    """
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", body) if p.strip()]
    pieces: list[str] = []
    for paragraph in paragraphs:
        while len(paragraph) > limit:
            cut = paragraph.rfind("\n", 0, limit + 1)
            if cut < limit // 2:
                cut = paragraph.rfind(" ", 0, limit + 1)
            if cut < limit // 2:
                cut = limit
            pieces.append(paragraph[:cut].strip())
            paragraph = paragraph[cut:].strip()
        if paragraph:
            pieces.append(paragraph)

    groups: list[str] = []
    current = ""
    for piece in pieces:
        candidate = current + ("\n\n" if current else "") + piece
        if current and len(candidate) > limit:
            groups.append(current)
            current = piece
        else:
            current = candidate
    if current:
        groups.append(current)
    return groups


def build_chunks(
    markdown: str,
    source_name: str,
    *,
    max_chars: int = 1200,
) -> list[Chunk]:
    """Pure deterministic transformation; no API calls or file-system writes."""
    if max_chars < 400:
        raise ValueError("max_chars must be at least 400")
    if not markdown.strip():
        raise ValueError("Knowledge base is empty")
    sections = _headings_and_sections(markdown)
    if not sections:
        raise ValueError("No nonempty sections found")

    chunks: list[Chunk] = []
    for section in sections:
        attrs = _section_attributes(section["parent"], section["title"])
        prefix = f"{section['parent']}\n{section['title']}\n\n".strip() + "\n\n"
        budget = max_chars - len(prefix)
        if budget < 200:
            raise ValueError("Section headings are too long for chunk budget")
        for position, text in enumerate(_split_body(section["body"], budget)):
            full = prefix + text
            metadata: dict[str, Any] = {
                "kb_version": KB_VERSION,
                "source": source_name,
                "section_heading": section["title"],
                "parent_heading": section["parent"],
                "section_id": attrs["section_id"],
                "section_type": attrs["section_type"],
                "project_id": attrs["project_id"],
                "evidence_status": attrs["evidence_status"],
                "chunk_index": position,
            }
            identity = json.dumps(
                {"text": full, "metadata": metadata},
                ensure_ascii=False, sort_keys=True,
            )
            chunks.append(Chunk(
                id=hashlib.sha256(identity.encode("utf-8")).hexdigest(),
                text=full,
                metadata=metadata,
            ))

    if len({chunk.id for chunk in chunks}) != len(chunks):
        raise ValueError("Duplicate vector IDs detected")
    return chunks


def read_source(source: Path) -> str:
    if not source.is_file():
        raise FileNotFoundError(f"Knowledge base not found: {source}")
    if source.suffix.lower() in {".md", ".markdown"}:
        return source.read_text(encoding="utf-8-sig")
    if source.suffix.lower() == ".pdf":
        # PDF compatibility is best-effort; Markdown provides heading structure.
        from langchain_community.document_loaders import PyPDFLoader
        pages = PyPDFLoader(str(source)).load()
        if not pages:
            raise ValueError("No PDF pages extracted")
        # Page text does not preserve Markdown headings: for PDF, use a single
        # nonproject section rather than fabricating section/project metadata.
        return "# PDF IMPORT (UNSTRUCTURED)\n## Extracted text\n\n" + "\n\n".join(
            page.page_content for page in pages
        )
    raise ValueError("Supported input formats: .md, .markdown, .pdf")


def make_report(chunks: list[Chunk], source: Path, namespace: str) -> dict[str, Any]:
    return {
        "kb_version": KB_VERSION,
        "source": str(source),
        "namespace": namespace,
        "chunks": len(chunks),
        "by_section_type": dict(sorted(Counter(
            c.metadata["section_type"] for c in chunks
        ).items())),
        "by_project": dict(sorted(Counter(
            c.metadata["project_id"] or "(non-project)" for c in chunks
        ).items())),
        "by_section": dict(sorted(Counter(
            c.metadata["section_id"] for c in chunks
        ).items())),
        "max_chunk_chars": max((len(c.text) for c in chunks), default=0),
        "all_vector_ids_unique": len({c.id for c in chunks}) == len(chunks),
    }


def upload_staging(chunks: list[Chunk], index_name: str, namespace: str) -> None:
    if namespace != DEFAULT_NAMESPACE:
        raise ValueError(
            f"Safety guard: only the staging namespace {DEFAULT_NAMESPACE!r} "
            "is permitted by this v3 ingestion command."
        )
    # Late imports ensure dry-run never requires embedding/Pinecone packages.
    from langchain_google_genai import GoogleGenerativeAIEmbeddings
    from langchain_pinecone import PineconeVectorStore
    from config import EMBEDDING_MODEL

    store = PineconeVectorStore(
        index_name=index_name,
        embedding=GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL),
        namespace=namespace,
    )
    store.add_texts(
        texts=[c.text for c in chunks],
        metadatas=[c.metadata for c in chunks],
        ids=[c.id for c in chunks],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--max-chars", type=int, default=1200)
    parser.add_argument("--namespace", default=DEFAULT_NAMESPACE)
    parser.add_argument("--report", type=Path, help="Optional local JSON report path")
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate and report only (default without --upload)")
    parser.add_argument("--upload", action="store_true",
                        help="Explicitly upload to v3 staging; never clears vectors")
    parser.add_argument("--index", default="portfolio-knowledge-base")
    args = parser.parse_args()

    chunks = build_chunks(
        read_source(args.input), args.input.name,
        max_chars=args.max_chars,
    )
    report = make_report(chunks, args.input, args.namespace)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(report, indent=2, ensure_ascii=False))

    if args.upload:
        upload_staging(chunks, args.index, args.namespace)
        print(f"Uploaded {len(chunks)} chunks to staging namespace {args.namespace}.")
    else:
        print("DRY RUN: no embedding calls, Pinecone mutations or deletions.")


if __name__ == "__main__":
    main()
