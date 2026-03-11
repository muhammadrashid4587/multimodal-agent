"""Document processing tool for parsing, extraction, and Q&A."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import anthropic

logger = logging.getLogger(__name__)


@dataclass
class DocumentResult:
    content: str = ""
    tables: list[list[list[str]]] = field(default_factory=list)
    summary: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


class DocumentTool:
    """Document processing: parse, extract tables, summarize, Q&A."""

    def __init__(self, model: str = "claude-sonnet-4-6"):
        self.model = model
        self._client = anthropic.AsyncAnthropic()

    async def parse_document(self, file_path: str) -> DocumentResult:
        """Parse a document and extract its content."""
        path = Path(file_path)
        suffix = path.suffix.lower()

        if suffix in {".txt", ".md"}:
            content = path.read_text()
        elif suffix == ".html":
            content = self._strip_html(path.read_text())
        elif suffix == ".csv":
            content = path.read_text()
        else:
            content = f"[Unsupported format: {suffix}]"

        word_count = len(content.split())
        return DocumentResult(
            content=content,
            metadata={"file": path.name, "format": suffix, "word_count": word_count},
        )

    async def extract_tables(self, file_path: str) -> DocumentResult:
        """Extract tables from a document."""
        doc = await self.parse_document(file_path)
        if not doc.content:
            return doc

        response = await self._client.messages.create(
            model=self.model, max_tokens=2048,
            system="Extract all tables from the document. Return each table as a list of rows, where each row is a list of cell values. Return JSON: [[row1], [row2], ...]",
            messages=[{"role": "user", "content": f"Extract tables from:\n{doc.content[:3000]}"}],
        )

        import json
        try:
            text = response.content[0].text
            start = text.index("[")
            end = text.rindex("]") + 1
            tables = json.loads(text[start:end])
            if tables and isinstance(tables[0], list) and isinstance(tables[0][0], str):
                doc.tables = [tables]
            else:
                doc.tables = tables
        except (ValueError, json.JSONDecodeError):
            pass
        return doc

    async def summarize(self, file_path: str, max_length: str = "medium") -> DocumentResult:
        """Summarize a document."""
        doc = await self.parse_document(file_path)
        length_hints = {"short": "2-3 sentences", "medium": "1-2 paragraphs", "long": "detailed multi-paragraph"}
        hint = length_hints.get(max_length, "1-2 paragraphs")

        response = await self._client.messages.create(
            model=self.model, max_tokens=1024,
            system=f"Summarize the document in {hint}.",
            messages=[{"role": "user", "content": doc.content[:5000]}],
        )
        doc.summary = response.content[0].text.strip()
        return doc

    async def answer_question(self, file_path: str, question: str) -> DocumentResult:
        """Answer a question about a document."""
        doc = await self.parse_document(file_path)
        response = await self._client.messages.create(
            model=self.model, max_tokens=1024,
            system="Answer the question based on the document content. Cite specific parts of the document.",
            messages=[{"role": "user", "content": f"Document:\n{doc.content[:5000]}\n\nQuestion: {question}"}],
        )
        doc.summary = response.content[0].text.strip()
        return doc

    def _strip_html(self, html: str) -> str:
        """Simple HTML tag stripping."""
        import re
        clean = re.sub(r"<[^>]+>", " ", html)
        return re.sub(r"\s+", " ", clean).strip()
