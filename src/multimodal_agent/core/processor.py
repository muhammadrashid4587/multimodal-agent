"""Multi-modal input processor with routing and type detection."""

from __future__ import annotations

import base64
import logging
import mimetypes
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class ModalityType(str, Enum):
    IMAGE = "image"
    TEXT = "text"
    DOCUMENT = "document"
    STRUCTURED_DATA = "structured_data"
    AUDIO_METADATA = "audio_metadata"


class InputPayload(BaseModel):
    modality: ModalityType
    content: str | bytes = ""
    file_path: str | None = None
    mime_type: str = ""
    metadata: dict[str, Any] = {}


class ProcessedInput(BaseModel):
    modality: ModalityType
    text_representation: str = ""
    base64_data: str | None = None
    mime_type: str = ""
    features: dict[str, Any] = {}
    metadata: dict[str, Any] = {}


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".svg"}
DOCUMENT_EXTENSIONS = {".pdf", ".doc", ".docx", ".txt", ".md", ".html", ".csv"}


class MultiModalProcessor:
    """Process different input types into a unified format."""

    def detect_modality(self, file_path: str | None = None, content: Any = None) -> ModalityType:
        """Detect the modality of an input."""
        if file_path:
            suffix = Path(file_path).suffix.lower()
            if suffix in IMAGE_EXTENSIONS:
                return ModalityType.IMAGE
            if suffix in DOCUMENT_EXTENSIONS:
                return ModalityType.DOCUMENT
        if isinstance(content, dict | list):
            return ModalityType.STRUCTURED_DATA
        return ModalityType.TEXT

    def process(self, payload: InputPayload) -> ProcessedInput:
        """Process an input payload into a standardized format."""
        if payload.modality == ModalityType.IMAGE:
            return self._process_image(payload)
        elif payload.modality == ModalityType.DOCUMENT:
            return self._process_document(payload)
        elif payload.modality == ModalityType.STRUCTURED_DATA:
            return self._process_structured(payload)
        else:
            return self._process_text(payload)

    def process_file(self, file_path: str) -> ProcessedInput:
        """Process a file, auto-detecting its modality."""
        modality = self.detect_modality(file_path=file_path)
        mime_type = mimetypes.guess_type(file_path)[0] or ""
        payload = InputPayload(modality=modality, file_path=file_path, mime_type=mime_type)
        return self.process(payload)

    def _process_image(self, payload: InputPayload) -> ProcessedInput:
        """Process an image input."""
        base64_data = None
        mime_type = payload.mime_type or "image/png"

        if payload.file_path:
            data = Path(payload.file_path).read_bytes()
            base64_data = base64.standard_b64encode(data).decode()
            mime_type = mimetypes.guess_type(payload.file_path)[0] or mime_type

            # Get image dimensions
            features: dict[str, Any] = {"file_size": len(data)}
            try:
                from PIL import Image
                img = Image.open(payload.file_path)
                features["width"] = img.width
                features["height"] = img.height
                features["format"] = img.format
                features["mode"] = img.mode
            except Exception:
                pass

            return ProcessedInput(
                modality=ModalityType.IMAGE,
                base64_data=base64_data,
                mime_type=mime_type,
                features=features,
            )

        return ProcessedInput(modality=ModalityType.IMAGE, mime_type=mime_type)

    def _process_document(self, payload: InputPayload) -> ProcessedInput:
        """Process a document input."""
        text = ""
        if payload.file_path:
            path = Path(payload.file_path)
            if path.suffix in {".txt", ".md", ".html", ".csv"}:
                text = path.read_text()
            else:
                text = f"[Document: {path.name}]"

        return ProcessedInput(
            modality=ModalityType.DOCUMENT,
            text_representation=text,
            features={"char_count": len(text), "word_count": len(text.split())},
        )

    def _process_structured(self, payload: InputPayload) -> ProcessedInput:
        """Process structured data."""
        import json
        text = json.dumps(payload.content, indent=2, default=str) if isinstance(payload.content, dict | list) else str(payload.content)

        return ProcessedInput(
            modality=ModalityType.STRUCTURED_DATA,
            text_representation=text,
            features={"type": type(payload.content).__name__},
        )

    def _process_text(self, payload: InputPayload) -> ProcessedInput:
        """Process text input."""
        text = str(payload.content) if payload.content else ""
        if payload.file_path:
            text = Path(payload.file_path).read_text()

        return ProcessedInput(
            modality=ModalityType.TEXT,
            text_representation=text,
            features={"char_count": len(text), "word_count": len(text.split())},
        )
