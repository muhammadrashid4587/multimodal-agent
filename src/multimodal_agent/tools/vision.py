"""Vision tool for image analysis using Claude's vision API."""

from __future__ import annotations

import base64
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import anthropic

logger = logging.getLogger(__name__)


@dataclass
class VisionResult:
    description: str = ""
    extracted_text: str = ""
    objects: list[dict[str, Any]] = None
    metadata: dict[str, Any] = None

    def __post_init__(self):
        if self.objects is None:
            self.objects = []
        if self.metadata is None:
            self.metadata = {}


class VisionTool:
    """Image analysis using Claude's vision capabilities."""

    def __init__(self, model: str = "claude-sonnet-4-6"):
        self.model = model
        self._client = anthropic.AsyncAnthropic()

    async def describe_image(self, image_path: str, detail: str = "detailed") -> VisionResult:
        """Describe an image in natural language."""
        image_content = self._load_image(image_path)
        response = await self._client.messages.create(
            model=self.model, max_tokens=1024,
            messages=[{
                "role": "user",
                "content": [
                    image_content,
                    {"type": "text", "text": f"Describe this image in {detail} detail."},
                ],
            }],
        )
        return VisionResult(description=response.content[0].text)

    async def extract_text(self, image_path: str) -> VisionResult:
        """Extract text (OCR) from an image."""
        image_content = self._load_image(image_path)
        response = await self._client.messages.create(
            model=self.model, max_tokens=2048,
            messages=[{
                "role": "user",
                "content": [
                    image_content,
                    {"type": "text", "text": "Extract ALL text visible in this image. Preserve formatting and layout."},
                ],
            }],
        )
        return VisionResult(extracted_text=response.content[0].text)

    async def detect_objects(self, image_path: str) -> VisionResult:
        """Detect and list objects in an image."""
        image_content = self._load_image(image_path)
        response = await self._client.messages.create(
            model=self.model, max_tokens=1024,
            messages=[{
                "role": "user",
                "content": [
                    image_content,
                    {"type": "text", "text": "List all distinct objects visible in this image. For each, provide: name, approximate position (top-left/center/etc), and confidence."},
                ],
            }],
        )
        return VisionResult(description=response.content[0].text)

    async def compare_images(self, image_path_1: str, image_path_2: str) -> VisionResult:
        """Compare two images and describe differences."""
        img1 = self._load_image(image_path_1)
        img2 = self._load_image(image_path_2)
        response = await self._client.messages.create(
            model=self.model, max_tokens=1024,
            messages=[{
                "role": "user",
                "content": [
                    img1,
                    img2,
                    {"type": "text", "text": "Compare these two images. Describe the key similarities and differences."},
                ],
            }],
        )
        return VisionResult(description=response.content[0].text)

    async def answer_about_image(self, image_path: str, question: str) -> VisionResult:
        """Answer a question about an image."""
        image_content = self._load_image(image_path)
        response = await self._client.messages.create(
            model=self.model, max_tokens=1024,
            messages=[{
                "role": "user",
                "content": [
                    image_content,
                    {"type": "text", "text": question},
                ],
            }],
        )
        return VisionResult(description=response.content[0].text)

    def _load_image(self, image_path: str) -> dict[str, Any]:
        """Load an image and prepare it for the API."""
        data = Path(image_path).read_bytes()
        b64 = base64.standard_b64encode(data).decode()

        import mimetypes
        mime_type = mimetypes.guess_type(image_path)[0] or "image/png"

        return {
            "type": "image",
            "source": {"type": "base64", "media_type": mime_type, "data": b64},
        }
