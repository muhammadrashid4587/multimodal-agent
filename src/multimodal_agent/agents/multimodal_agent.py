"""Multi-modal agent that processes any combination of inputs."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import anthropic

from multimodal_agent.core.processor import MultiModalProcessor, InputPayload, ProcessedInput, ModalityType
from multimodal_agent.core.fusion import ModalityFusion, FusedRepresentation
from multimodal_agent.core.router import TaskRouter
from multimodal_agent.tools.vision import VisionTool
from multimodal_agent.tools.document import DocumentTool

logger = logging.getLogger(__name__)


@dataclass
class AnalysisResult:
    summary: str = ""
    details: dict[str, Any] = field(default_factory=dict)
    modalities_processed: list[str] = field(default_factory=list)
    tools_used: list[str] = field(default_factory=list)


class MultiModalAgent:
    """Agent that processes any combination of inputs (images + text + data)."""

    def __init__(self, model: str = "claude-sonnet-4-6"):
        self.model = model
        self.processor = MultiModalProcessor()
        self.fusion = ModalityFusion()
        self.router = TaskRouter()
        self.vision = VisionTool(model=model)
        self.document = DocumentTool(model=model)
        self._client = anthropic.AsyncAnthropic()

    async def analyze(
        self,
        text: str | None = None,
        image_path: str | None = None,
        document_path: str | None = None,
        data: dict[str, Any] | None = None,
        task: str = "Analyze the provided inputs comprehensively.",
    ) -> AnalysisResult:
        """Analyze a combination of inputs."""
        inputs: list[ProcessedInput] = []

        if text:
            payload = InputPayload(modality=ModalityType.TEXT, content=text)
            inputs.append(self.processor.process(payload))

        if image_path:
            payload = InputPayload(modality=ModalityType.IMAGE, file_path=image_path)
            inputs.append(self.processor.process(payload))

        if document_path:
            payload = InputPayload(modality=ModalityType.DOCUMENT, file_path=document_path)
            inputs.append(self.processor.process(payload))

        if data:
            payload = InputPayload(modality=ModalityType.STRUCTURED_DATA, content=data)
            inputs.append(self.processor.process(payload))

        if not inputs:
            return AnalysisResult(summary="No inputs provided.")

        # Route and plan
        plan = self.router.route(inputs, task)
        fused = self.fusion.fuse(inputs)

        # Build the analysis prompt
        messages_content: list[dict[str, Any]] = []

        # Add images first
        for img_data in fused.image_data:
            messages_content.append({
                "type": "image",
                "source": {"type": "base64", "media_type": img_data["mime_type"], "data": img_data["data"]},
            })

        # Add text context
        context = f"Task: {task}\n\nAvailable context:\n{fused.text_context}" if fused.text_context else f"Task: {task}"
        messages_content.append({"type": "text", "text": context})

        response = await self._client.messages.create(
            model=self.model, max_tokens=2048,
            system="You are a multi-modal analysis assistant. Analyze all provided inputs comprehensively.",
            messages=[{"role": "user", "content": messages_content}],
        )

        return AnalysisResult(
            summary=response.content[0].text,
            details={"features": fused.features, "strategy": plan.strategy},
            modalities_processed=[m.value for m in fused.modalities],
            tools_used=plan.tools,
        )

    async def describe(self, image_path: str) -> str:
        """Quick image description."""
        result = await self.vision.describe_image(image_path)
        return result.description

    async def extract_text(self, image_path: str) -> str:
        """Extract text from an image."""
        result = await self.vision.extract_text(image_path)
        return result.extracted_text

    async def summarize_document(self, document_path: str) -> str:
        """Summarize a document."""
        result = await self.document.summarize(document_path)
        return result.summary
