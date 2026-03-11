"""Tests for processor, router, and fusion."""

import pytest
from multimodal_agent.core.processor import MultiModalProcessor, InputPayload, ModalityType
from multimodal_agent.core.router import TaskRouter
from multimodal_agent.core.fusion import ModalityFusion, FusionStrategy


class TestProcessor:
    def test_detect_image(self):
        proc = MultiModalProcessor()
        assert proc.detect_modality(file_path="photo.png") == ModalityType.IMAGE
        assert proc.detect_modality(file_path="doc.jpg") == ModalityType.IMAGE

    def test_detect_document(self):
        proc = MultiModalProcessor()
        assert proc.detect_modality(file_path="file.pdf") == ModalityType.DOCUMENT
        assert proc.detect_modality(file_path="notes.md") == ModalityType.DOCUMENT

    def test_detect_text(self):
        proc = MultiModalProcessor()
        assert proc.detect_modality(content="hello") == ModalityType.TEXT

    def test_detect_structured(self):
        proc = MultiModalProcessor()
        assert proc.detect_modality(content={"key": "value"}) == ModalityType.STRUCTURED_DATA

    def test_process_text(self):
        proc = MultiModalProcessor()
        result = proc.process(InputPayload(modality=ModalityType.TEXT, content="Hello world"))
        assert result.text_representation == "Hello world"
        assert result.features["word_count"] == 2


class TestRouter:
    def test_single_image(self):
        router = TaskRouter()
        from multimodal_agent.core.processor import ProcessedInput
        inputs = [ProcessedInput(modality=ModalityType.IMAGE)]
        plan = router.route(inputs)
        assert "vision" in plan.tools
        assert plan.strategy == "single_modality"

    def test_image_and_text(self):
        router = TaskRouter()
        from multimodal_agent.core.processor import ProcessedInput
        inputs = [
            ProcessedInput(modality=ModalityType.IMAGE),
            ProcessedInput(modality=ModalityType.TEXT, text_representation="describe this"),
        ]
        plan = router.route(inputs)
        assert "vision" in plan.tools
        assert plan.strategy == "vision_primary"

    def test_comparison_task(self):
        router = TaskRouter()
        from multimodal_agent.core.processor import ProcessedInput
        inputs = [ProcessedInput(modality=ModalityType.IMAGE)]
        plan = router.route(inputs, task="Compare these two screenshots")
        assert plan.strategy == "comparison"


class TestFusion:
    def test_concatenation(self):
        fusion = ModalityFusion(strategy=FusionStrategy.CONCATENATION)
        from multimodal_agent.core.processor import ProcessedInput
        inputs = [
            ProcessedInput(modality=ModalityType.TEXT, text_representation="Hello"),
            ProcessedInput(modality=ModalityType.TEXT, text_representation="World"),
        ]
        result = fusion.fuse(inputs)
        assert "Hello" in result.text_context
        assert "World" in result.text_context

    def test_weighted(self):
        fusion = ModalityFusion(strategy=FusionStrategy.WEIGHTED)
        from multimodal_agent.core.processor import ProcessedInput
        inputs = [
            ProcessedInput(modality=ModalityType.IMAGE, text_representation="image data"),
            ProcessedInput(modality=ModalityType.TEXT, text_representation="text data"),
        ]
        result = fusion.fuse(inputs)
        assert len(result.modalities) == 2
