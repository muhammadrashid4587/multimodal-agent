"""Modality fusion — combine features from different input types."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from multimodal_agent.core.processor import ProcessedInput, ModalityType

logger = logging.getLogger(__name__)


class FusionStrategy(str, Enum):
    CONCATENATION = "concatenation"
    WEIGHTED = "weighted"
    CROSS_MODAL = "cross_modal"


@dataclass
class FusedRepresentation:
    text_context: str = ""
    modalities: list[ModalityType] = field(default_factory=list)
    features: dict[str, Any] = field(default_factory=dict)
    image_data: list[dict[str, str]] = field(default_factory=list)
    strategy_used: FusionStrategy = FusionStrategy.CONCATENATION


class ModalityFusion:
    """Combine features from different modalities into a unified representation."""

    def __init__(self, strategy: FusionStrategy = FusionStrategy.WEIGHTED):
        self.strategy = strategy
        self._modality_weights: dict[ModalityType, float] = {
            ModalityType.IMAGE: 0.4,
            ModalityType.TEXT: 0.3,
            ModalityType.DOCUMENT: 0.3,
            ModalityType.STRUCTURED_DATA: 0.2,
        }

    def fuse(self, inputs: list[ProcessedInput]) -> FusedRepresentation:
        """Fuse multiple processed inputs into a single representation."""
        if self.strategy == FusionStrategy.CONCATENATION:
            return self._concatenation_fusion(inputs)
        elif self.strategy == FusionStrategy.WEIGHTED:
            return self._weighted_fusion(inputs)
        else:
            return self._cross_modal_fusion(inputs)

    def _concatenation_fusion(self, inputs: list[ProcessedInput]) -> FusedRepresentation:
        """Simple concatenation of all text representations."""
        parts: list[str] = []
        image_data: list[dict[str, str]] = []
        all_features: dict[str, Any] = {}

        for inp in inputs:
            if inp.text_representation:
                parts.append(f"[{inp.modality.value}] {inp.text_representation}")
            if inp.base64_data:
                image_data.append({"data": inp.base64_data, "mime_type": inp.mime_type})
            all_features.update(inp.features)

        return FusedRepresentation(
            text_context="\n\n".join(parts),
            modalities=[inp.modality for inp in inputs],
            features=all_features,
            image_data=image_data,
            strategy_used=FusionStrategy.CONCATENATION,
        )

    def _weighted_fusion(self, inputs: list[ProcessedInput]) -> FusedRepresentation:
        """Weight-based fusion prioritizing certain modalities."""
        sorted_inputs = sorted(
            inputs,
            key=lambda i: self._modality_weights.get(i.modality, 0.1),
            reverse=True,
        )

        parts: list[str] = []
        image_data: list[dict[str, str]] = []
        all_features: dict[str, Any] = {}

        for inp in sorted_inputs:
            weight = self._modality_weights.get(inp.modality, 0.1)
            if inp.text_representation:
                parts.append(f"[{inp.modality.value} (w={weight:.1f})] {inp.text_representation}")
            if inp.base64_data:
                image_data.append({"data": inp.base64_data, "mime_type": inp.mime_type})
            for k, v in inp.features.items():
                all_features[f"{inp.modality.value}_{k}"] = v

        return FusedRepresentation(
            text_context="\n\n".join(parts),
            modalities=[inp.modality for inp in sorted_inputs],
            features=all_features,
            image_data=image_data,
            strategy_used=FusionStrategy.WEIGHTED,
        )

    def _cross_modal_fusion(self, inputs: list[ProcessedInput]) -> FusedRepresentation:
        """Cross-modal fusion with explicit connections between modalities."""
        result = self._weighted_fusion(inputs)

        # Add cross-modal context
        modality_set = set(inp.modality for inp in inputs)
        cross_refs: list[str] = []

        if ModalityType.IMAGE in modality_set and ModalityType.TEXT in modality_set:
            cross_refs.append("Cross-modal: Interpret image in light of text context and vice versa.")
        if ModalityType.DOCUMENT in modality_set and ModalityType.STRUCTURED_DATA in modality_set:
            cross_refs.append("Cross-modal: Use structured data to validate document content.")

        if cross_refs:
            result.text_context += "\n\n" + "\n".join(cross_refs)
        result.strategy_used = FusionStrategy.CROSS_MODAL
        return result

    def set_weight(self, modality: ModalityType, weight: float) -> None:
        self._modality_weights[modality] = weight
