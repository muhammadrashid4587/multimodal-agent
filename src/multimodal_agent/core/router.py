"""Task router — analyze inputs and select processing strategy."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from multimodal_agent.core.processor import ProcessedInput, ModalityType

logger = logging.getLogger(__name__)


@dataclass
class ProcessingPlan:
    tools: list[str] = field(default_factory=list)
    strategy: str = ""
    reasoning: str = ""
    priority_modality: ModalityType | None = None


class TaskRouter:
    """Route tasks to appropriate tools based on input modalities."""

    TOOL_CAPABILITIES: dict[str, set[ModalityType]] = {
        "vision": {ModalityType.IMAGE},
        "document": {ModalityType.DOCUMENT, ModalityType.TEXT},
        "data_analyzer": {ModalityType.STRUCTURED_DATA},
        "general": {ModalityType.TEXT},
    }

    def route(self, inputs: list[ProcessedInput], task: str = "") -> ProcessingPlan:
        """Create a processing plan based on inputs and task."""
        modalities = set(inp.modality for inp in inputs)
        plan = ProcessingPlan()

        # Select tools based on modalities
        for tool_name, supported in self.TOOL_CAPABILITIES.items():
            if modalities & supported:
                plan.tools.append(tool_name)

        # Determine strategy
        if len(modalities) == 1:
            plan.strategy = "single_modality"
            plan.priority_modality = list(modalities)[0]
        elif ModalityType.IMAGE in modalities:
            plan.strategy = "vision_primary"
            plan.priority_modality = ModalityType.IMAGE
        else:
            plan.strategy = "text_primary"
            plan.priority_modality = ModalityType.TEXT

        # Task-specific routing
        task_lower = task.lower()
        if any(w in task_lower for w in ["compare", "diff", "difference"]):
            plan.strategy = "comparison"
            plan.tools.insert(0, "vision") if "vision" not in plan.tools else None
        elif any(w in task_lower for w in ["extract", "ocr", "read"]):
            plan.strategy = "extraction"
        elif any(w in task_lower for w in ["analyze", "chart", "graph", "trend"]):
            plan.strategy = "analysis"

        plan.reasoning = f"Selected {len(plan.tools)} tools for {len(modalities)} modalities, strategy: {plan.strategy}"
        return plan

    def get_tool_for_modality(self, modality: ModalityType) -> str:
        """Get the best tool for a given modality."""
        for tool_name, supported in self.TOOL_CAPABILITIES.items():
            if modality in supported:
                return tool_name
        return "general"
