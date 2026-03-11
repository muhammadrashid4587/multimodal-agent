"""Multimodal Agent — Process images, text, and documents."""

__version__ = "0.1.0"

from multimodal_agent.core.processor import MultiModalProcessor, ModalityType
from multimodal_agent.core.router import TaskRouter
from multimodal_agent.agents.multimodal_agent import MultiModalAgent

__all__ = ["MultiModalProcessor", "ModalityType", "TaskRouter", "MultiModalAgent"]
