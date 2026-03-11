"""Analyst agent specialized in visual data analysis."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import anthropic

from multimodal_agent.tools.vision import VisionTool

logger = logging.getLogger(__name__)


@dataclass
class AnalysisInsight:
    category: str
    finding: str
    confidence: float = 0.0
    evidence: str = ""


@dataclass
class VisualAnalysis:
    insights: list[AnalysisInsight] = field(default_factory=list)
    summary: str = ""
    data_points: list[dict[str, Any]] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)


class AnalystAgent:
    """Specialized in analyzing visual data: charts, screenshots, diagrams."""

    def __init__(self, model: str = "claude-sonnet-4-6"):
        self.model = model
        self.vision = VisionTool(model=model)
        self._client = anthropic.AsyncAnthropic()

    async def analyze_chart(self, image_path: str, context: str = "") -> VisualAnalysis:
        """Analyze a chart or graph image."""
        result = await self.vision.answer_about_image(
            image_path,
            f"Analyze this chart/graph. {context}\n\n"
            "Provide: 1) What type of chart it is 2) Key data points and trends "
            "3) Notable patterns or anomalies 4) Actionable insights",
        )

        insights = await self._extract_insights(result.description)
        return VisualAnalysis(
            insights=insights,
            summary=result.description,
            recommendations=[i.finding for i in insights if i.category == "recommendation"],
        )

    async def analyze_screenshot(self, image_path: str, focus: str = "") -> VisualAnalysis:
        """Analyze a UI screenshot."""
        prompt = "Analyze this screenshot."
        if focus:
            prompt += f" Focus on: {focus}"
        prompt += "\n\nDescribe: 1) Layout and structure 2) UI elements 3) Content 4) UX observations"

        result = await self.vision.answer_about_image(image_path, prompt)
        insights = await self._extract_insights(result.description)
        return VisualAnalysis(insights=insights, summary=result.description)

    async def analyze_diagram(self, image_path: str) -> VisualAnalysis:
        """Analyze a technical diagram (architecture, flow, etc.)."""
        result = await self.vision.answer_about_image(
            image_path,
            "Analyze this technical diagram. Describe: 1) Type of diagram "
            "2) Components and their relationships 3) Data flow or process flow "
            "4) Key architectural decisions or patterns",
        )
        insights = await self._extract_insights(result.description)
        return VisualAnalysis(insights=insights, summary=result.description)

    async def _extract_insights(self, analysis_text: str) -> list[AnalysisInsight]:
        """Extract structured insights from analysis text."""
        response = await self._client.messages.create(
            model=self.model, max_tokens=1024,
            system=(
                "Extract key insights from this analysis. For each insight, provide:\n"
                "- category: trend/anomaly/recommendation/observation\n"
                "- finding: concise description\n"
                "- confidence: 0.0-1.0\n"
                "Return one insight per line in format: [category] finding (confidence: X.X)"
            ),
            messages=[{"role": "user", "content": analysis_text}],
        )

        insights: list[AnalysisInsight] = []
        for line in response.content[0].text.split("\n"):
            line = line.strip()
            if not line or not line.startswith("["):
                continue
            try:
                bracket_end = line.index("]")
                category = line[1:bracket_end].lower()
                rest = line[bracket_end + 1:].strip()
                confidence = 0.7
                if "(confidence:" in rest:
                    conf_start = rest.index("(confidence:")
                    conf_str = rest[conf_start + 12:].rstrip(")")
                    confidence = float(conf_str.strip())
                    rest = rest[:conf_start].strip()
                insights.append(AnalysisInsight(category=category, finding=rest, confidence=confidence))
            except (ValueError, IndexError):
                continue
        return insights
