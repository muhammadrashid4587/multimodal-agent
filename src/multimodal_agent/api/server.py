"""FastAPI server for the multimodal agent."""

from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel

from multimodal_agent.agents.multimodal_agent import MultiModalAgent

app = FastAPI(title="Multimodal Agent API", version="0.1.0")
agent = MultiModalAgent()


class AnalyzeRequest(BaseModel):
    text: str | None = None
    image_path: str | None = None
    document_path: str | None = None
    task: str = "Analyze the provided inputs."


class DescribeRequest(BaseModel):
    image_path: str


@app.get("/health")
async def health():
    return {"status": "healthy"}


@app.post("/analyze")
async def analyze(request: AnalyzeRequest):
    result = await agent.analyze(
        text=request.text,
        image_path=request.image_path,
        document_path=request.document_path,
        task=request.task,
    )
    return {
        "summary": result.summary,
        "modalities": result.modalities_processed,
        "tools_used": result.tools_used,
    }


@app.post("/describe")
async def describe(request: DescribeRequest):
    description = await agent.describe(request.image_path)
    return {"description": description}
