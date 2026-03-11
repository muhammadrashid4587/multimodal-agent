# Multimodal Agent

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Multi-modal AI agent** — Process images, text, and documents with intelligent routing and modality fusion.

## Architecture

```
┌──────────────────────────────────────────────┐
│          Multimodal Agent                    │
│  Plan tools │ Execute │ Synthesize results   │
├──────────────────────────────────────────────┤
│              Task Router                     │
│  Analyze modalities → Select strategy        │
├──────────────────────────────────────────────┤
│  ┌────────────┐  ┌──────────┐  ┌──────────┐ │
│  │ Vision     │  │ Document │  │ Modality │ │
│  │ Tool       │  │ Tool     │  │ Fusion   │ │
│  │ Describe   │  │ Parse    │  │ Combine  │ │
│  │ Extract    │  │ Tables   │  │ Attend   │ │
│  │ Compare    │  │ Q&A      │  │ Merge    │ │
│  └────────────┘  └──────────┘  └──────────┘ │
├──────────────────────────────────────────────┤
│           Processor │ FastAPI                │
└──────────────────────────────────────────────┘
```

## Features

- **Vision Analysis** — Image description, OCR, object detection, comparison
- **Document Processing** — PDF parsing, table extraction, Q&A
- **Modality Fusion** — Combine insights from different input types
- **Smart Routing** — Auto-detect input types and select processing strategy
- **Analyst Agent** — Specialized analysis of charts, screenshots, diagrams

## Quick Start

```python
from multimodal_agent import MultiModalAgent

agent = MultiModalAgent()
result = await agent.analyze(image_path="chart.png", text="Explain the trends")
print(result.summary)
```

## License

MIT
