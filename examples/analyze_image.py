"""Example: Analyze an image with text context."""

import asyncio
from multimodal_agent.agents.multimodal_agent import MultiModalAgent


async def main():
    agent = MultiModalAgent()

    # Analyze with text context
    result = await agent.analyze(
        text="This is a quarterly sales report. Identify key trends and anomalies.",
        image_path="chart.png",
        task="Analyze the sales chart and provide insights.",
    )

    print(f"Modalities: {result.modalities_processed}")
    print(f"Tools: {result.tools_used}")
    print(f"\nAnalysis:\n{result.summary}")


if __name__ == "__main__":
    asyncio.run(main())
