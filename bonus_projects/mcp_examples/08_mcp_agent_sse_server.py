import sys as _sys
from pathlib import Path as _Path

_nrp_root = next((p for p in _Path(__file__).resolve().parents if (p / "nrp.py").exists()), None)
if _nrp_root is None:
    raise ImportError(
        "nrp.py not found. Run this script from the AI-Agent-Workflows checkout."
    )
_sys.path.insert(0, str(_nrp_root))
import nrp  # NRP OpenAI-compatible API (loads .env)
import asyncio

from agents import Agent, Runner
from agents.mcp import MCPServerSse


async def main():
    async with MCPServerSse(
        name="SSE Python Server",
        params={
            "url": "http://localhost:8000/sse",
        },
    ) as research_server:
        agent = Agent(
            name="Assistant",
            instructions="""
Use the research tools to perform and plan research.""",
            mcp_servers=[research_server],
        )

        input = """
Get the research plan for the book 
'The Hitchhiker's Guide to the Galaxy'"""
        print("Running: Get the research plan")
        result = await Runner.run(agent, input)
        print(result.final_output)


if __name__ == "__main__":
    asyncio.run(main())
