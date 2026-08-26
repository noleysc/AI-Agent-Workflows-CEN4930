import sys as _sys
from pathlib import Path as _Path

_nrp_root = next((p for p in _Path(__file__).resolve().parents if (p / "nrp.py").exists()), None)
if _nrp_root is not None:
    _sys.path.insert(0, str(_nrp_root))
    import nrp  # NRP OpenAI-compatible API (loads .env)
from typing import List

from agents import Agent, Runner
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel

# Load environment variables from .env file
load_dotenv()

# Create an MCP server
mcp = FastMCP("Research Tools")

# Agent Instructions
instructions = """
You are a research planning assistant.

**TASK INSTRUCTIONS**
- You will be given a research topic.
- Your task is to provide a plan on how to research this topic.
- Output 5 concise tasks (5 words or less) to your plan.
"""

class ResearchPlanModel(BaseModel):
    tasks: List[str]
    """A list of tasks to perform for research."""

@mcp.tool()
async def get_research_plan(subject: str) -> ResearchPlanModel:
    try:
        agent = Agent(
            name="Research Planner",
            instructions=instructions,
            output_type=ResearchPlanModel,
        )       

        result = await Runner.run(
            agent,
            input=subject,
        )

        return result.final_output
    except Exception as e:
        return ResearchPlanModel(tasks=[str(e)])


if __name__ == "__main__":
    mcp.run(transport="sse")
