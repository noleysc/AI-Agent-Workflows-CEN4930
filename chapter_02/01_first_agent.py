import os
import sys as _sys
from pathlib import Path as _Path

# NRP tokens cannot ingest traces on api.openai.com.
os.environ["OPENAI_AGENTS_DISABLE_TRACING"] = "1"

_nrp_root = next((p for p in _Path(__file__).resolve().parents if (p / "nrp.py").exists()), None)
if _nrp_root is None:
    raise ImportError(
        "nrp.py not found. Run this script from the AI-Agent-Workflows checkout."
    )
_sys.path.insert(0, str(_nrp_root))
import nrp  # NRP OpenAI-compatible API (loads .env)
from agents import Agent, RunConfig, Runner, set_trace_processors, set_tracing_disabled
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
set_tracing_disabled(True)
set_trace_processors([])

# Agent Instructions
instructions = """
You are a research planning assistant.

**TASK INSTRUCTIONS**
- You will be given a research topic.
- Your task is to provide a plan on how to research this topic.
- Output 5 concise tasks (5 words or less) to your plan.
"""

agent = Agent(
    name="Research Planner",
    instructions=instructions,
    model=nrp.CHAT_MODEL,
)

input = "learn about AI agents"

result = Runner.run_sync(
    agent,
    input=input,
    run_config=RunConfig(tracing_disabled=True),
)

print(result.final_output)
