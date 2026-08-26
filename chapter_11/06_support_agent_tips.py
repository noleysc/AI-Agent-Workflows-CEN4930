import sys as _sys
from pathlib import Path as _Path

_nrp_root = next((p for p in _Path(__file__).resolve().parents if (p / "nrp.py").exists()), None)
if _nrp_root is None:
    raise ImportError(
        "nrp.py not found. Run this script from the AI-Agent-Workflows checkout."
    )
_sys.path.insert(0, str(_nrp_root))
import nrp  # NRP OpenAI-compatible API (loads .env)
from agents import Agent, ModelSettings, function_tool


@function_tool
def escalate_to_human(ticket_id: str, reason: str) -> str:
    """Escalate this conversation to a human. Use for angry users, large refunds, or unclear policy."""
    # Create/route ticket in your system...
    return f"Escalated ticket {ticket_id}: {reason}"


retrieval_agent = Agent(...)  # add retrieval agent here

support = Agent(
    name="Triage Support Agent",
    instructions=(
        "Verify identity before account actions. Cite policies. "
        "If unsure or user is upset, call escalate_to_human."
        "Pass on complex queries to retrieval_agent."
    ),
    tools=[
        escalate_to_human,
        retrieval_agent.as_tool(),
    ],  # add order lookup, refund APIs, and RAG tool here
    model_settings=ModelSettings(tool_choice="auto"),
)
