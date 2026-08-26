import sys as _sys
from pathlib import Path as _Path

_nrp_root = next((p for p in _Path(__file__).resolve().parents if (p / "nrp.py").exists()), None)
if _nrp_root is not None:
    _sys.path.insert(0, str(_nrp_root))
    import nrp  # NRP OpenAI-compatible API (loads .env)
from agents import Agent, function_tool


@function_tool
def retrieve(query: str, corpus: str = "product_docs", top_k: int = 5) -> list[dict]:
    """Return top-k passages with metadata from the selected index."""
    # ... ANN + metadata filtering ...
    return [{"text": "...", "source": "KB-123", "section": "Refunds"}]


answerer = Agent(
    name="RAG-Answerer",
    instructions=(
        "Use ONLY the passages provided via `retrieve`. "
        "If none answer the question, reply: "
        "'I don't know based on the available documents.' "
        "Return a short summary and cite sources."
    ),
    tools=[retrieve],
)
