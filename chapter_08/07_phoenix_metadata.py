import sys as _sys
from pathlib import Path as _Path

_nrp_root = next((p for p in _Path(__file__).resolve().parents if (p / "nrp.py").exists()), None)
if _nrp_root is not None:
    _sys.path.insert(0, str(_nrp_root))
    import nrp  # NRP OpenAI-compatible API (loads .env)
from agents import trace
from openinference.instrumentation import using_metadata, using_session, using_user
from phoenix.otel import register

register(project_name="Agents In Action")  # A

with (
    using_session("s-123"),
    using_user("u-42"),
    using_metadata({"turn_id": "t-1", "intent": "generate_image"}),
):
    with trace("agent-turn"):  # B
        pass  # C
        # ... your agent code here ...
