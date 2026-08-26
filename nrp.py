"""Point this repo at the National Research Platform (NRP) LLM API.

NRP exposes an OpenAI-compatible Chat Completions endpoint, not the OpenAI
Responses API. Import this module before creating Agents or OpenAI clients.

Get a token at https://nrp.ai/llmtoken and put it in `.env` as OPENAI_API_KEY.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

NRP_BASE_URL = "https://ellm.nrp-nautilus.io/v1"
DEFAULT_CHAT_MODEL = "gpt-oss"
DEFAULT_EMBEDDING_MODEL = "qwen3-embedding"
DEFAULT_VISION_MODEL = "qwen3-small"

# Live catalog ids from https://ellm.nrp-nautilus.io/v1/models (chat, not embeddings).
NRP_CHAT_MODELS = frozenset(
    {
        "gpt-oss",
        "qwen3",
        "qwen3-small",
        "qwen3-4bit",
        "gemma",
        "gemma-small",
        "gemma-small-e4b",
        "gemma4-12b",
        "gemma4-small",
        "glm-5",
        "kimi",
        "minimax-m2",
        "deepseek-v4-flash",
    }
)

_REPO_ROOT = Path(__file__).resolve().parent
load_dotenv(_REPO_ROOT / ".env")
load_dotenv()

os.environ["OPENAI_BASE_URL"] = (
    os.getenv("OPENAI_BASE_URL") or NRP_BASE_URL
).rstrip("/")
if os.environ["OPENAI_BASE_URL"].endswith("/chat/completions"):
    os.environ["OPENAI_BASE_URL"] = os.environ["OPENAI_BASE_URL"][: -len("/chat/completions")]

_requested_model = (os.getenv("OPENAI_DEFAULT_MODEL") or DEFAULT_CHAT_MODEL).strip()
if _requested_model.lower() not in NRP_CHAT_MODELS:
    _requested_model = DEFAULT_CHAT_MODEL
os.environ["OPENAI_DEFAULT_MODEL"] = _requested_model

# Must be set before the Agents SDK tracing provider first reads the env var.
os.environ["OPENAI_AGENTS_DISABLE_TRACING"] = "1"

CHAT_MODEL = os.environ["OPENAI_DEFAULT_MODEL"]
EMBEDDING_MODEL = os.getenv("NRP_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
VISION_MODEL = os.getenv("NRP_VISION_MODEL", DEFAULT_VISION_MODEL)
BASE_URL = os.environ["OPENAI_BASE_URL"]

_configured = False


def configure() -> None:
    """Apply NRP defaults to the OpenAI Agents SDK (idempotent)."""
    global _configured

    api_key = (os.getenv("OPENAI_API_KEY") or "").strip()
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key

    try:
        from openai import AsyncOpenAI
        from agents import (
            Agent,
            set_default_openai_api,
            set_default_openai_client,
            set_tracing_disabled,
        )
        from agents.models.multi_provider import MultiProvider
        from agents.run_config import RunConfig
    except ImportError:
        return

    if api_key:
        set_default_openai_client(
            AsyncOpenAI(api_key=api_key, base_url=BASE_URL),
            use_for_tracing=False,
        )
    # NRP implements Chat Completions. The SDK default is /v1/responses, which
    # Envoy AI Gateway rejects with "No matching route found".
    set_default_openai_api("chat_completions")
    set_tracing_disabled(True)

    if _configured:
        return
    _configured = True

    _orig_mp_init = MultiProvider.__init__

    def _mp_init(self, *args, **kwargs):
        kwargs.setdefault("openai_use_responses", False)
        _orig_mp_init(self, *args, **kwargs)

    MultiProvider.__init__ = _mp_init  # type: ignore[method-assign]

    _orig_agent_init = Agent.__init__

    def _agent_init(self, *args, **kwargs):
        model = kwargs.get("model")
        if model is None:
            kwargs["model"] = CHAT_MODEL
        elif isinstance(model, str) and model.lower() not in NRP_CHAT_MODELS:
            # Leftover OpenAI ids such as gpt-4o / gpt-5.6-luna are not in the NRP gateway.
            kwargs["model"] = CHAT_MODEL
        _orig_agent_init(self, *args, **kwargs)
        if getattr(self, "model", None) is None:
            self.model = CHAT_MODEL

    Agent.__init__ = _agent_init  # type: ignore[method-assign]

    _orig_run_init = RunConfig.__init__

    def _run_init(self, *args, **kwargs):
        kwargs.setdefault("tracing_disabled", True)
        _orig_run_init(self, *args, **kwargs)

    RunConfig.__init__ = _run_init  # type: ignore[method-assign]


configure()
