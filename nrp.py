"""Point this repo at the National Research Platform (NRP) LLM API.

NRP exposes an OpenAI-compatible Chat Completions endpoint. Import this module
before creating Agents or OpenAI clients.

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

_REPO_ROOT = Path(__file__).resolve().parent
load_dotenv(_REPO_ROOT / ".env")
load_dotenv()

os.environ.setdefault("OPENAI_BASE_URL", NRP_BASE_URL)
os.environ.setdefault("OPENAI_DEFAULT_MODEL", DEFAULT_CHAT_MODEL)
# NRP tokens cannot write traces to platform.openai.com.
os.environ.setdefault("OPENAI_AGENTS_DISABLE_TRACING", "1")

CHAT_MODEL = os.getenv("OPENAI_DEFAULT_MODEL", DEFAULT_CHAT_MODEL)
EMBEDDING_MODEL = os.getenv("NRP_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
VISION_MODEL = os.getenv("NRP_VISION_MODEL", DEFAULT_VISION_MODEL)
BASE_URL = os.environ["OPENAI_BASE_URL"]


def configure() -> None:
    """Apply NRP defaults to the OpenAI Agents SDK (idempotent)."""
    api_key = (os.getenv("OPENAI_API_KEY") or "").strip()
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key

    try:
        from openai import AsyncOpenAI
        from agents import (
            set_default_openai_api,
            set_default_openai_client,
            set_tracing_disabled,
        )
    except ImportError:
        return

    if api_key:
        set_default_openai_client(
            AsyncOpenAI(api_key=api_key, base_url=BASE_URL),
            use_for_tracing=False,
        )
    # NRP implements Chat Completions, not the OpenAI Responses API.
    set_default_openai_api("chat_completions")
    set_tracing_disabled(True)


configure()
