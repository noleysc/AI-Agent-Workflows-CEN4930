#!/usr/bin/env python3
"""Quick NRP auth check. Run from repo root: python check_nrp_auth.py"""

from __future__ import annotations

import os
import sys

import nrp
from openai import OpenAI


def main() -> int:
    key = os.getenv("OPENAI_API_KEY") or ""
    print(f"base_url={nrp.BASE_URL}")
    print(f"model={nrp.CHAT_MODEL}")
    print(f"OPENAI_API_KEY set={bool(key)} length={len(key)}")
    if key:
        print(f"OPENAI_API_KEY prefix={key[:6]!r} suffix={key[-4:]!r}")
    else:
        print("ERROR: OPENAI_API_KEY is empty. Put your NRP token in .env")
        print("Create one at https://nrp.ai/llmtoken")
        return 1

    client = OpenAI(api_key=key, base_url=nrp.BASE_URL)
    try:
        models = [m.id for m in client.models.list().data]
        print(f"GET /models OK ({len(models)} models)")
    except Exception as exc:
        print(f"GET /models FAILED: {type(exc).__name__}: {exc}")
        print("Your token may be missing, expired, or not LLM-enabled.")
        return 2

    try:
        completion = client.chat.completions.create(
            model=nrp.CHAT_MODEL,
            messages=[{"role": "user", "content": "Reply with exactly: NRP_OK"}],
            max_tokens=16,
            temperature=0,
        )
        text = (completion.choices[0].message.content or "").strip()
        print(f"POST /chat/completions OK: {text!r}")
        return 0
    except Exception as exc:
        print(f"POST /chat/completions FAILED: {type(exc).__name__}: {exc}")
        print()
        print("403 usually means:")
        print("  1. OPENAI_API_KEY is not a valid NRP LLM token")
        print("  2. Your NRP group does not have the LLM flag")
        print("  3. The token was pasted with quotes/spaces — recreate .env")
        print("Mint a fresh token at https://nrp.ai/llmtoken and replace OPENAI_API_KEY.")
        return 3


if __name__ == "__main__":
    sys.exit(main())
