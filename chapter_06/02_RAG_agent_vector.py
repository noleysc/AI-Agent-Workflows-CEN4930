import sys as _sys
from pathlib import Path as _Path

_nrp_root = next((p for p in _Path(__file__).resolve().parents if (p / "nrp.py").exists()), None)
if _nrp_root is None:
    raise ImportError(
        "nrp.py not found. Run this script from the AI-Agent-Workflows checkout."
    )
_sys.path.insert(0, str(_nrp_root))
import os

os.environ["OPENAI_AGENTS_DISABLE_TRACING"] = "1"

import nrp  # NRP OpenAI-compatible API (loads .env)
import uuid
from pathlib import Path

import chromadb
import tiktoken
from agents import (  # OpenAI Agents SDK
    Agent,
    RunConfig,
    Runner,
    function_tool,
    set_trace_processors,
    set_tracing_disabled,
)
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
set_tracing_disabled(True)
set_trace_processors([])

CHAPTER_DIR = Path(__file__).resolve().parent
SCRIPT_PATH = CHAPTER_DIR / "sample_documents" / "back_to_the_future.txt"
CHROMA_PATH = CHAPTER_DIR / "chroma_script_store"

# ------------------------------------------------------------------
# 1. Load + chunk the script
# ------------------------------------------------------------------
script_text = SCRIPT_PATH.read_text(encoding="utf-8")


def simple_chunk(text, max_tokens=200):
    tokenizer = tiktoken.get_encoding("cl100k_base")
    words, chunk, chunks = text.split(), [], []
    for w in words:
        if len(tokenizer.encode(" ".join(chunk + [w]))) > max_tokens:
            chunks.append(" ".join(chunk))
            chunk = [w]
        else:
            chunk.append(w)
    if chunk:
        chunks.append(" ".join(chunk))
    return chunks


docs = simple_chunk(script_text, max_tokens=200)

# ------------------------------------------------------------------
# 2. Create (or connect to) a Chroma collection with OpenAI embeddings
# ------------------------------------------------------------------
client = chromadb.PersistentClient(
    path=str(CHROMA_PATH)  # on-disk so we reuse later
)
collection_name = "bttf_script"

# Try to get existing collection first
try:
    collection = client.get_collection(collection_name)
except Exception:
    # Collection doesn't exist, create it without embedding function
    collection = client.create_collection(name=collection_name)

# Populate once (skip if already populated)
if collection.count() == 0:
    collection.add(ids=[str(uuid.uuid4()) for _ in docs], documents=docs)


# ------------------------------------------------------------------
# 3. Define a semantic search tool that queries Chroma
# ------------------------------------------------------------------
@function_tool
def search_script(query: str, top_k: int = 3) -> str:
    res = collection.query(query_texts=[query], n_results=top_k)
    if res and "documents" in res and res["documents"] and res["documents"][0]:
        return "\n\n".join(res["documents"][0])  # combine best chunks
    return "No relevant documents found."


# ------------------------------------------------------------------
# 4. Build the agent
# ------------------------------------------------------------------
agent = Agent(
    name="Script Agent",
    model=nrp.CHAT_MODEL,
    instructions=(
        "You answer questions about the movie *Back to the Future*.\n"
        "Call `search_script` at most once per question, then answer from the "
        "returned passages. Do not keep calling tools after you have enough text. "
        "If search returns nothing useful, say you could not find it."
    ),
    tools=[search_script],
)

run_config = RunConfig(tracing_disabled=True)

# ------------------------------------------------------------------
# 5. Ask a question
# ------------------------------------------------------------------
print(f"Using model={nrp.CHAT_MODEL} base_url={nrp.BASE_URL}")
print(f"Chroma collection '{collection_name}' has {collection.count()} chunks")
_key = os.getenv("OPENAI_API_KEY") or ""
print(f"OPENAI_API_KEY set={bool(_key)} length={len(_key)}")
if not _key:
    raise SystemExit(
        "OPENAI_API_KEY is empty. Put your NRP token from https://nrp.ai/llmtoken in .env"
    )

query = "Where does Doc tell Marty to meet him, and at what time?"
try:
    result = Runner.run_sync(agent, query, max_turns=5, run_config=run_config)
except Exception as exc:
    if "403" in str(exc) or type(exc).__name__ == "PermissionDeniedError":
        raise SystemExit(
            "NRP returned 403 (auth/permission). Run: python check_nrp_auth.py\n"
            "Then mint a fresh token at https://nrp.ai/llmtoken and update .env "
            "OPENAI_API_KEY (no quotes)."
        ) from exc
    raise
print("\n--- ANSWER ---\n", result.final_output)

query = "What happens at 1:15AM"
result = Runner.run_sync(agent, query, max_turns=5, run_config=run_config)
print("\n--- ANSWER ---\n", result.final_output)
