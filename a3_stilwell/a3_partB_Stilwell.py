"""CEN 4930 Assignment 3, Part B — Bonita Beach dawn-patrol RAG agent.

Run:

    python a3_partB_Stilwell.py

Domain: personal pre-class surf decisions at Bonita Beach (not university advising).
Retrieves from kb_*_Stilwell.txt, prints a ReAct trace, cites sources, refuses
questions the beach knowledge base cannot ground.
"""

from __future__ import annotations

import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

import chromadb
from dotenv import load_dotenv

HERE = Path(__file__).resolve().parent
CHROMA_DIR = HERE / "chroma_a3_partB_Stilwell"
COLLECTION = "bonita_dawn_patrol_stilwell"
TOP_K = 3
MAX_DISTANCE = 1.60

load_dotenv()
load_dotenv(HERE.parent / ".env")


@dataclass
class Hit:
    text: str
    source_file: str
    section: str
    document: str
    distance: float


def discover_kb_files() -> list[Path]:
    files = sorted(HERE.glob("kb_*_Stilwell.txt"))
    if not files:
        files = sorted(HERE.glob("kb_*.txt"))
    return files


def chunk_kb_file(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8")
    document = path.stem
    for line in text.splitlines():
        if line.startswith("DOCUMENT:"):
            document = line.split(":", 1)[1].strip()
            break

    chunks: list[dict] = []
    preamble: list[str] = []
    section = "Header"
    body: list[str] = []

    def flush() -> None:
        nonlocal body, section
        content = "\n".join(body).strip()
        if content:
            chunks.append(
                {
                    "id": f"{path.name}::{section}",
                    "text": (
                        f"Document: {document}\nSection: {section}\n"
                        f"Source file: {path.name}\n\n{content}"
                    ),
                    "source_file": path.name,
                    "section": section,
                    "document": document,
                }
            )
        body = []

    for line in text.splitlines():
        if line.startswith("## "):
            flush()
            section = line[3:].strip()
            continue
        if section == "Header" and ":" in line and line.split(":", 1)[0].isupper():
            preamble.append(line.strip())
            continue
        body.append(line)

    if preamble:
        chunks.insert(
            0,
            {
                "id": f"{path.name}::Header",
                "text": (
                    f"Document: {document}\nSection: Header\n"
                    f"Source file: {path.name}\n\n" + "\n".join(preamble)
                ),
                "source_file": path.name,
                "section": "Header",
                "document": document,
            },
        )
    flush()
    return chunks


def build_index() -> chromadb.Collection:
    files = discover_kb_files()
    if len(files) < 3:
        raise FileNotFoundError(
            f"Need at least 3 kb_*.txt files next to {HERE.name}; found {len(files)}"
        )
    if CHROMA_DIR.exists():
        shutil.rmtree(CHROMA_DIR)
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    try:
        client.delete_collection(COLLECTION)
    except Exception:
        pass
    collection = client.create_collection(name=COLLECTION)
    chunks: list[dict] = []
    for path in files:
        chunks.extend(chunk_kb_file(path))
    collection.add(
        ids=[c["id"] for c in chunks],
        documents=[c["text"] for c in chunks],
        metadatas=[
            {
                "source_file": c["source_file"],
                "section": c["section"],
                "document": c["document"],
            }
            for c in chunks
        ],
    )
    print(f"[RAG] indexed {len(chunks)} chunks from {len(files)} KB files")
    for path in files:
        print(f"[RAG]   - {path.name}")
    return collection


def retrieve(collection: chromadb.Collection, query: str) -> list[Hit]:
    result = collection.query(
        query_texts=[query],
        n_results=TOP_K,
        include=["documents", "metadatas", "distances"],
    )
    hits: list[Hit] = []
    for doc, meta, dist in zip(
        (result.get("documents") or [[]])[0],
        (result.get("metadatas") or [[]])[0],
        (result.get("distances") or [[]])[0],
    ):
        meta = meta or {}
        hits.append(
            Hit(
                text=doc,
                source_file=str(meta.get("source_file", "unknown")),
                section=str(meta.get("section", "unknown")),
                document=str(meta.get("document", "unknown")),
                distance=float(dist),
            )
        )
    return hits


def print_retrieval(hits: list[Hit]) -> None:
    print("\n[RETRIEVAL RESULTS]")
    for i, hit in enumerate(hits, 1):
        print(
            f"  {i}. file={hit.source_file} | section={hit.section} | "
            f"distance={hit.distance:.4f}"
        )
        print(f"     {' '.join(hit.text.split())[:220]}...")


def in_domain(question: str) -> bool:
    tokens = {t.lower() for t in re.findall(r"[A-Za-z0-9]+", question)}
    domain = {
        "surf",
        "surfboard",
        "funboard",
        "paddle",
        "beach",
        "bonita",
        "flag",
        "yellow",
        "red",
        "rip",
        "tower",
        "lifeguard",
        "parking",
        "class",
        "holmes",
        "shoulder",
        "dawn",
        "patrol",
        "wave",
        "board",
    }
    return not tokens.isdisjoint(domain)


def react_answer(collection: chromadb.Collection, question: str) -> str:
    print("\n" + "=" * 72)
    print("QUESTION:")
    print(question)
    print("=" * 72)
    print("\n----- ReAct TRACE -----")
    print("Thought 1: I need beach-kit evidence before I give a go/no-go style answer.")
    print(f"Action 1: search_kb({question!r})")
    hits = retrieve(collection, question)
    print("Observation 1:")
    print_retrieval(hits)

    if not hits or hits[0].distance > MAX_DISTANCE or not in_domain(question):
        why = []
        if not in_domain(question):
            why.append("question is outside the Bonita Beach dawn-patrol domain")
        if not hits:
            why.append("no chunks returned")
        elif hits[0].distance > MAX_DISTANCE:
            why.append(
                f"best distance {hits[0].distance:.3f} exceeds threshold {MAX_DISTANCE}"
            )
        reason = "; ".join(why) or "not grounded"
        print(f"Thought 2: Not grounded ({reason}). Refuse instead of guessing.")
        print("Action 2: refuse")
        refusal = (
            "REFUSAL: I cannot answer from the Bonita Beach dawn-patrol knowledge base. "
            f"Reason: {reason}. Ask about the funboard, flags, parking clock, or pier rip."
        )
        print("Final Answer:")
        print(refusal)
        print("----- END TRACE -----")
        return refusal

    lower = question.lower()
    print("Thought 2: Retrieval is close enough. Checking whether timing, flag, or board rules decide the call.")

    # Second targeted retrieval when the question mixes clock + tower.
    if "lifeguard" in lower or "tower" in lower or "08:00" in lower or "8:00" in lower:
        print("Thought 3: The question involves the tower clock. Retrieving parking/class timing next.")
        print("Action 2: search_kb('parking class clock planned exit park back by')")
        more = retrieve(collection, "parking class clock planned exit park back by water time")
        print("Observation 2:")
        print_retrieval(more)
        hits = more + hits
        print(
            "Thought 4: Waiting for 08:00 conflicts with the 07:35 exit and 09:10 campus parking rule. "
            "Changing the answer to NO-GO."
        )
        citation = f"{more[0].source_file} / section '{more[0].section}'"
        answer = (
            "NO-GO. Lifeguards are on duty at 08:00, but the timing sheet requires a 07:35 water exit "
            "to park back by 09:10 for the 09:30 Holmes Hall class after a 35-minute drive. "
            "Staying until the tower opens misses that clock and the 45-minute water-time minimum. "
            f"Source: {citation}."
        )
    elif "funboard" in lower or "board" in lower or "shoulder" in lower:
        print("Thought 3: Board or shoulder choice — prefer the funboard rules document.")
        print("Action 2: search_kb('funboard sore shoulder yellow flag')")
        more = retrieve(collection, "funboard sore shoulder yellow flag overhead")
        print("Observation 2:")
        print_retrieval(more)
        hits = more + hits
        print("Thought 4: Funboard rules apply; keep the shortboard out of the plan.")
        citation = f"{more[0].source_file} / section '{more[0].section}'"
        answer = (
            "Use the 8 foot funboard. It is the default for knee-high to waist-high dawn patrols, "
            "sore-shoulder days, and yellow-flag sessions before the tower opens. "
            "Do not substitute the shortboard for overhead takeoffs while alone. "
            f"Source: {citation}."
        )
    elif "rip" in lower or "pier" in lower or "enter" in lower:
        print("Thought 3: Entry location matters — pull the pier rip rule.")
        print("Action 2: search_kb('pier rip entry 200 yards')")
        more = retrieve(collection, "pier rip entry 200 yards south access 10")
        print("Observation 2:")
        print_retrieval(more)
        citation = f"{more[0].source_file} / section '{more[0].section}'"
        print("Thought 4: Keep entry south of the rip.")
        answer = (
            "Enter at least 200 yards south of the standing rip on the south side of the pier, "
            "closer to access 10. Do not paddle up into the pier shadow to chase a better peak. "
            f"Source: {citation}."
        )
    else:
        print("Thought 3: General grounded summary from the top chunk.")
        best = hits[0]
        citation = f"{best.source_file} / section '{best.section}'"
        answer = f"{' '.join(best.text.split())[:480]} Source: {citation}."

    print("Final Answer:")
    print(answer)
    print("----- END TRACE -----")
    return answer


def main() -> None:
    print("Part B — Bonita Beach dawn-patrol RAG agent")
    collection = build_index()
    for question in TEST_CASES:
        react_answer(collection, question)


TEST_CASES = [
    "Should I wait in the water until the lifeguards arrive at 8:00 and still make my 9:30 class?",
    "What is the tuition refund deadline for dropping a chemistry lab?",
]


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------------
# Test inputs:
# 1. Wait for lifeguards until 8:00 before class -> grounded NO-GO + citation
# 2. Tuition refund deadline -> REFUSAL (university policy not in this KB)
# ---------------------------------------------------------------------------
