"""CEN 4930 Assignment 3, Part A — RAG advising agent with ReAct.

Run from any working directory:

    python a3_partA_Stilwell.py

Builds a ChromaDB index from course_*_Stilwell.txt files next to this script,
retrieves with semantic search, prints an explicit ReAct reasoning trace, cites
source file + section, and refuses questions that are not grounded in the KB.
"""

from __future__ import annotations

import os
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import chromadb
from dotenv import load_dotenv

HERE = Path(__file__).resolve().parent
CHROMA_DIR = HERE / "chroma_a3_partA_Stilwell"
COLLECTION = "fgcu_courses_stilwell"
TOP_K = 4
# Chroma default ONNX MiniLM uses L2 distance; lower is better.
MAX_DISTANCE = 1.55
COURSE_CODE = re.compile(r"\b([A-Z]{3})\s*(\d{4})\b", re.I)

load_dotenv()
load_dotenv(HERE.parent / ".env")


@dataclass
class Chunk:
    chunk_id: str
    text: str
    source_file: str
    section: str
    course_number: str


@dataclass
class Hit:
    text: str
    source_file: str
    section: str
    course_number: str
    distance: float


def discover_course_files() -> list[Path]:
    files = sorted(HERE.glob("course_*_Stilwell.txt"))
    if not files:
        files = sorted(HERE.glob("course_*.txt"))
    return files


def chunk_course_file(path: Path) -> list[Chunk]:
    """Chunk by labeled preamble lines and ## sections for citeable units."""
    text = path.read_text(encoding="utf-8")
    course = path.stem.replace("course_", "").replace("_Stilwell", "").replace("_", " ")
    match = COURSE_CODE.search(text)
    if match:
        course = f"{match.group(1).upper()} {match.group(2)}"

    chunks: list[Chunk] = []
    preamble: list[str] = []
    section = "Header"
    body: list[str] = []

    def flush() -> None:
        nonlocal body, section
        content = "\n".join(body).strip()
        if content:
            chunks.append(
                Chunk(
                    chunk_id=f"{path.name}::{section}",
                    text=f"Course {course}\nSection: {section}\nSource file: {path.name}\n\n{content}",
                    source_file=path.name,
                    section=section,
                    course_number=course,
                )
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
            Chunk(
                chunk_id=f"{path.name}::Header",
                text=(
                    f"Course {course}\nSection: Header\nSource file: {path.name}\n\n"
                    + "\n".join(preamble)
                ),
                source_file=path.name,
                section="Header",
                course_number=course,
            ),
        )
    flush()
    return chunks


def build_index(rebuild: bool = True) -> chromadb.Collection:
    files = discover_course_files()
    if len(files) < 5:
        raise FileNotFoundError(
            f"Need at least 5 course_*.txt files next to {HERE.name}; found {len(files)}"
        )

    if rebuild and CHROMA_DIR.exists():
        shutil.rmtree(CHROMA_DIR)
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    try:
        client.delete_collection(COLLECTION)
    except Exception:
        pass
    collection = client.create_collection(name=COLLECTION)

    chunks: list[Chunk] = []
    for path in files:
        chunks.extend(chunk_course_file(path))

    collection.add(
        ids=[c.chunk_id for c in chunks],
        documents=[c.text for c in chunks],
        metadatas=[
            {
                "source_file": c.source_file,
                "section": c.section,
                "course_number": c.course_number,
            }
            for c in chunks
        ],
    )
    print(f"[RAG] indexed {len(chunks)} chunks from {len(files)} course files into {CHROMA_DIR.name}")
    for path in files:
        print(f"[RAG]   - {path.name}")
    return collection


def retrieve(collection: chromadb.Collection, query: str, top_k: int = TOP_K) -> list[Hit]:
    result = collection.query(
        query_texts=[query],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )
    hits: list[Hit] = []
    docs = (result.get("documents") or [[]])[0]
    metas = (result.get("metadatas") or [[]])[0]
    dists = (result.get("distances") or [[]])[0]
    for doc, meta, dist in zip(docs, metas, dists):
        meta = meta or {}
        hits.append(
            Hit(
                text=doc,
                source_file=str(meta.get("source_file", "unknown")),
                section=str(meta.get("section", "unknown")),
                course_number=str(meta.get("course_number", "unknown")),
                distance=float(dist),
            )
        )
    return hits


def print_retrieval(hits: list[Hit]) -> None:
    print("\n[RETRIEVAL RESULTS]")
    if not hits:
        print("  (no chunks returned)")
        return
    for i, hit in enumerate(hits, 1):
        print(
            f"  {i}. file={hit.source_file} | section={hit.section} | "
            f"course={hit.course_number} | distance={hit.distance:.4f}"
        )
        preview = " ".join(hit.text.split())
        print(f"     {preview[:220]}...")


def codes_in(text: str) -> set[str]:
    return {f"{a.upper()} {b}" for a, b in COURSE_CODE.findall(text)}


def prereq_line_from_hits(hits: list[Hit], course: str) -> str:
    course_key = course.upper()
    header_hit = None
    section_hit = None
    for hit in hits:
        if course_key and course_key not in hit.course_number.upper() and course_key not in hit.text.upper():
            continue
        for line in hit.text.splitlines():
            if line.upper().startswith("PREREQUISITES:"):
                return line.split(":", 1)[1].strip()
        if hit.section.lower() == "header":
            header_hit = hit
        if hit.section.lower() == "prerequisites":
            section_hit = hit
    if section_hit:
        # Strip the metadata lines we prepended during chunking.
        lines = [
            line
            for line in section_hit.text.splitlines()
            if not line.startswith("Course ")
            and not line.startswith("Section:")
            and not line.startswith("Source file:")
            and line.strip()
        ]
        return " ".join(lines)
    if header_hit:
        for line in header_hit.text.splitlines():
            if line.upper().startswith("PREREQUISITES:"):
                return line.split(":", 1)[1].strip()
    return ""


def grounded_enough(question: str, hits: list[Hit]) -> tuple[bool, str]:
    if not hits:
        return False, "the vector search returned no chunks"
    best = hits[0]
    if best.distance > MAX_DISTANCE:
        return (
            False,
            f"best chunk distance {best.distance:.3f} exceeds threshold {MAX_DISTANCE}",
        )
    q_codes = codes_in(question)
    hit_codes = set()
    for hit in hits:
        hit_codes |= codes_in(hit.text)
        hit_codes.add(hit.course_number.upper())
    advising_terms = {
        "course",
        "prerequisite",
        "prerequisites",
        "credit",
        "credits",
        "enroll",
        "enrollment",
        "take",
        "class",
        "corequisite",
        "restriction",
        "mac",
        "cop",
        "cen",
    }
    tokens = {t.lower() for t in re.findall(r"[A-Za-z0-9]+", question)}
    if q_codes and not (q_codes & {c.upper() for c in hit_codes}):
        # Question names a course we did not retrieve.
        missing = ", ".join(sorted(q_codes))
        return False, f"question mentions {missing}, but retrieval did not ground those codes"
    if not q_codes and tokens.isdisjoint(advising_terms):
        return False, "question is outside the course-catalog knowledge base"
    return True, "retrieval is close enough and on-topic"


def is_eligibility_question(question: str) -> bool:
    lower = question.lower()
    return bool(
        re.search(
            r"\b(can i take|am i eligible|may i enroll|have completed|but not|without completing)\b",
            lower,
        )
    )


def completed_courses_from_question(question: str) -> tuple[set[str], set[str]]:
    """Return (completed, missing) only from explicit completion language."""
    lower = question.lower()
    completed: set[str] = set()
    missing: set[str] = set()
    # "completed A and B but not C"
    parts = re.split(
        r"\bbut not\b|\bexcept\b|\bwithout\b|\bhaven'?t completed\b|\bnot completed\b",
        lower,
    )
    head = parts[0] if parts else lower
    tail = " ".join(parts[1:]) if len(parts) > 1 else ""
    if re.search(r"\b(completed|have taken|finished|passed)\b", head):
        # Keep codes after the completion verb, drop the course being asked about after "take".
        take_target = None
        take_match = re.search(r"\btake\s+([a-z]{3}\s*\d{4})", head)
        if take_match:
            take_target = next(iter(codes_in(take_match.group(0))), None)
        for code in codes_in(head):
            if take_target and code == take_target:
                continue
            completed.add(code)
    for code in codes_in(tail):
        missing.add(code)
    return completed, missing


def react_answer(collection: chromadb.Collection, question: str) -> str:
    print("\n" + "=" * 72)
    print("STUDENT QUESTION:")
    print(question)
    print("=" * 72)

    print("\n----- ReAct TRACE -----")
    print("Thought 1: I need course and prerequisite evidence from the knowledge base before I answer.")
    print(f"Action 1: search_courses(query={question!r}, top_k={TOP_K})")
    hits = retrieve(collection, question, TOP_K)
    print("Observation 1:")
    print_retrieval(hits)

    ok, reason = grounded_enough(question, hits)
    if not ok:
        print(
            f"Thought 2: The retrieval is not grounded enough ({reason}). "
            "I must refuse instead of inventing catalog rules."
        )
        print("Action 2: refuse")
        refusal = (
            "REFUSAL: I cannot answer from the course knowledge base. "
            f"Reason: {reason}. "
            "Ask a question about the courses in the uploaded catalog files "
            "(for example CEN 4930 prerequisites or COP 3530 credits)."
        )
        print("Final Answer:")
        print(refusal)
        print("----- END TRACE -----")
        return refusal

    print(
        "Thought 2: Retrieval looks usable. I will inspect the top chunks for "
        "prerequisites, credits, and restrictions, then check them against the student's stated courses."
    )

    target_codes = codes_in(question)
    # Prefer a 4xxx elective / the course after "take"
    target = None
    take_match = re.search(r"\btake\s+([A-Za-z]{3}\s*\d{4})", question, re.I)
    if take_match:
        target = " ".join(codes_in(take_match.group(0)))
    if not target and target_codes:
        # Highest course number often is the target.
        target = sorted(target_codes, key=lambda c: int(c.split()[1]), reverse=True)[0]

    if target:
        print(
            f"Thought 3: The question centers on {target}. "
            "I will retrieve that course's prerequisite section specifically."
        )
        print(f"Action 2: search_courses(query='{target} prerequisites enrollment restrictions')")
        more = retrieve(collection, f"{target} prerequisites enrollment restrictions", TOP_K)
        print("Observation 2:")
        print_retrieval(more)
        # Prefer hits for the target course.
        hits = more + [h for h in hits if h not in more]

    completed, missing = completed_courses_from_question(question)
    prereq_text = prereq_line_from_hits(hits, target or "")
    print(
        f"Thought 4: Student completed={sorted(completed) or 'unspecified'}; "
        f"explicitly missing={sorted(missing) or 'unspecified'}; "
        f"catalog prerequisite text={prereq_text or 'not found in retrieved chunks'}."
    )

    best = hits[0]
    citation = f"{best.source_file} / section '{best.section}'"
    # Also cite a prerequisites section if present.
    for hit in hits:
        if hit.section.lower() == "prerequisites" and (
            not target or target.upper() in hit.course_number.upper() or target.upper() in hit.text.upper()
        ):
            citation = f"{hit.source_file} / section '{hit.section}'"
            best = hit
            break

    if target and is_eligibility_question(question) and prereq_text:
        # Read the short PREREQUISITES: fact line from the course file when possible.
        header_prereq = ""
        target_key = target.replace(" ", "").upper()
        for path in discover_course_files():
            stem_key = path.stem.replace("course_", "").replace("_Stilwell", "").replace("_", "").upper()
            if stem_key != target_key and target.upper() not in path.read_text(encoding="utf-8")[:200].upper():
                continue
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.upper().startswith("PREREQUISITES:"):
                    header_prereq = line.split(":", 1)[1].strip()
                    break
            if header_prereq:
                break
        clean_prereq = header_prereq or prereq_line_from_hits(hits, target) or prereq_text
        required = codes_in(header_prereq) if header_prereq else codes_in(clean_prereq)
        required.discard(target.upper())
        if not required and "none" in clean_prereq.lower():
            required = set()
        if completed or missing:
            blocked = sorted((missing & required) | (required - completed))
        else:
            blocked = []

        if blocked:
            print(
                "Thought 5: At least one required prerequisite is missing. "
                "Changing the answer from a possible yes to NO."
            )
            answer = (
                f"No. Based on the catalog, {target} requires {clean_prereq}. "
                f"You are missing {', '.join(blocked)}, so you are not cleared to take {target} yet. "
                f"Source: {citation}."
            )
        elif completed and required and required.issubset(completed):
            print(
                "Thought 5: Every listed prerequisite appears in the completed set. "
                "Changing the answer to a conditional YES grounded in the catalog."
            )
            answer = (
                f"Yes, based on the retrieved catalog text, {target} requires {prereq_text}, "
                f"and your completed courses cover that list. Still confirm holds and term offering "
                f"with an advisor. Source: {citation}."
            )
        else:
            print(
                "Thought 5: I can cite the catalog requirement, but the question does not fully "
                "map every prerequisite to a completed course. I will state the requirement, not invent eligibility."
            )
            answer = (
                f"According to the knowledge base, {target} lists prerequisites as: {prereq_text}. "
                f"I cannot certify enrollment without a complete completed-course list that covers each one. "
                f"Source: {citation}."
            )
    else:
        # Informational grounded Q&A: pick the best section for the ask.
        print("Thought 5: This is an informational catalog question; summarizing the best grounded section.")
        chosen = best
        lower_q = question.lower()
        wanted = "header"
        if "prerequisite" in lower_q:
            wanted = "prerequisites"
        elif "credit" in lower_q:
            wanted = "credits"
        elif "restrict" in lower_q or "enroll" in lower_q:
            wanted = "enrollment restrictions"
        elif "description" in lower_q or "about" in lower_q:
            wanted = "description"
        for hit in hits:
            if hit.section.lower() == wanted and (
                not target or target.upper() in hit.course_number.upper()
            ):
                chosen = hit
                break
        citation = f"{chosen.source_file} / section '{chosen.section}'"
        # Prefer a clean CREDITS / PREREQUISITES line from the header when useful.
        if wanted == "credits":
            for hit in hits:
                if target and target.upper() in hit.course_number.upper():
                    for line in hit.text.splitlines():
                        if line.upper().startswith("CREDITS:"):
                            answer = (
                                f"{target} is {line.split(':', 1)[1].strip()} credit hours "
                                f"according to the catalog. Source: {hit.source_file} / section '{hit.section}'."
                            )
                            break
                    else:
                        continue
                    break
            else:
                answer = f"{' '.join(chosen.text.split())[:500]} Source: {citation}."
        elif wanted == "prerequisites":
            clean = prereq_line_from_hits(hits, target or "")
            for hit in hits:
                for line in hit.text.splitlines():
                    if line.upper().startswith("PREREQUISITES:"):
                        clean = line.split(":", 1)[1].strip()
                        citation = f"{hit.source_file} / section '{hit.section}'"
            answer = (
                f"Prerequisites for {target or 'this course'}: {clean or 'see cited section'}. "
                f"Source: {citation}."
            )
        else:
            answer = f"{' '.join(chosen.text.split())[:500]} Source: {citation}."

    # If answer somehow has no source, refuse.
    if "Source:" not in answer:
        answer = (
            "REFUSAL: I retrieved catalog text but could not form a citable answer. "
            "I will not guess."
        )

    print("Final Answer:")
    print(answer)
    print("----- END TRACE -----")
    return answer


def maybe_run_agent_layer(collection: chromadb.Collection, question: str, draft: str) -> None:
    """Optional Agents SDK pass when OPENAI_API_KEY is present."""
    if not os.getenv("OPENAI_API_KEY"):
        print("\n[MODEL] OPENAI_API_KEY not set — using the printed ReAct controller answer.")
        return
    try:
        from agents import Agent, Runner, function_tool, set_tracing_disabled

        set_tracing_disabled(True)

        @function_tool
        def search_courses(query: str) -> str:
            """Semantic search over the course knowledge base."""
            hits = retrieve(collection, query, TOP_K)
            print_retrieval(hits)
            lines = []
            for hit in hits:
                lines.append(
                    f"[{hit.source_file} | {hit.section} | distance={hit.distance:.3f}]\n{hit.text}"
                )
            return "\n\n".join(lines) if lines else "No relevant documents found."

        agent = Agent(
            name="Advising ReAct Agent",
            instructions=(
                "You are a university catalog advising agent. "
                "Use search_courses before answering. "
                "Think step by step in a ReAct style. "
                "Cite source file and section. "
                "If evidence is missing, refuse."
            ),
            tools=[search_courses],
            model=os.getenv("OPENAI_DEFAULT_MODEL", "gpt-4.1"),
        )
        print("\n[MODEL] Running optional Agents SDK ReAct layer...")
        result = Runner.run_sync(agent, question)
        print("[MODEL FINAL]")
        print(result.final_output)
        print("[CONTROLLER FINAL was]")
        print(draft)
    except Exception as exc:
        print(f"[MODEL] optional agent layer skipped: {type(exc).__name__}: {exc}")


def main() -> None:
    print("Part A — RAG + ReAct course advising agent")
    print(f"knowledge base directory: {HERE}")
    collection = build_index(rebuild=True)

    for question in TEST_CASES:
        draft = react_answer(collection, question)
        # Only spend an API call on the primary in-scope scenario when a key exists.
        if question == TEST_CASES[0]:
            maybe_run_agent_layer(collection, question, draft)


TEST_CASES = [
    "Can I take CEN 4930 next semester if I have completed CEN 3031 and COP 3530 but not MAC 2312?",
    "What are the prerequisites for COP 3530?",
    "How many credit hours is MAC 2312?",
    "What time does the campus pizza dining hall close on Friday?",
    "Who won the Super Bowl last year and what was the score?",
]


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------------
# Test inputs exercised above. At least two are out of scope and must refuse.
#
# 1. CEN 4930 with CEN 3031 + COP 3530 but not MAC 2312 -> grounded NO + citation
# 2. Prerequisites for COP 3530 -> grounded answer + citation
# 3. Credit hours for MAC 2312 -> grounded answer + citation
# 4. Pizza dining hall hours -> REFUSAL (not in KB)
# 5. Super Bowl question -> REFUSAL (not in KB)
# ---------------------------------------------------------------------------
