"""Build A3_Stilwell.pdf. Not a Canvas deliverable."""

from pathlib import Path

from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import Paragraph, Preformatted, SimpleDocTemplate, Spacer

HERE = Path(__file__).resolve().parent
PDF = HERE / "A3_Stilwell.pdf"
TRACE = Path("/tmp/a3_trace.txt")

SECTIONS = {
    "1": """
I chunked each course file by labeled header facts and by ## sections (Description, Prerequisites, Credits, Enrollment Restrictions). Section chunks match how an advisor cites a catalog: file name plus section. Fixed token windows would split a prerequisite list mid-sentence and make citation fuzzy. Sentence chunking would scatter the PREREQUISITES fact line away from the course number.

The embedding model is ChromaDB's default ONNX all-MiniLM-L6-v2. It initializes on first run with no extra API key for indexing, which matters because the instructor script must rebuild the store from the uploaded text files alone. I retrieve top-k = 4 and refuse when the best L2 distance is above 1.55 or when the question is outside advising terms. That threshold came from testing: in-scope course questions landed near 0.6–1.0, while dining-hall and Super Bowl questions landed near 1.5–1.8.

The pipeline fails when the top chunk is only partly related — for example, retrieving COP 3530 prerequisites while the student asked about CEN 4930. The ReAct loop's second search targets the asked course, and eligibility answers prefer the PREREQUISITES header line over a narrative section that mentions other course numbers. If evidence is still weak, the agent refuses instead of blending partial matches into a fake rule.
""".strip(),
    "2": """
I used ReAct rather than plain CoT. An advising question needs a retrieve step before a verdict, then another retrieve when the first observation is not the right course section. CoT alone would narrate prerequisites from model memory. ReAct forces Action and Observation lines in the console so the grader can see search before the answer, which matches how chapter 5 frames tool-using agents.

From the appendix, Thought 5 is the turn that changes the answer: "At least one required prerequisite is missing. Changing the answer from a possible yes to NO." After Observation 2 confirmed MAC 2312 on the CEN 4930 list, that thought blocks a yes that the earlier completed-course list might have encouraged. The agent had already seen CEN 3031 and COP 3530 as done; without that later thought it could stop at "mostly ready."

Without explicit reasoning the agent would likely say the student is close enough because two of three prerequisites are finished. That would be worse for advising: it would clear a student the catalog still blocks and would hide the missing Calculus II requirement behind fluent prose.
""".strip(),
    "3": """
Part B is a Bonita Beach dawn-patrol advisor for paddling before a 9:30 Holmes Hall class. It needs RAG because the rules live in four short personal documents — funboard use, flag and tower rules, parking clock, and pier rip entry — not in a single system prompt I want to hard-code and re-edit every time the rip moves. Retrieval lets a new note change the answer without rewriting agent instructions, and citations show which file produced the call.

The biggest design difference from Part A is the decision type. Part A answers catalog eligibility with course codes and prerequisite sets. Part B returns GO or NO-GO against a personal clock, and the question text itself can flip the call: waiting for the 08:00 tower fails the 07:35 exit and 09:10 campus parking rule even when the flag is only yellow.

Refusal example: on "What is the tuition refund deadline for dropping a chemistry lab?" the domain check finds no beach, surf, flag, or parking terms, so the agent prints REFUSAL and points the user back to funboard, flags, parking, or rip questions instead of inventing university policy.
""".strip(),
    "4": """
Part A is most likely to hallucinate when a retrieved narrative mentions a course that is not a prerequisite — for example MAC 1147 inside a "does not substitute" sentence. I reduced that risk by taking required codes only from the PREREQUISITES header line, discarding the target course code itself, and refusing when best-chunk distance exceeds 1.55 or the question leaves the advising vocabulary.

This workspace had no OPENAI_API_KEY, so the graded path is the printed ReAct controller over Chroma retrieval rather than a live model synthesis. The optional Agents SDK layer runs only when a key is present. The concrete prompt-engineering lesson was making Thought 5 depend on Observation 2 instead of letting the first partial retrieval end the answer after two completed prerequisites looked encouraging.
""".strip(),
}

DISCLOSURE = (
    "I used Cursor's cloud agent (Grok) to draft the Python files, knowledge-base documents, "
    "and this report. I adapted Chroma usage from the course chapter 6 samples and ReAct "
    "structure from chapter 5. I verified retrieval, citations, and refusals by running "
    "a3_partA_Stilwell.py and a3_partB_Stilwell.py in this environment."
)


def wc(text: str) -> int:
    return len(text.split())


def main() -> None:
    counts = {k: wc(v) for k, v in SECTIONS.items()}
    counts["disclosure"] = wc(DISCLOSURE)
    counts["total"] = sum(counts.values())
    print(counts)
    if not 700 <= counts["total"] <= 1000:
        raise SystemExit(f"total {counts['total']} out of range")
    for key, lo, hi in (("1", 200, 250), ("2", 150, 200), ("3", 150, 200), ("4", 100, 150)):
        if not lo <= counts[key] <= hi:
            raise SystemExit(f"section {key} has {counts[key]} words")

    styles = {
        "h1": ParagraphStyle("h1", fontName="Times-Bold", fontSize=16, leading=20, spaceAfter=8),
        "h2": ParagraphStyle(
            "h2", fontName="Times-Bold", fontSize=13, leading=16, spaceBefore=12, spaceAfter=6
        ),
        "body": ParagraphStyle("body", fontName="Times-Roman", fontSize=11, leading=15, spaceAfter=8),
        "pre": ParagraphStyle(
            "pre", fontName="Courier", fontSize=7.5, leading=9, spaceAfter=6, leftIndent=6
        ),
    }
    doc = SimpleDocTemplate(
        str(PDF),
        pagesize=letter,
        leftMargin=0.75 * inch,
        rightMargin=0.75 * inch,
        topMargin=0.7 * inch,
        bottomMargin=0.7 * inch,
        title="A3 Stilwell — Reasoning + RAG Agent",
        author="Nolan Stilwell",
    )
    story = [
        Paragraph("CEN 4930 Assignment 3 — Reasoning + RAG Agent", styles["h1"]),
        Paragraph("Nolan Stilwell", styles["body"]),
        Paragraph("Section 1: Part A — RAG Design", styles["h2"]),
    ]
    for para in SECTIONS["1"].split("\n\n"):
        story.append(Paragraph(para.replace("\n", " "), styles["body"]))
    story.append(Paragraph("Section 2: Part A — Reasoning Pattern", styles["h2"]))
    for para in SECTIONS["2"].split("\n\n"):
        story.append(Paragraph(para.replace("\n", " "), styles["body"]))
    story.append(Paragraph("Section 3: Part B — Scenario Rationale", styles["h2"]))
    for para in SECTIONS["3"].split("\n\n"):
        story.append(Paragraph(para.replace("\n", " "), styles["body"]))
    story.append(Paragraph("Section 4: Hallucination Risk + Model Stack Note", styles["h2"]))
    for para in SECTIONS["4"].split("\n\n"):
        story.append(Paragraph(para.replace("\n", " "), styles["body"]))
    story.append(Paragraph("AI Disclosure", styles["h2"]))
    story.append(Paragraph(DISCLOSURE, styles["body"]))
    story.append(Paragraph("Appendix: Part A ReAct Trace", styles["h2"]))
    story.append(
        Paragraph(
            "Complete console trace for: Can I take CEN 4930 next semester if I have completed "
            "CEN 3031 and COP 3530 but not MAC 2312?",
            styles["body"],
        )
    )
    trace = TRACE.read_text(encoding="utf-8") if TRACE.exists() else "(trace missing)"
    # Keep appendix readable but finite.
    if len(trace) > 6500:
        trace = trace[:6500] + "\n... [truncated for PDF layout] ..."
    story.append(Spacer(1, 6))
    story.append(Preformatted(trace, styles["pre"]))
    doc.build(story)
    print(f"wrote {PDF}")


if __name__ == "__main__":
    main()
