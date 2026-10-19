"""Build the Part A diagram and A2_Stilwell.pdf. Not part of the Canvas run."""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import Image as RLImage
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

HERE = Path(__file__).resolve().parent
DIAGRAM = HERE / "architecture_partA_Stilwell.png"
PDF = HERE / "A2_Stilwell.pdf"

SANS = "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"
SANS_BOLD = "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"

SECTIONS = {
    "1": """
I built the policy tool because the document is a local file I wrote. Gulf Cypress University is fictional, so no hosted MCP server can return CMS-4, DAR-1, and ACN-2. Putting those names in a prompt would not make them a tool. Writing the server meant choosing stdio, the tool schema, and what happens when the process dies. A pre-built server would have hidden those choices.

lookup_major_change_policy takes one string, query. The Policy Agent already holds a JSON handoff, so it can ask in one sentence for the process, deadline, forms, GPA minimum, and prerequisites. The tool always returns a FACTS block from the labeled lines at the top of policy_doc_Stilwell.txt, then up to three narrative sections scored against the query. The facts are the contract. A weak query can miss the Deadlines heading and still cannot drop the deadline, because those lines are not scored. I called the tool through the Agents SDK over stdio. The Policy Agent's filtered list contains only that tool, even though the same process also serves the Part B hazard tool.

If connect() fails, the script prints MCP OFFLINE and the handoff says not to invent form numbers. The advisor can still write a plan, but the policy section has to say the deadline was not confirmed. The record check does not use the server, so a dead process cannot also erase the GPA.
""".strip(),
    "2": """
The transcript and the registrar's rule are different kinds of claims. The Eligibility Agent may read only student_record_Stilwell.txt and checks GPA, posted prerequisites, standing, and holds. The Policy Agent may call only the MCP tool and does not recompute the GPA. The Advisor Agent is the only one that speaks to the student, and the output has to land on four headings. If the first two jobs shared an agent, a number from the advising file could be repeated as if the registrar had published it.

The first handoff is JSON. The eligibility object carries the name, cumulative GPA, prerequisite check, status, and the note that MAC 2311 is still in progress. JSON keeps 3.41 from being softened to "about a 3.4." The second handoff is JSON too: eligibility_json and policy_text side by side, plus the original question. The advisor has to know which lines came from the file and which came from the server.

On "Switch major now", the Eligibility Agent's guardrail trips before any tool call. The student hears: "Please send a full advising question of at least five words about a major, a course plan, or an academic policy. I cannot start an eligibility check from a fragment." A longer pizza question trips the same guardrail because it is outside advising, business, and technology. A plan with blank sections trips the output guardrail. A plan built from the record does not. The advisor retries once, then a second thin plan is flagged.
""".strip(),
    "3": """
I paddle at Bonita Beach before a 9:30 class in Holmes Hall. The car has to be parked by 9:10, my shoulder has been sore, and the tower is empty until 8:00. I wanted a go or no-go with a time on it.

The Session Log Agent only reads session_log_Stilwell.txt: out of the water at 7:35, no overhead takeoffs, no buddy. The Hazard Agent only calls lookup_beach_hazards, and the server is filtered so it cannot see the university policy tool. The Skipper Agent calls build_dawn_patrol_call and cannot invent a new exit time. Drop the hazard agent and the personal log can ignore a yellow flag. Drop the log agent and a beach note can ignore the class.

The choice that differs most from Part A is the answer. Part A must produce four sections and retries when they are thin. Part B returns GO or NO-GO, and the two tests are supposed to disagree. Waiting for the lifeguard at 8:00 fails the 45-minute rule once the drive is included, so that question is NO-GO even though the early session is GO. The question has to ride in the handoff or the skipper cannot tell those cases apart.
""".strip(),
    "4": """
The course samples import FastMCP from mcp.server.fastmcp. On mcp 2 that import raises ModuleNotFoundError and says the class moved to MCPServer. The server tries FastMCP, then falls back. After that, a stdio call to lookup_major_change_policy returned the CMS-4 lines. I also called my output-guardrail decorator directly. It is not a function. The run died with TypeError: 'OutputGuardrail' object is not callable, and the finally block stopped the policy server. I call OutputGuardrail.run now. The thin plan trips. The full plan does not.

The model never finished a handoff here. After the guardrail allowed the Computer Science question, Runner.run entered the Eligibility Agent and raised OpenAIError: Missing credentials. I did not fake a completion. I also did not ask the model to hand off by obeying "Always hand off." The script passes a JSON payload into the next run, and the synthesis tool is closed over that payload.
""".strip(),
}

DISCLOSURE = (
    "I used Cursor's cloud agent, running Grok 4.7, to draft the Python files, the fictional "
    "data files, the architecture diagram, and this report. I kept the Agents SDK patterns from "
    "the chapter 3 and chapter 4 samples in the course repository. The eligibility rules, the "
    "policy text, and the dawn-patrol scenario were written for this submission. I checked them "
    "by running the guardrails, the file parsers, and a live stdio call into "
    "mcp_server_Stilwell.py. No live model completion was available here because OPENAI_API_KEY "
    "was not set."
)


def _font(path: str, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(path, size)


def _wrap(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont, width: int) -> list[str]:
    lines: list[str] = []
    for paragraph in text.split("\n"):
        words = paragraph.split()
        current = ""
        for word in words:
            trial = word if not current else f"{current} {word}"
            if draw.textlength(trial, font=font) <= width:
                current = trial
            else:
                if current:
                    lines.append(current)
                current = word
        lines.append(current)
    return lines


def _box(draw, xy, fill, outline, title, body, title_font, body_font, title_fill="white"):
    x0, y0, x1, y1 = xy
    draw.rounded_rectangle(xy, radius=16, fill=fill, outline=outline, width=3)
    draw.text((x0 + 18, y0 + 12), title, font=title_font, fill=title_fill)
    y = y0 + 44
    for line in _wrap(draw, body, body_font, (x1 - x0) - 36):
        draw.text((x0 + 18, y), line, font=body_font, fill="#1c2430")
        y += 22
    return y1


def _arrow(draw, x, y0, y1):
    draw.line((x, y0, x, y1 - 8), fill="#334155", width=4)
    draw.polygon([(x - 8, y1 - 12), (x + 8, y1 - 12), (x, y1)], fill="#334155")


def draw_diagram(path: Path) -> None:
    width, height = 1600, 1640
    image = Image.new("RGB", (width, height), "#f7f4ee")
    draw = ImageDraw.Draw(image)
    title_font = _font(SANS_BOLD, 34)
    head_font = _font(SANS_BOLD, 22)
    body_font = _font(SANS, 18)
    small_bold = _font(SANS_BOLD, 18)
    small = _font(SANS, 17)
    draw.text((48, 28), "Part A — major-change advising architecture", font=title_font, fill="#1e293b")
    draw.text(
        (48, 74),
        "Sequential handoffs. Each agent has one tool. Guardrails sit on the entry and the exit.",
        font=body_font,
        fill="#475569",
    )

    blocks = [
        (120, 100, "#e2e8f0", "#64748b", "#0f172a", "Student question",
         "Plain text. The scenario question is a Computer Science major change."),
        (250, 130, "#fef3c7", "#b45309", "#92400e", "INPUT GUARDRAIL — on the Eligibility Agent",
         "Rejects fewer than 5 words, or a question outside advising, business, or technology. A trip stops the pipeline before any tool call."),
        (410, 140, "#dbeafe", "#1d4ed8", "#1e3a8a", "1. Eligibility Agent",
         "Role: read the academic record and check GPA, posted prerequisites, standing, and holds. Does not quote registrar policy."),
        (580, 120, "#f1f5f9", "#334155", "#0f172a", "HANDOFF 1 — Eligibility Agent → Policy Agent",
         "Format: JSON. Field eligibility carries name, GPA, courses, prerequisite_check, status, and the MAC 2311 note."),
        (730, 140, "#dbeafe", "#1d4ed8", "#1e3a8a", "2. Policy Agent",
         "Role: retrieve the official process, deadline, forms, GPA rule, and prerequisites. Does not recompute the student's GPA."),
        (900, 130, "#f1f5f9", "#334155", "#0f172a", "HANDOFF 2 — Policy Agent → Advisor Agent",
         "Format: JSON. Fields eligibility_json and policy_text sit side by side with the original student question."),
        (1060, 120, "#fef3c7", "#b45309", "#92400e", "OUTPUT GUARDRAIL — on the Advisor Agent",
         "Requires all four sections, each at least 80 characters. A thin plan retries once, then the failure is flagged."),
        (1210, 140, "#dbeafe", "#1d4ed8", "#1e3a8a", "3. Advisor Agent",
         "Role: synthesize a personal action plan. Copies the synthesis tool into the four required sections."),
        (1380, 150, "#ecfdf5", "#047857", "#065f46", "Final output",
         "Student Summary / Eligibility Status / Policy Requirements / Recommended Next Steps"),
    ]
    # recompute arrows between known bottoms
    prev_bottom = None
    for top, tall, fill, outline, title_fill, title, body in blocks:
        _box(draw, (70, top, 980, top + tall), fill, outline, title, body, head_font, body_font, title_fill)
        if prev_bottom is not None:
            _arrow(draw, 525, prev_bottom, top)
        prev_bottom = top + tall

    tools = [
        (410, 140, "Tool: read_student_record", "Local function tool.\nServes: this script, not MCP.\nFile: student_record_Stilwell.txt"),
        (730, 140, "Tool: lookup_major_change_policy", "Custom MCP server.\nScript: mcp_server_Stilwell.py\nTransport: stdio. File: policy_doc_Stilwell.txt"),
        (1210, 140, "Tool: synthesize_action_plan", "Local function tool.\nServes: this script, not MCP.\nClosed over handoff 2."),
    ]
    for top, tall, title, body in tools:
        _box(draw, (1020, top, 1540, top + tall), "#ecfdf5", "#047857", title, body, small_bold, small, "#065f46")
        draw.line((980, top + tall // 2, 1020, top + tall // 2), fill="#047857", width=3)

    draw.text(
        (70, 1565),
        "Servers: the record tool and the synthesis tool are local. Only the policy tool is served by mcp_server_Stilwell.py.",
        font=small,
        fill="#334155",
    )
    image.save(path)


def _count(text: str) -> int:
    return len(text.split())


def build_pdf(path: Path, diagram: Path) -> None:
    styles = {
        "h1": ParagraphStyle(
            "h1", fontName="Times-Bold", fontSize=16, leading=20, spaceAfter=8, textColor="#1e293b",
        ),
        "h2": ParagraphStyle(
            "h2", fontName="Times-Bold", fontSize=13, leading=16, spaceBefore=12, spaceAfter=6, textColor="#1e3a8a",
        ),
        "body": ParagraphStyle(
            "body", fontName="Times-Roman", fontSize=11, leading=15, spaceAfter=8, textColor="#111827",
        ),
        "caption": ParagraphStyle(
            "caption", fontName="Times-Italic", fontSize=10, leading=13, spaceBefore=4, spaceAfter=8, textColor="#475569",
        ),
    }
    doc = SimpleDocTemplate(
        str(path),
        pagesize=letter,
        leftMargin=0.8 * inch,
        rightMargin=0.8 * inch,
        topMargin=0.7 * inch,
        bottomMargin=0.7 * inch,
        title="A2 Stilwell — Multi-Agent and MCP",
        author="Nolan Stilwell",
    )
    story = [
        Paragraph("CEN 4930 Assignment 2 — Multi-Agent and MCP Integration", styles["h1"]),
        Paragraph("Nolan Stilwell", styles["body"]),
        Paragraph("Section 1: Part A — MCP Design Rationale", styles["h2"]),
    ]
    for paragraph in SECTIONS["1"].split("\n\n"):
        story.append(Paragraph(paragraph.replace("\n", " "), styles["body"]))
    story.append(Paragraph("Section 2: Part A — Multi-Agent Architecture Rationale", styles["h2"]))
    for paragraph in SECTIONS["2"].split("\n\n"):
        story.append(Paragraph(paragraph.replace("\n", " "), styles["body"]))
    story.append(Spacer(1, 8))
    story.append(RLImage(str(diagram), width=7.0 * inch, height=7.0 * inch * (1640 / 1600)))
    story.append(Paragraph(
        "Figure 1. Part A flow. Input guardrail, three agents, two JSON handoffs, output guardrail, and the server for each tool.",
        styles["caption"],
    ))
    story.append(Paragraph("Section 3: Part B — Scenario Rationale", styles["h2"]))
    for paragraph in SECTIONS["3"].split("\n\n"):
        story.append(Paragraph(paragraph.replace("\n", " "), styles["body"]))
    story.append(Paragraph("Section 4: What Broke + Model Stack Note", styles["h2"]))
    for paragraph in SECTIONS["4"].split("\n\n"):
        story.append(Paragraph(paragraph.replace("\n", " "), styles["body"]))
    story.append(Paragraph("AI Disclosure", styles["h2"]))
    story.append(Paragraph(DISCLOSURE, styles["body"]))
    doc.build(story)


def main() -> None:
    counts = {key: _count(value) for key, value in SECTIONS.items()}
    counts["disclosure"] = _count(DISCLOSURE)
    counts["total"] = sum(counts.values())
    print(counts)
    if not 700 <= counts["total"] <= 1000:
        raise SystemExit(f"word count {counts['total']} is outside 700-1000")
    for key, low, high in (("1", 200, 250), ("2", 200, 250), ("3", 150, 200), ("4", 100, 150)):
        if not low <= counts[key] <= high:
            raise SystemExit(f"section {key} has {counts[key]} words, expected {low}-{high}")
    draw_diagram(DIAGRAM)
    build_pdf(PDF, DIAGRAM)
    print(f"wrote {DIAGRAM}")
    print(f"wrote {PDF}")


if __name__ == "__main__":
    main()
