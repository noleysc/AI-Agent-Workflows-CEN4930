"""CEN 4930 Assignment 2, Part A — multi-agent major-change advising.

Run from any working directory:

    python a2_partA_Stilwell.py

Three agents run in sequence. Each one has its own tool. The policy tool is
served by mcp_server_Stilwell.py over stdio. Handoffs are JSON payloads
printed between agents. An input guardrail stops the pipeline before the
Eligibility Agent calls a model. An output guardrail checks the Advisor
Agent's four-part plan and retries once when the structure is thin.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import sys
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict

from agents import (
    Agent,
    GuardrailFunctionOutput,
    InputGuardrailTripwireTriggered,
    ModelSettings,
    OutputGuardrailTripwireTriggered,
    RunContextWrapper,
    RunHooks,
    Runner,
    TResponseInputItem,
    function_tool,
    input_guardrail,
    output_guardrail,
    set_tracing_disabled,
)
from agents.mcp import MCPServerStdio, MCPServerStdioParams, create_static_tool_filter

HERE = Path(__file__).resolve().parent
STUDENT_RECORD = HERE / "student_record_Stilwell.txt"
SERVER_SCRIPT = HERE / "mcp_server_Stilwell.py"
MODEL = os.getenv("OPENAI_DEFAULT_MODEL", "gpt-4.1")
MIN_WORDS = 5
SECTION_MIN_CHARS = 80

# Coarse entry filter from the assignment: academic advising for a business
# or technology program. Matching is on whole words so "class" inside another
# phrase cannot sneak a query through.
DOMAIN_TERMS = {
    "academic",
    "advising",
    "advisor",
    "business",
    "catalog",
    "college",
    "computer",
    "course",
    "courses",
    "credit",
    "credits",
    "cs",
    "degree",
    "gpa",
    "grade",
    "grades",
    "graduation",
    "major",
    "minor",
    "prerequisite",
    "prerequisites",
    "prereq",
    "program",
    "registrar",
    "research",
    "science",
    "semester",
    "tech",
    "technology",
    "transcript",
    "transfer",
    "university",
}

GRADE_POINTS = {
    "A": 4.0,
    "A-": 3.7,
    "B+": 3.3,
    "B": 3.0,
    "B-": 2.7,
    "C+": 2.3,
    "C": 2.0,
    "C-": 1.7,
    "D": 1.0,
    "F": 0.0,
}

load_dotenv()
load_dotenv(HERE.parent / ".env")
set_tracing_disabled(True)


class AdvisingPlan(BaseModel):
    """Final plan. The console print adds the four required headings."""

    model_config = ConfigDict(extra="forbid")

    student_summary: str
    eligibility_status: str
    policy_requirements: str
    recommended_next_steps: str


class ConsoleHooks(RunHooks):
    """Print every tool call the SDK actually invokes."""

    def __init__(self) -> None:
        self.tool_results: list[tuple[str, str]] = []

    async def on_agent_start(self, context, agent) -> None:
        print(f"\n----- AGENT: {agent.name} -----")

    async def on_tool_start(self, context, agent, tool) -> None:
        arguments = getattr(context, "tool_arguments", "") or ""
        print(f"[TOOL CALL] {agent.name} -> {tool.name}")
        if arguments:
            print(f"[TOOL ARGS] {arguments}")

    async def on_tool_end(self, context, agent, tool, result) -> None:
        text = result if isinstance(result, str) else str(result)
        self.tool_results.append((tool.name, text))
        print(f"[TOOL RESULT] {tool.name}")
        print(text)


def _words(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9']+", text)


def _as_text(value: str | list[TResponseInputItem]) -> str:
    if isinstance(value, str):
        return value
    parts: list[str] = []
    for item in value:
        if isinstance(item, dict):
            content = item.get("content", "")
            parts.append(content if isinstance(content, str) else str(content))
        else:
            parts.append(str(item))
    return "\n".join(parts)


def screen_student_query(query: str) -> tuple[bool, str, str]:
    """Return (allowed, reason, message the student should see)."""
    words = _words(query)
    if len(words) < MIN_WORDS:
        return (
            False,
            f"query has {len(words)} words (minimum is {MIN_WORDS})",
            "Please send a full advising question of at least five words about a "
            "major, a course plan, or an academic policy. I cannot start an "
            "eligibility check from a fragment.",
        )
    tokens = {word.lower() for word in words}
    if tokens.isdisjoint(DOMAIN_TERMS):
        return (
            False,
            "query is not about academic advising, business, or technology",
            "I can only take questions about university advising for business or "
            "technology programs, including major changes, prerequisites, and "
            "policy. Please rephrase the request around that topic.",
        )
    return True, f"query has {len(words)} words and is in scope", ""


@input_guardrail(run_in_parallel=False)
async def eligibility_input_guardrail(
    ctx: RunContextWrapper[None],
    agent: Agent,
    incoming: str | list[TResponseInputItem],
) -> GuardrailFunctionOutput:
    allowed, reason, message = screen_student_query(_as_text(incoming))
    print(f"[INPUT GUARDRAIL] Eligibility Agent checked the query ({reason})")
    if not allowed:
        print("[INPUT GUARDRAIL] TRIPWIRE — pipeline stopped before any tool call")
        print(f"[INPUT GUARDRAIL] student message: {message}")
    return GuardrailFunctionOutput(
        output_info={"reason": reason, "student_message": message},
        tripwire_triggered=not allowed,
    )


def _missing_sections(plan: AdvisingPlan) -> list[str]:
    missing = []
    for name in (
        "student_summary",
        "eligibility_status",
        "policy_requirements",
        "recommended_next_steps",
    ):
        if len(getattr(plan, name).strip()) < SECTION_MIN_CHARS:
            missing.append(name)
    return missing


async def check_advising_output(
    ctx: RunContextWrapper,
    agent: Agent,
    output: AdvisingPlan,
) -> GuardrailFunctionOutput:
    missing = _missing_sections(output)
    if missing:
        print(f"[OUTPUT GUARDRAIL] TRIPWIRE — thin or missing sections: {missing}")
    else:
        print("[OUTPUT GUARDRAIL] four sections are present and long enough")
    return GuardrailFunctionOutput(
        output_info={"thin_sections": missing, "minimum_chars": SECTION_MIN_CHARS},
        tripwire_triggered=bool(missing),
    )


advising_output_guardrail = output_guardrail(check_advising_output)


def _record_lines(text: str) -> dict[str, str]:
    facts: dict[str, str] = {}
    for line in text.splitlines():
        if line.startswith("## "):
            break
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        if key.isupper():
            facts[key.strip()] = value.strip()
    return facts


def _grade_meets(grade: str, minimum: str) -> bool:
    return GRADE_POINTS.get(grade, -1) >= GRADE_POINTS.get(minimum, 2.0)


def read_student_record_text(intended_major: str) -> str:
    """Read the advising file and check it against the checklist on that file."""
    if not STUDENT_RECORD.is_file():
        return json.dumps({"error": f"missing {STUDENT_RECORD.name}"})
    facts = _record_lines(STUDENT_RECORD.read_text(encoding="utf-8"))
    completed = {}
    for item in facts.get("COMPLETED", "").split(","):
        item = item.strip()
        if "|" not in item:
            continue
        course, grade = item.split("|", 1)
        completed[course.strip()] = grade.strip()
    in_progress = [item.strip() for item in facts.get("IN_PROGRESS", "").split(",") if item.strip()]

    asked = intended_major.strip() or "unspecified"
    checklist_major = facts.get("INTENDED_MAJOR", "")
    asked_key = asked.lower()
    cs_aliases = ("computer science", "computer-science", "comp sci", "cs", "computing")
    major_is_cs = any(
        alias == asked_key or f" {alias} " in f" {asked_key} " or asked_key.startswith(alias + " ")
        for alias in cs_aliases
    ) or "computer science" in asked_key
    gpa = float(facts.get("CUMULATIVE_GPA", "0") or 0)
    minimum = float(facts.get("CHECKLIST_GPA_MINIMUM", "3") or 3)
    holds = facts.get("HOLDS", "unknown")
    standing = facts.get("ACADEMIC_STANDING", "unknown")

    prereq_groups: list[list[tuple[str, str]]] = []
    for group in facts.get("CHECKLIST_PREREQS", "").split(","):
        options = []
        for option in group.split(" or "):
            option = option.strip()
            if "|" not in option:
                continue
            course, minimum_grade = option.split("|", 1)
            options.append((course.strip(), minimum_grade.strip()))
        if options:
            prereq_groups.append(options)

    prereq_report = []
    prereqs_met = True
    for options in prereq_groups:
        met = False
        detail = []
        for course, minimum_grade in options:
            grade = completed.get(course)
            if grade and _grade_meets(grade, minimum_grade):
                met = True
                detail.append(f"{course} {grade} meets {minimum_grade}")
            elif course in in_progress:
                detail.append(f"{course} is in progress and does not count yet")
            elif grade:
                detail.append(f"{course} {grade} is below {minimum_grade}")
            else:
                detail.append(f"{course} is not completed")
        prereqs_met = prereqs_met and met
        prereq_report.append({"options": [f"{c} ({g})" for c, g in options], "met": met, "detail": detail})

    major_known = major_is_cs
    if not major_known or asked_key in {"unspecified", "unknown", "none"}:
        status = "CHECKLIST_NOT_ON_FILE"
        reason = (
            "The advising file only has a Computer Science checklist. "
            f"It cannot clear a change to '{asked}'."
        )
    elif holds.lower() not in {"none", "no", "n/a"}:
        status = "NOT_ELIGIBLE"
        reason = f"A hold is on the record: {holds}."
    elif "good" not in standing.lower():
        status = "NOT_ELIGIBLE"
        reason = f"Academic standing is '{standing}'."
    elif gpa < minimum:
        status = "NOT_ELIGIBLE"
        reason = f"Cumulative GPA {gpa:.2f} is below the checklist minimum {minimum:.2f}."
    elif not prereqs_met:
        status = "NOT_ELIGIBLE"
        reason = "At least one prerequisite group on the checklist is unfinished."
    else:
        status = "ELIGIBLE"
        reason = (
            f"Cumulative GPA {gpa:.2f} meets {minimum:.2f}, every checklist "
            "prerequisite is posted at C or better, standing is good, and no hold is listed."
        )

    payload = {
        "student_name": facts.get("STUDENT_NAME", ""),
        "student_id": facts.get("STUDENT_ID", ""),
        "current_major": facts.get("CURRENT_MAJOR", ""),
        "intended_major_on_file": checklist_major,
        "intended_major_asked": asked,
        "cumulative_gpa": gpa,
        "term_gpa": facts.get("TERM_GPA", ""),
        "earned_credits": facts.get("EARNED_CREDITS", ""),
        "academic_standing": standing,
        "holds": holds,
        "checklist_gpa_minimum": minimum,
        "completed_courses": completed,
        "in_progress": in_progress,
        "prerequisite_check": prereq_report,
        "eligibility_status": status,
        "eligibility_reason": reason,
        "degree_map_note": facts.get("DEGREE_MAP_NOTE", ""),
        "concentration_interest": facts.get("CONCENTRATION_INTEREST", ""),
        "advisor": facts.get("ADVISOR", ""),
        "source_file": STUDENT_RECORD.name,
    }
    return json.dumps(payload, indent=2)


@function_tool
def read_student_record(intended_major: str) -> str:
    """Read student_record_Stilwell.txt and check GPA, prerequisites, standing, and holds.

    intended_major: the major the student asked about, for example Computer Science.
    """
    print(f"[TOOL CALL] local read_student_record intended_major={intended_major!r}")
    payload = read_student_record_text(intended_major)
    print("[TOOL RESULT] read_student_record")
    print(payload)
    return payload


def _policy_value(policy_text: str, label: str) -> str:
    for line in policy_text.splitlines():
        if line.startswith(label):
            return line.split(":", 1)[1].strip()
    return ""


def build_advising_plan(eligibility_json: str, policy_text: str, question: str) -> AdvisingPlan:
    """Turn the two handoff payloads into the four required sections."""
    try:
        eligibility = json.loads(eligibility_json)
    except json.JSONDecodeError:
        eligibility = {"eligibility_reason": eligibility_json, "eligibility_status": "UNPARSED"}

    name = eligibility.get("student_name") or "The student"
    status = eligibility.get("eligibility_status", "UNKNOWN")
    deadline = _policy_value(policy_text, "DEADLINE") or "deadline not returned by the policy tool"
    forms = _policy_value(policy_text, "FORMS") or "forms not returned by the policy tool"
    gpa_rule = _policy_value(policy_text, "DEPARTMENT_GPA_MINIMUM")
    prereqs = _policy_value(policy_text, "PREREQUISITES")
    offline = "MCP_STATUS: OFFLINE" in policy_text or policy_text.startswith("ERROR:")

    summary = (
        f"{name} (ID {eligibility.get('student_id', 'unknown')}) is a "
        f"{eligibility.get('current_major', 'current major not on file')} student "
        f"with a {eligibility.get('cumulative_gpa', 'unknown')} cumulative GPA and "
        f"{eligibility.get('earned_credits', 'unknown')} earned credits. "
        f"They asked: {question.strip()} "
        f"The record lists {eligibility.get('concentration_interest', 'no concentration')} "
        f"as the concentration of interest, and {eligibility.get('advisor', 'the advisor')} "
        f"is the faculty advisor on the file."
    )
    eligibility_text = (
        f"Status: {status}. {eligibility.get('eligibility_reason', '')} "
        f"Checklist minimum on the advising file: {eligibility.get('checklist_gpa_minimum', 'n/a')}. "
        f"In progress: {', '.join(eligibility.get('in_progress') or []) or 'none'}. "
        f"{eligibility.get('degree_map_note', '')}"
    )
    if offline:
        policy_text_out = (
            "The policy MCP server did not return a document, so no deadline or form "
            f"number from this run can be treated as official. Raw notice: {policy_text}"
        )
    else:
        policy_text_out = (
            f"Department GPA minimum cited by the policy tool: {gpa_rule or 'see facts'}. "
            f"Prerequisites cited by the policy tool: {prereqs or 'see facts'}. "
            f"Deadline: {deadline}. Required forms: {forms}. "
            "CMS-4 is the change of major, DAR-1 is the degree audit request, and "
            "ACN-2 is the advisor conference note. The registrar must have CMS-4 "
            "before the deadline or the review moves to the next term."
        )
    steps = (
        f"1. Confirm {name}'s ACN-2 with {eligibility.get('advisor', 'the faculty advisor')} "
        "so the conference note exists before the form is signed. "
        "2. Order DAR-1 at least one business day before submitting the major change. "
        f"3. Submit CMS-4 for {eligibility.get('intended_major_on_file', 'the intended major')} "
        f"by {deadline}. "
        "4. Keep MAC 2311 on the schedule even though the checklist already accepts MAC 1147, "
        "because the degree map still expects Calculus I before COP 2006. "
        "5. After the registrar posts the major, declare the software engineering "
        "concentration with a Computer Science advisor. Do not put that concentration on CMS-4."
    )
    return AdvisingPlan(
        student_summary=summary,
        eligibility_status=eligibility_text,
        policy_requirements=policy_text_out,
        recommended_next_steps=steps,
    )


def make_synthesis_tool(eligibility_json: str, policy_text: str, question: str):
    """Tool closed over the handoff so the model cannot drop a form number."""

    @function_tool(name_override="synthesize_action_plan")
    def synthesize_action_plan() -> str:
        """Build the four-part advising plan from the eligibility and policy handoff."""
        print("[TOOL CALL] local synthesize_action_plan")
        plan = build_advising_plan(eligibility_json, policy_text, question)
        payload = plan.model_dump_json(indent=2)
        print("[TOOL RESULT] synthesize_action_plan")
        print(payload)
        return payload

    return synthesize_action_plan


def render_plan(plan: AdvisingPlan) -> str:
    return (
        "Student Summary\n"
        f"{plan.student_summary.strip()}\n\n"
        "Eligibility Status\n"
        f"{plan.eligibility_status.strip()}\n\n"
        "Policy Requirements\n"
        f"{plan.policy_requirements.strip()}\n\n"
        "Recommended Next Steps\n"
        f"{plan.recommended_next_steps.strip()}\n"
    )


def _settings() -> ModelSettings:
    # Some reasoning models reject an explicit temperature.
    if MODEL.startswith(("gpt-5", "o1", "o3", "o4")):
        return ModelSettings(tool_choice="required", parallel_tool_calls=False)
    return ModelSettings(temperature=0, tool_choice="required", parallel_tool_calls=False)


def _last_tool(hooks: ConsoleHooks, name: str) -> str:
    for tool_name, text in reversed(hooks.tool_results):
        if name in tool_name:
            return text
    return ""


def _print_handoff(source: str, target: str, payload: object) -> None:
    print(f"\n========== HANDOFF: {source} -> {target} ==========")
    print("payload format: JSON")
    if isinstance(payload, str):
        print(payload)
    else:
        print(json.dumps(payload, indent=2))
    print("========== END HANDOFF ==========\n")


def _build_agents(policy_server: MCPServerStdio | None) -> tuple[Agent, Agent]:
    settings = _settings()
    eligibility = Agent(
        name="Eligibility Agent",
        instructions=(
            "You are the Eligibility Agent for Gulf Cypress University advising. "
            "Your only job is the student's academic record. Call read_student_record "
            "exactly once. Pass the intended major named in the question. If the "
            "question does not name a major, pass Computer Science only when the "
            "question is about switching into computing; otherwise pass unspecified. "
            "After the tool returns, stop. Do not invent GPA values, courses, or policy."
        ),
        tools=[read_student_record],
        input_guardrails=[eligibility_input_guardrail],
        model=MODEL,
        model_settings=settings,
    )
    policy_tools_note = (
        "Call the MCP tool lookup_major_change_policy exactly once. "
        "The query must ask for the Computer Science major-change process, "
        "deadline, forms, GPA minimum, and prerequisites."
    )
    policy = Agent(
        name="Policy Agent",
        instructions=(
            "You are the Policy Agent. You do not re-check the student's GPA. "
            "You only retrieve official major-change rules. "
            + policy_tools_note
            + " After the tool returns, stop. Do not invent form numbers or deadlines."
        ),
        mcp_servers=[policy_server] if policy_server is not None else [],
        model=MODEL,
        model_settings=settings,
    )
    return eligibility, policy


def _advisor_agent(eligibility_json: str, policy_text: str, question: str) -> Agent:
    return Agent(
        name="Advisor Agent",
        instructions=(
            "You are the Advisor Agent for Gulf Cypress University. "
            "Call synthesize_action_plan exactly once. It takes no arguments. "
            "Then copy its four JSON fields verbatim into student_summary, "
            "eligibility_status, policy_requirements, and recommended_next_steps. "
            "Do not shorten them and do not add a fifth section."
        ),
        tools=[make_synthesis_tool(eligibility_json, policy_text, question)],
        output_type=AdvisingPlan,
        output_guardrails=[advising_output_guardrail],
        model=MODEL,
        model_settings=_settings(),
    )


async def _run_advisor(advisor: Agent, handoff: dict, hooks: ConsoleHooks) -> AdvisingPlan | None:
    prompt = (
        "The handoff below is already loaded into synthesize_action_plan. "
        "Call that tool once and copy its four fields into the output.\n\n"
        + json.dumps(handoff, indent=2)
    )
    for attempt in (1, 2):
        try:
            result = await Runner.run(advisor, prompt, hooks=hooks, max_turns=6)
            plan = result.final_output
            if not isinstance(plan, AdvisingPlan):
                plan = AdvisingPlan.model_validate(plan)
            print("\n===== FINAL ADVISING PLAN =====")
            print(render_plan(plan))
            print("[OUTPUT GUARDRAIL] Advisor Agent output accepted")
            return plan
        except OutputGuardrailTripwireTriggered as tripped:
            info = tripped.guardrail_result.output.output_info
            print(f"[OUTPUT GUARDRAIL] attempt {attempt} rejected the plan: {info}")
            if attempt == 1:
                print("[OUTPUT GUARDRAIL] retrying Advisor Agent with a structure reminder")
                prompt += (
                    "\n\nThe previous draft failed the output guardrail. "
                    "Each of the four fields must be a full paragraph copied from "
                    "the synthesize_action_plan JSON. Do not leave a field blank."
                )
            else:
                print("[OUTPUT GUARDRAIL] FAILURE FLAGGED after the retry")
    return None


async def advise(question: str, policy_server: MCPServerStdio | None) -> None:
    print("\n" + "=" * 72)
    print("STUDENT QUESTION:")
    print(question.strip())
    print("=" * 72)

    allowed, reason, message = screen_student_query(question)
    print(f"[INPUT GUARDRAIL] pre-check on Eligibility Agent: {reason}")
    if not allowed:
        print("[INPUT GUARDRAIL] TRIPWIRE — Eligibility Agent will not start")
        print(f"[INPUT GUARDRAIL] student message: {message}")
        # Also exercise the SDK guardrail path. With run_in_parallel=False this
        # trips before the model call, so a rejected question does not need a
        # completion. If the SDK itself errors, the pre-check above already
        # delivered the student message.
        eligibility, _ = _build_agents(policy_server)
        try:
            await Runner.run(eligibility, question, max_turns=2)
            print("[INPUT GUARDRAIL] unexpected: SDK run was not stopped")
        except InputGuardrailTripwireTriggered as tripped:
            print("[INPUT GUARDRAIL] SDK tripwire confirmed")
            print(f"[INPUT GUARDRAIL] output_info: {tripped.guardrail_result.output.output_info}")
        except Exception as exc:
            print(f"[INPUT GUARDRAIL] SDK confirmation skipped: {type(exc).__name__}: {exc}")
            print("[INPUT GUARDRAIL] the pre-check above already stopped the pipeline")
        return

    eligibility, policy = _build_agents(policy_server)
    eligibility_hooks = ConsoleHooks()
    eligibility_result = await Runner.run(
        eligibility,
        question,
        hooks=eligibility_hooks,
        max_turns=4,
    )
    eligibility_json = _last_tool(eligibility_hooks, "read_student_record")
    if not eligibility_json:
        eligibility_json = read_student_record_text("Computer Science")
        print("[HANDOFF NOTE] model did not return the tool text; using the tool file read")
    try:
        eligibility_payload = json.loads(eligibility_json)
    except json.JSONDecodeError:
        eligibility_payload = {"raw": eligibility_json}
    _print_handoff(
        "Eligibility Agent",
        "Policy Agent",
        {
            "from": "Eligibility Agent",
            "to": "Policy Agent",
            "why": "Policy lookup should use the intended major and must not recompute GPA.",
            "eligibility": eligibility_payload,
            "agent_final_output": str(eligibility_result.final_output),
        },
    )

    policy_hooks = ConsoleHooks()
    if policy_server is None:
        policy_text = (
            "MCP_STATUS: OFFLINE\n"
            "The custom policy server did not start. Do not invent deadlines or form numbers. "
            "Tell the student to open policy_doc_Stilwell.txt at the registrar."
        )
        print("[MCP OFFLINE] Policy Agent has no server. Handoff will carry the offline notice.")
    else:
        policy_question = (
            "Using only your MCP tool, look up the major-change process, deadline, "
            "required forms, GPA minimum, and prerequisites for this handoff:\n"
            + json.dumps(eligibility_payload)
        )
        policy_result = await Runner.run(policy, policy_question, hooks=policy_hooks, max_turns=4)
        policy_text = _last_tool(policy_hooks, "lookup_major_change_policy")
        if not policy_text:
            policy_text = str(policy_result.final_output)
            print("[HANDOFF NOTE] MCP tool text was empty; passing the Policy Agent output")

    advisor_handoff = {
        "from": "Policy Agent",
        "to": "Advisor Agent",
        "why": "The advisor needs both the record check and the official process.",
        "student_question": question.strip(),
        "eligibility_json": eligibility_json,
        "policy_text": policy_text,
    }
    _print_handoff("Policy Agent", "Advisor Agent", advisor_handoff)

    advisor = _advisor_agent(eligibility_json, policy_text, question)
    advisor_hooks = ConsoleHooks()
    plan = await _run_advisor(advisor, advisor_handoff, advisor_hooks)
    if plan is None:
        print("[OUTPUT GUARDRAIL] showing the synthesis tool output so the gap is visible")
        fallback = build_advising_plan(eligibility_json, policy_text, question)
        print(render_plan(fallback))


async def demonstrate_output_guardrail() -> None:
    """Show the same guardrail function rejecting a thin plan and passing a full one."""
    print("\n" + "=" * 72)
    print("OUTPUT GUARDRAIL FIXTURE (no model call)")
    print("=" * 72)
    advisor = Agent(name="Advisor Agent", instructions="fixture")
    thin = AdvisingPlan(
        student_summary="Maya is a student.",
        eligibility_status="Eligible.",
        policy_requirements="",
        recommended_next_steps="File a form.",
    )
    thin_result = await advising_output_guardrail.run(None, advisor, thin)  # type: ignore[arg-type]
    print(f"[OUTPUT GUARDRAIL] thin fixture tripwire={thin_result.output.tripwire_triggered}")
    print(f"[OUTPUT GUARDRAIL] thin fixture info={thin_result.output.output_info}")
    full = build_advising_plan(
        read_student_record_text("Computer Science"),
        "DEADLINE: Friday of the third week of the term, 5:00 p.m. Eastern\n"
        "FORMS: CMS-4; DAR-1; ACN-2\n"
        "DEPARTMENT_GPA_MINIMUM: 3.00\n"
        "PREREQUISITES: COP 1500 or COP 2000 with C or better",
        "I want to switch my major to Computer Science. What do I need to do?",
    )
    full_result = await advising_output_guardrail.run(None, advisor, full)  # type: ignore[arg-type]
    print(f"[OUTPUT GUARDRAIL] complete fixture tripwire={full_result.output.tripwire_triggered}")
    if thin_result.output.tripwire_triggered and not full_result.output.tripwire_triggered:
        print("[OUTPUT GUARDRAIL] fixture passed: thin plan trips, complete plan does not")
    else:
        print("[OUTPUT GUARDRAIL] FAILURE FLAGGED — fixture did not behave as expected")


async def main() -> None:
    print("Part A — Gulf Cypress University major-change advising")
    print(f"model: {MODEL}")
    print(f"record: {STUDENT_RECORD.name}")
    print(f"mcp server: {SERVER_SCRIPT.name}")

    params = MCPServerStdioParams(
        command=sys.executable,
        args=[str(SERVER_SCRIPT)],
        cwd=str(HERE),
    )
    policy_server = MCPServerStdio(
        name="Gulf Cypress Policy Server",
        params=params,
        cache_tools_list=True,
        client_session_timeout_seconds=60,
        tool_filter=create_static_tool_filter(
            allowed_tool_names=["lookup_major_change_policy"]
        ),
    )
    connected = False
    try:
        await policy_server.connect()
        connected = True
        listed = await policy_server.list_tools()
        names = [tool.name for tool in listed]
        print(f"[MCP] connected to {policy_server.name}")
        print(f"[MCP] tools visible to the Policy Agent: {names}")
        if "lookup_major_change_policy" not in names:
            raise RuntimeError(f"policy tool missing from {names}")
    except Exception as exc:
        print(f"[MCP OFFLINE] {type(exc).__name__}: {exc}")
        print("[MCP OFFLINE] Continuing with an explicit offline notice in the handoff.")
        if connected:
            await policy_server.cleanup()
        policy_server = None  # type: ignore[assignment]
        connected = False

    failures: list[BaseException] = []
    try:
        for question in GUARDRAIL_CASES:
            await advise(question, None)
        await demonstrate_output_guardrail()
        for question in SUCCESS_CASES:
            try:
                await advise(question, policy_server if connected else None)
            except Exception as exc:
                failures.append(exc)
                print(f"[RUN FAILURE] {type(exc).__name__}: {exc}")
    finally:
        if connected and policy_server is not None:
            await policy_server.cleanup()
            print("[MCP] policy server stopped")
    if failures:
        raise failures[0]


# Executable cases. The commented block at the bottom of the file is the
# submission copy of these same inputs.
SUCCESS_CASES = [
    "I want to switch my major to Computer Science. What do I need to do?",
]
GUARDRAIL_CASES = [
    "Switch major now",
    "What is the best pizza topping for a Friday night with friends?",
]


if __name__ == "__main__":
    asyncio.run(main())


# ---------------------------------------------------------------------------
# Test inputs exercised above. At least one must trip the input guardrail.
#
# 1. "I want to switch my major to Computer Science. What do I need to do?"
#    Full pipeline. Eligibility Agent -> Policy Agent -> Advisor Agent.
#
# 2. "Switch major now"
#    Three words. Input guardrail trips before any tool call.
#
# 3. "What is the best pizza topping for a Friday night with friends?"
#    Long enough, but not an advising, business, or technology question.
#    Input guardrail trips.
#
# 4. Output-guardrail fixture: a plan with blank policy requirements trips,
#    and a plan built from the record and the policy facts does not.
# ---------------------------------------------------------------------------
