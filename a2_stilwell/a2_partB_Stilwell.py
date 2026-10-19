"""CEN 4930 Assignment 2, Part B — dawn-patrol go / no-go.

Run from any working directory:

    python a2_partB_Stilwell.py

Nolan keeps a session log and a one-page hazard sheet for Bonita Beach
before a 9:30 class. Three agents split that check: the Session Log Agent
reads personal constraints, the Hazard Agent queries a custom MCP tool, and
the Skipper Agent turns both handoffs into a decision. This is not a research
summary. The output is a go, delay, or no-go call with a clock time attached.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from agents import (
    Agent,
    ModelSettings,
    RunHooks,
    Runner,
    function_tool,
    set_tracing_disabled,
)
from agents.mcp import MCPServerStdio, MCPServerStdioParams, create_static_tool_filter

HERE = Path(__file__).resolve().parent
SESSION_LOG = HERE / "session_log_Stilwell.txt"
SERVER_SCRIPT = HERE / "mcp_server_Stilwell.py"
MODEL = os.getenv("OPENAI_DEFAULT_MODEL", "gpt-4.1")

load_dotenv()
load_dotenv(HERE.parent / ".env")
set_tracing_disabled(True)


class ConsoleHooks(RunHooks):
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


def _facts(path: Path) -> dict[str, str]:
    facts: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("## "):
            break
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        if key.isupper():
            facts[key.strip()] = value.strip()
    return facts


def _fact_line(text: str, label: str) -> str:
    prefix = label if label.endswith(":") else f"{label}:"
    for line in text.splitlines():
        if line.startswith(prefix):
            return line.split(":", 1)[1].strip()
    return ""


def session_log_payload() -> str:
    facts = _facts(SESSION_LOG)
    payload = {
        "surfer": facts.get("SURFER", ""),
        "site": facts.get("SITE", ""),
        "class_name": facts.get("CLASS_NAME", ""),
        "class_time": facts.get("CLASS_TIME", ""),
        "park_back_by": facts.get("PARK_BACK_BY", ""),
        "drive_minutes": facts.get("DRIVE_MINUTES", ""),
        "board": facts.get("BOARD", ""),
        "shoulder": facts.get("SHOULDER", ""),
        "buddy": facts.get("BUDDY", ""),
        "hard_no": facts.get("HARD_NO", ""),
        "earliest_entry": facts.get("EARLIEST_ENTRY", ""),
        "planned_exit": facts.get("PLANNED_EXIT", ""),
        "sleep_hours": facts.get("SLEEP_HOURS", ""),
        "source_file": SESSION_LOG.name,
    }
    return json.dumps(payload, indent=2)


@function_tool
def read_session_log() -> str:
    """Read session_log_Stilwell.txt and return the dawn-patrol constraints as JSON."""
    print("[TOOL CALL] local read_session_log")
    text = session_log_payload()
    print("[TOOL RESULT] read_session_log")
    print(text)
    return text


def build_dawn_patrol_call(session_json: str, hazard_text: str, question: str) -> str:
    """Decide from the session constraints and the hazard sheet, not from a hunch."""
    try:
        session = json.loads(session_json)
    except json.JSONDecodeError:
        session = {"raw": session_json}

    flag = _fact_line(hazard_text, "FLAG").lower()
    lightning = _fact_line(hazard_text, "LIGHTNING").lower()
    hazard_level = _fact_line(hazard_text, "HAZARD_LEVEL") or "unknown"
    lifeguard = _fact_line(hazard_text, "LIFEGUARD_ON_DUTY") or "08:00"
    rip = _fact_line(hazard_text, "RIP")
    offline = "MCP_STATUS: OFFLINE" in hazard_text or hazard_text.startswith("ERROR:")
    waiting_for_guards = "lifeguard" in question.lower()

    if offline:
        decision = "NO-GO"
        because = "The hazard MCP tool did not answer, so the water call stays closed."
    elif "none" not in lightning and lightning:
        decision = "NO-GO"
        because = f"Lightning note is '{lightning}', which is a hard stop on the session log."
    elif "double red" in flag or flag == "closed":
        decision = "NO-GO"
        because = "The sheet shows a closure flag."
    elif waiting_for_guards:
        decision = "NO-GO"
        because = (
            f"Lifeguards are on the tower at {lifeguard}. Leaving the water then, "
            f"plus a {session.get('drive_minutes', '35')} minute drive, misses the "
            f"{session.get('park_back_by', '09:10')} campus parking cutoff and leaves "
            "less than the 45 minute water-time minimum."
        )
    else:
        decision = "GO"
        because = (
            f"Flag is {flag or 'unspecified'} and the hazard level is {hazard_level}. "
            f"Entry at {session.get('earliest_entry', '06:25')} and exit at "
            f"{session.get('planned_exit', '07:35')} clears the class at "
            f"{session.get('class_time', '09:30')}."
        )

    lines = [
        "Decision",
        f"{decision}. {because}",
        "",
        "Conditions",
        (
            f"Shoulder constraint: {session.get('shoulder', 'see log')}. "
            f"Buddy: {session.get('buddy', 'none')}. "
            f"Rip rule: {rip or 'see the hazard sheet'}. "
            "No overhead takeoffs while the tower is empty."
        ),
        "",
        "Timing",
        (
            f"Park the session car at {session.get('site', 'the access')} after the lot opens. "
            f"Be in the water by {session.get('earliest_entry', '06:25')} and out by "
            f"{session.get('planned_exit', '07:35')} so the {session.get('drive_minutes', '35')} "
            f"minute drive still hits campus parking by {session.get('park_back_by', '09:10')} "
            f"for {session.get('class_name', 'class')} at {session.get('class_time', '09:30')}."
        ),
        "",
        "What to pack",
        (
            f"Board: {session.get('board', 'funboard')}. Spring suit is already in the hatch. "
            "Bring the meter card, a towel, and a dry shirt for Holmes Hall. "
            "Do not pack the shortboard."
        ),
    ]
    return "\n".join(lines)


def make_skipper_tool(session_json: str, hazard_text: str, question: str):
    """Closed over the handoff so the skipper cannot drop the exit time."""

    @function_tool(name_override="build_dawn_patrol_call")
    def build_dawn_patrol_call_tool() -> str:
        """Turn the session log and the hazard sheet into a go or no-go call."""
        print("[TOOL CALL] local build_dawn_patrol_call")
        text = build_dawn_patrol_call(session_json, hazard_text, question)
        print("[TOOL RESULT] build_dawn_patrol_call")
        print(text)
        return text

    return build_dawn_patrol_call_tool


def _settings() -> ModelSettings:
    # Reasoning-model names in this SDK reject a temperature field.
    if MODEL.startswith(("gpt-5", "o1", "o3", "o4")):
        return ModelSettings(tool_choice="required", parallel_tool_calls=False)
    return ModelSettings(temperature=0, tool_choice="required", parallel_tool_calls=False)


def _print_handoff(source: str, target: str, payload: object) -> None:
    print(f"\n========== HANDOFF: {source} -> {target} ==========")
    print("payload format: JSON")
    print(json.dumps(payload, indent=2) if not isinstance(payload, str) else payload)
    print("========== END HANDOFF ==========\n")


def _last_tool(hooks: ConsoleHooks, name: str) -> str:
    for tool_name, text in reversed(hooks.tool_results):
        if name in tool_name:
            return text
    return ""


async def patrol(question: str, hazard_server: MCPServerStdio | None) -> None:
    print("\n" + "=" * 72)
    print("QUESTION:")
    print(question.strip())
    print("=" * 72)

    settings = _settings()
    session_agent = Agent(
        name="Session Log Agent",
        instructions=(
            "You are the Session Log Agent. Call read_session_log once and stop. "
            "Do not decide whether the surfer should paddle out. "
            "Do not describe the weather."
        ),
        tools=[read_session_log],
        model=MODEL,
        model_settings=settings,
    )
    hazard_agent = Agent(
        name="Hazard Agent",
        instructions=(
            "You are the Hazard Agent. Call lookup_beach_hazards exactly once. "
            "Ask about the flag, lightning, the rip, lifeguard hours, and the hazard level. "
            "Do not rewrite the surfer's class schedule."
        ),
        mcp_servers=[hazard_server] if hazard_server is not None else [],
        model=MODEL,
        model_settings=settings,
    )

    session_hooks = ConsoleHooks()
    session_result = await Runner.run(session_agent, question, hooks=session_hooks, max_turns=4)
    session_json = _last_tool(session_hooks, "read_session_log")
    if not session_json:
        session_json = session_log_payload()
        print("[HANDOFF NOTE] session tool text missing; passing the parsed log")
    _print_handoff(
        "Session Log Agent",
        "Hazard Agent",
        {
            "from": "Session Log Agent",
            "to": "Hazard Agent",
            "why": "The hazard lookup needs the site and the class deadline, not a second copy of the diary.",
            "session": json.loads(session_json) if session_json.lstrip().startswith("{") else session_json,
            "agent_final_output": str(session_result.final_output),
        },
    )

    hazard_hooks = ConsoleHooks()
    if hazard_server is None:
        hazard_text = "MCP_STATUS: OFFLINE\nThe beach hazard server did not start."
        print("[MCP OFFLINE] Hazard Agent has no server.")
    else:
        hazard_prompt = (
            "Look up beach hazards for this session handoff. Use only the MCP tool.\n"
            + session_json
        )
        hazard_result = await Runner.run(
            hazard_agent, hazard_prompt, hooks=hazard_hooks, max_turns=4
        )
        hazard_text = _last_tool(hazard_hooks, "lookup_beach_hazards")
        if not hazard_text:
            hazard_text = str(hazard_result.final_output)
            print("[HANDOFF NOTE] hazard tool text missing; passing the agent output")

    skipper_handoff = {
        "from": "Hazard Agent",
        "to": "Skipper Agent",
        "why": "The skipper needs the personal clock and the sheet's flag in one payload.",
        "question": question.strip(),
        "session_json": session_json,
        "hazard_text": hazard_text,
    }
    _print_handoff("Hazard Agent", "Skipper Agent", skipper_handoff)

    skipper = Agent(
        name="Skipper Agent",
        instructions=(
            "You are the Skipper Agent. Call build_dawn_patrol_call exactly once. "
            "It takes no arguments; the handoff is already loaded. "
            "Return the tool text unchanged as your final answer. "
            "Do not upgrade a NO-GO into a GO."
        ),
        tools=[make_skipper_tool(session_json, hazard_text, question)],
        model=MODEL,
        model_settings=settings,
    )
    skipper_hooks = ConsoleHooks()
    skipper_result = await Runner.run(
        skipper,
        "Make the dawn-patrol call from this JSON. Use the tool and return its text.\n"
        + json.dumps(skipper_handoff),
        hooks=skipper_hooks,
        max_turns=4,
    )
    print("\n===== SKIPPER CALL =====")
    printed = _last_tool(skipper_hooks, "build_dawn_patrol_call")
    print(printed or str(skipper_result.final_output))


async def main() -> None:
    print("Part B — Bonita Beach dawn patrol before class")
    print(f"model: {MODEL}")
    print(f"session log: {SESSION_LOG.name}")
    print(f"mcp server: {SERVER_SCRIPT.name}")

    hazard_server = MCPServerStdio(
        name="Bonita Beach Hazard Server",
        params=MCPServerStdioParams(
            command=sys.executable,
            args=[str(SERVER_SCRIPT)],
            cwd=str(HERE),
        ),
        cache_tools_list=True,
        client_session_timeout_seconds=60,
        tool_filter=create_static_tool_filter(allowed_tool_names=["lookup_beach_hazards"]),
    )
    connected = False
    try:
        await hazard_server.connect()
        connected = True
        names = [tool.name for tool in await hazard_server.list_tools()]
        print(f"[MCP] connected to {hazard_server.name}")
        print(f"[MCP] tools visible to the Hazard Agent: {names}")
        if "lookup_beach_hazards" not in names:
            raise RuntimeError(f"hazard tool missing from {names}")
    except Exception as exc:
        print(f"[MCP OFFLINE] {type(exc).__name__}: {exc}")
        if connected:
            await hazard_server.cleanup()
        hazard_server = None  # type: ignore[assignment]
        connected = False

    try:
        for question in TEST_CASES:
            await patrol(question, hazard_server if connected else None)
    finally:
        if connected and hazard_server is not None:
            await hazard_server.cleanup()
            print("[MCP] hazard server stopped")


TEST_CASES = [
    "Should I paddle out at Bonita Beach this morning before my 9:30 class?",
    "Can I stay in the water until the lifeguards arrive and still make class?",
]


if __name__ == "__main__":
    asyncio.run(main())


# ---------------------------------------------------------------------------
# Test inputs exercised above.
#
# 1. "Should I paddle out at Bonita Beach this morning before my 9:30 class?"
#    Session Log Agent -> Hazard Agent -> Skipper Agent. Expected call: GO,
#    with an exit at 07:35 and no overhead takeoffs.
#
# 2. "Can I stay in the water until the lifeguards arrive and still make class?"
#    Same handoffs. Expected call: NO-GO, because the tower opens at 08:00
#    and the drive back would miss the 09:10 parking cutoff.
# ---------------------------------------------------------------------------
