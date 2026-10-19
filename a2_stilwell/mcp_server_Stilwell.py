"""Custom MCP server for CEN 4930 Assignment 2.

Serves two tools over stdio. Part A uses lookup_major_change_policy.
Part B uses lookup_beach_hazards. The course samples import FastMCP,
which mcp 2 renamed to MCPServer. This file accepts either name so the
same script starts in the course environment and on a current install.

Do not print to stdout. Stdio is the MCP transport.
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
POLICY_PATH = HERE / "policy_doc_Stilwell.txt"
HAZARD_PATH = HERE / "beach_hazards_Stilwell.txt"


def _make_server(name: str):
    """Return a FastMCP (mcp 1) or MCPServer (mcp 2) instance."""
    try:
        from mcp.server.fastmcp import FastMCP
    except ModuleNotFoundError:
        from mcp.server.mcpserver import MCPServer as FastMCP
    return FastMCP(name)


def _log(message: str) -> None:
    print(f"[MCP SERVER] {message}", file=sys.stderr, flush=True)


def _read(path: Path) -> str:
    if not path.is_file():
        return f"ERROR: data file not found: {path.name}"
    return path.read_text(encoding="utf-8")


def _facts_and_sections(text: str) -> tuple[list[str], list[tuple[str, str]]]:
    facts: list[str] = []
    sections: list[tuple[str, str]] = []
    title = "PREAMBLE"
    body: list[str] = []
    for line in text.splitlines():
        if line.startswith("## "):
            if body:
                sections.append((title, "\n".join(body).strip()))
            title = line[3:].strip()
            body = []
            continue
        if ":" in line and not line.startswith(" ") and title == "PREAMBLE":
            key, value = line.split(":", 1)
            if key.isupper() and value.strip():
                facts.append(f"{key.strip()}: {value.strip()}")
                continue
        body.append(line)
    if body:
        sections.append((title, "\n".join(body).strip()))
    return facts, sections


def _score(query: str, title: str, body: str) -> int:
    words = {word for word in query.lower().replace("-", " ").split() if len(word) > 2}
    haystack = f"{title} {body}".lower()
    return sum(1 for word in words if word in haystack)


def _search(path: Path, query: str, tool_name: str) -> str:
    text = _read(path)
    if text.startswith("ERROR:"):
        _log(f"{tool_name} missing file")
        return text
    facts, sections = _facts_and_sections(text)
    ranked = sorted(
        ((score, title, body) for title, body in sections if (score := _score(query, title, body))),
        key=lambda item: item[0],
        reverse=True,
    )
    chosen = ranked[:3]
    if not chosen:
        chosen = [(0, title, body) for title, body in sections[:2]]
    lines = [
        f"TOOL: {tool_name}",
        f"SOURCE_FILE: {path.name}",
        f"QUERY: {query.strip()}",
        "FACTS:",
        *facts,
        "MATCHED_SECTIONS: " + ", ".join(title for _, title, _ in chosen),
    ]
    for _, title, body in chosen:
        lines.append("")
        lines.append(f"## {title}")
        lines.append(body)
    payload = "\n".join(lines).strip()
    _log(f"{tool_name} query={query.strip()!r} sections={len(chosen)} chars={len(payload)}")
    return payload


mcp = _make_server("Stilwell Local MCP")


@mcp.tool()
def lookup_major_change_policy(query: str) -> str:
    """Look up Gulf Cypress University major-change rules.

    query: a short question about the process, deadline, forms, GPA, or prerequisites.
    Returns the labeled facts plus the policy sections that match the query.
    """
    return _search(POLICY_PATH, query, "lookup_major_change_policy")


@mcp.tool()
def lookup_beach_hazards(query: str) -> str:
    """Look up the Bonita Beach dawn-patrol hazard sheet.

    query: a short question about flags, lightning, the rip, lifeguards, or parking.
    Returns the labeled facts plus the hazard notes that match the query.
    """
    return _search(HAZARD_PATH, query, "lookup_beach_hazards")


if __name__ == "__main__":
    _log("starting stdio server with lookup_major_change_policy and lookup_beach_hazards")
    mcp.run(transport="stdio")
