import sys as _sys
from pathlib import Path as _Path

_nrp_root = next((p for p in _Path(__file__).resolve().parents if (p / "nrp.py").exists()), None)
if _nrp_root is not None:
    _sys.path.insert(0, str(_nrp_root))
    import nrp  # NRP OpenAI-compatible API (loads .env)
import asyncio

from agents import Agent, Runner
from agents.mcp import MCPServerStdio


async def main():
    # Instantiate the servers first…
    thinking_srv = MCPServerStdio(
        name="sequential-thinking",
        params={
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-sequential-thinking"],
        },
    )

    instructions = """
You are helpful planning assistant.
    """
    agent = Agent(
        name="Assistant",
        instructions=instructions,
        mcp_servers=[thinking_srv],
    )

    async with thinking_srv:
        tools = await thinking_srv.list_tools()
        print("Available tools:", tools)
        goal = """
Discover and output the tool and functions you have available.
"""
        print("Running...", goal)
        result = await Runner.run(agent, goal)
        print(result.final_output)


if __name__ == "__main__":
    asyncio.run(main())
