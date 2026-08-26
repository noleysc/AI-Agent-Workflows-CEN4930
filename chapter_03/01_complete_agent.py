import sys as _sys
from pathlib import Path as _Path

_nrp_root = next((p for p in _Path(__file__).resolve().parents if (p / "nrp.py").exists()), None)
if _nrp_root is None:
    raise ImportError(
        "nrp.py not found. Run this script from the AI-Agent-Workflows checkout."
    )
_sys.path.insert(0, str(_nrp_root))
import nrp  # NRP OpenAI-compatible API (loads .env)

# agent.py
import asyncio
from pathlib import Path

from agents import Agent, Runner
from agents.mcp import MCPServerStdio, MCPServerStdioParams

SCRIPT = Path(__file__).with_name("01_complete_mcp_server.py").resolve()


async def main():
    async with MCPServerStdio(
        name="MCP Server",
        params=MCPServerStdioParams(
            command="mcp",
            args=["run", str(SCRIPT)],
        ),
    ) as mcp_server:
        agent = Agent(
            name="Assistant",
            instructions=(
                "You are a helpful assistant. "
                "Use the available tools and resources to answer the user's question. "
                "For math calculations, **always** use the add tool. "
                "For greeting the user by name, use the greeting resource or prompt from the mcp server."
            ),
            mcp_servers=[mcp_server],
        )

        result = await Runner.run(agent, "Hello, my name is Alice. What is 5 + 7?")
        print(result.final_output)


if __name__ == "__main__":
    asyncio.run(main())
