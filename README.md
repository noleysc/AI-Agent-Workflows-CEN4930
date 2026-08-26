# AI Agents In Action (2nd Edition)

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/downloads/) [![License](https://img.shields.io/badge/license-MIT-green)](LICENSE) [![NRP](https://img.shields.io/badge/NRP-LLM%20API-blue)](https://nrp.ai/documentation/userdocs/ai/llm-managed/api-access) [![MCP](https://img.shields.io/badge/Protocol-MCP-orange)](https://platform.openai.com/docs/guides/mcp)

This repository contains sample code for the book "Build a Deep Research Agent from Scratch." The examples use the OpenAI Agents SDK against the [National Research Platform (NRP)](https://nrp.ai/) OpenAI-compatible LLM API.

## Setup Instructions

### 1. Clone the Repository

To get started, clone this repository to your local machine:

```bash
git clone https://github.com/noleysc/AI-Agent-Workflows-CEN4930.git
cd AI-Agent-Workflows-CEN4930
```

### 2. Create Your Environment

This project requires **Python 3.11+**. Create and activate a Python virtual environment:

#### On Windows:

```bash
python -m venv venv
venv\Scripts\activate
```

#### On macOS/Linux:

```bash
python3 -m venv venv
source venv/bin/activate
```

If you prefer to use an external Python environment, ensure you set the Python path in VS Code:

1. Open the Command Palette (`Ctrl+Shift+P` or `Cmd+Shift+P` on macOS).
2. Search for "Python: Select Interpreter."
3. Choose the Python interpreter for your environment.

### 3. Install Dependencies

#### Path A: Using VS Code Debugging

If you have VS Code, you can simply start debugging (press `F5`) to run the examples. The required dependencies will be installed automatically as part of the debugging process.

#### Path B: Manual Installation

Alternatively, you can manually install the dependencies using pip:

```bash
pip install -r requirements.txt
```

`requirements.txt` includes an editable install of this repo so `import nrp` works from any chapter directory. That module points the OpenAI Agents SDK at NRP's Chat Completions endpoint.

### 4. Configure the NRP API key

Create a `.env` file in the root directory. Use the provided `.env.example` file as a template:

```
OPENAI_API_KEY=your_nrp_llm_token
OPENAI_BASE_URL=https://ellm.nrp-nautilus.io/v1
OPENAI_DEFAULT_MODEL=gpt-oss
```

1. Sign in at [https://nrp.ai/llmtoken](https://nrp.ai/llmtoken) with your institutional account.
2. Create a token for a group that has the LLM flag enabled.
3. Paste the token as `OPENAI_API_KEY`. Keep that name — the OpenAI Python SDK and Agents SDK both read it.
4. Leave `OPENAI_BASE_URL` set to NRP's Envoy AI Gateway: `https://ellm.nrp-nautilus.io/v1`.

Do not commit `.env`. List current NRP model ids with:

```bash
curl -H "Authorization: Bearer $OPENAI_API_KEY" https://ellm.nrp-nautilus.io/v1/models
```

Chat examples default to `gpt-oss`. Embedding examples use `qwen3-embedding`. Vision examples use `qwen3-small`. Override those with `OPENAI_DEFAULT_MODEL`, `NRP_EMBEDDING_MODEL`, and `NRP_VISION_MODEL` if you want a different model from the [NRP catalog](https://nrp.ai/documentation/userdocs/ai/llm-managed/models).

#### What still needs a real OpenAI key

NRP does not implement OpenAI's image generation, Sora, Realtime/WebRTC, or hosted MCP connector APIs. These samples will fail against NRP until you point them at OpenAI (or skip them):

- `chapter_07/07_image_generation_agent.py`, `chapter_07/08_image_vision_critic_agents.py` (generation tool)
- `chapter_08/02_app.py` and the chapter 8 HTML Realtime demos
- `bonus_projects/sora_*.py`, `bonus_projects/concurrent_image_generation.py`
- `bonus_projects/mcp_examples/07_mcp_agent_hosted_server.py`

OpenAI platform tracing is disabled by default because NRP tokens cannot write to `platform.openai.com`. Chapter 7 Phoenix examples still work if you run a local Phoenix collector.

### 5. Run the Code

To execute the sample code, navigate to the desired chapter and run the Python file. For example:

```bash
python chapter_02/01_first_agent.py
```

This will run the agent and display the output in the terminal.

## Notes

- Ensure you are using the correct Python interpreter that matches your environment.
- The `.env` file should not be shared or committed to version control to keep your API key secure.
- `nrp.py` loads `.env`, sets `OPENAI_BASE_URL`, switches the Agents SDK to Chat Completions, and disables OpenAI tracing. Example scripts import it at startup.

If you see `No matching route found` from the NRP gateway, the request used OpenAI's Responses API (`/v1/responses`) or a model NRP does not host (for example `gpt-4o` or `gpt-5.6-luna`). Import `nrp` first and use `gpt-oss` (or another id from `/v1/models`). A `[non-fatal] Tracing client error 401` means traces were sent to `api.openai.com`; `nrp.py` disables that.
