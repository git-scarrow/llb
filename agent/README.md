# LLB Hybrid Agent Client (agent)

User manual for the Hybrid Agent Client packaged under `agent/`.

## Installation

From the repository root:

```bash
cd agent
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

Optional dev/test extras:

```bash
python -m pip install -e ".[dev]"
```

## Configuration

The client reads configuration from environment variables. Create a `.env` in `agent/`
or export variables in your shell.

### .env template

```dotenv
# Core LB connection
AI_LB_URL=http://localhost:8000
WORKER_MODEL=auto
WORKER_MAX_TURNS=5
WORKER_TIMEOUT_SECS=60

# MCP tools (optional)
# Example using npx:
# MCP_FILESYSTEM_CMD="npx -y @modelcontextprotocol/server-filesystem /absolute/path"
# Example using uvx:
# MCP_FILESYSTEM_CMD="uvx mcp-server-filesystem --root /absolute/path"
# Optional shell tools:
# MCP_SHELL_CMD="npx -y @modelcontextprotocol/server-shell"
MCP_STARTUP_TIMEOUT_SECS=10

# Orchestrator settings
ORCH_PROVIDER=openai
ORCH_MODEL=gpt-4o-mini
ORCH_MAX_TASKS=16
ORCH_VERIFY_WITH_LLM=false

# Required when using OpenAI planner/verifier
OPENAI_API_KEY=sk-your-key
```

Notes:
- If `MCP_FILESYSTEM_CMD`/`MCP_SHELL_CMD` are unset, the Worker runs without tools.
- `ORCH_PROVIDER=openai` requires `OPENAI_API_KEY` to enable the OpenAI planner/verifier.

## Usage

### Worker

```bash
python -m agent worker --task "Summarize the latest status" --payload '{"temperature":0.2}'
```

### Orchestrator

```bash
python -m agent orchestrator --request "Break down the rollout plan into tasks"
```

### Environment check

```bash
python -m agent.check_env
```
