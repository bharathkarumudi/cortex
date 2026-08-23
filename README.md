# Cortex

Cortex is a self-hosted AI assistant that enhances chat quality through multi-step review processes and persistent user memory. It sits behind Open WebUI and provides independent checking of answers while maintaining conversation history across sessions.

## Quick Start

For the fastest way to get running:

1. **Clone the repository.**
2. **Configure your environment:** `cp .env.example .env` (create a `.env.example` first if you haven't). Fill it with your `OPENROUTER_API_KEY` and `WEBUI_SECRET_KEY`.
3. **Launch:** `docker compose up -d --build`
4. **Initialize Memory:** `docker compose run --rm quality-mcp python -m scripts.init_memory_collection`
5. **Connect to Open WebUI:** Add the two MCP endpoints provided below.

## Problem It Solves

Traditional chat assistants suffer from three main issues:

1. **Inconsistent Answer Quality** - Single model calls lack adversarial checks, often producing confidently wrong responses or padding simple answers with unrequested caveats.
2. **Stateless Conversations** - Without persistent memory, each new chat starts from scratch, requiring users to re-explain decisions and preferences repeatedly.
3. **Opaque Costs and Correctness** - Multi-model workflows can get expensive quickly without proper monitoring of which model produced what part of the answer.

## Solution

Cortex addresses these issues by implementing a quality review pipeline with persistent user memory.

### Quality Answer Pipeline
- **Refine** - Sharpens vague questions into actionable ones.
- **Recall** - Pulls relevant prior conversation history before answering.
- **Respond** - Drafts answers optimized for correctness and appropriate brevity.
- **Evaluate** - Independent model review for factual accuracy and unnecessary bulk.
- **Remember** - Saves valuable decisions and facts back to memory.

### Memory Management  
- User-scoped memory storage with strict isolation between users.
- Recall, save, list, delete, and restore operations.
- Soft-delete functionality (30 days retention).
- No cross-user data leakage.

## Key Features

- **Multi-User Isolation** - Each user's memory is completely isolated from others.
- **Thread-Safe Operations** - Proper handling of concurrent requests.
- **Cost Accounting** - Per-request token and cost tracking.
- **Production-Ready** - Config validation, observability, and test suite.
- **Transparent Operation** - Clear visibility into costs and what's being remembered.

## Architecture

The system separates the quality answer loop from memory management into independent services:

- **`quality-mcp` (The Orchestrator)** - Handles answer refinement, response drafting, and evaluation. It acts as the primary brain, calling the Memory service to recall context before a response and to save new insights after a successful evaluation.
- **`memory-mcp` (The State Manager)** - Manages user-scoped memory with strict access controls.

This separation ensures trust boundaries are maintained while allowing each component to be tested and extended independently.

## Requirements

- Python 3.11+
- Docker and Docker Compose
- Open WebUI integration
- OpenRouter API access (for LLMs)

## Getting Started

### Local Development

To run services directly on your machine (instead of Docker):

1. **Create a virtual environment:**
   ```bash
   python -m venv ~/venv/cortex
   source ~/venv/cortex/bin/activate
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   pip install -r requirements-dev.txt
   ```

3. **Run tests to validate setup:**
   ```bash
   pytest -q
   ruff check .
   ruff format --check .
   ```

4. **Run a service:**
   Ensure your environment variables are exported, then run:
   ```bash
   python -m services.quality_mcp.app
   # OR
   python -m services.memory_mcp.app
   ```

### Deployment

1. **Configure environment:**
   ```bash
   cp .env.example .env
   # Edit with your real OPENROUTER_API_KEY and WEBUI_SECRET_KEY
   ```

2. **Build and start services:**
   ```bash
   docker compose up -d --build
   docker compose run --rm quality-mcp python -m scripts.init_memory_collection
   ```

3. **Register Open WebUI connections:**
   In Open WebUI, add the following MCP endpoints:
   - **Quality Service:** `http://quality-mcp:8100/mcp`
   - **Memory Service:** `http://memory-mcp:8101/mcp`

4. **Configure User Identity:**
   Crucially, set the following custom header on **both** connections in Open WebUI to ensure user-scoped memory works correctly:
   ```json
   {"X-OpenWebUI-User-Id": "{{USER_ID}}"}
   ```

## Security Model

- **Identity-First:** User IDs are always extracted from the authentication context provided by the `X-OpenWebUI-User-Id` header.
- **Strict Scoping:** No tool ever accepts another user's ID as input; memory records are strictly scoped to the authenticated caller.
- **Isolation:** Service separation ensures that even if one service is compromised, the other's trust boundaries remain intact.

## Contributing

We welcome contributions! Whether it's bug reports, new MCP tools, or documentation improvements, please feel free to open an issue or submit a Pull Request.

## Why This Matters

Cortex is designed for users who want the benefits of AI assistance without sacrificing control over their data, infrastructure, or costs. It provides:
- **Trustworthy answers** that can be acted upon.
- **Persistent conversation history** that understands who you are.
- **Transparent cost tracking** and complete data ownership.

*Example usage: "What was my decision on the project scope last week?" — Cortex will recall that specific fact from your unique memory history.*
