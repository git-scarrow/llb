# OpenClaw Integration Guide

This directory contains the deployment configuration for integrating `llb` with the **OpenClaw** agent platform using the Infrastructure Sidecar pattern.

## Architecture

In this setup, `llb` acts as a "Sidecar" or "Downstream Dependency" to OpenClaw. OpenClaw treats `llb` strictly as a black-box OpenAI-compatible API provider.

```mermaid
graph LR
    OpenClaw[OpenClaw Agent] -- HTTP /v1/chat/completions --> LLB[Large Language Balancer]
    LLB -- Redis Protocol --> Redis[(Redis Registry)]
    LLB -- HTTP --> Providers[Upstream LLMs (vLLM, Ollama, etc)]
```

### Key Principles

1.  **Black Box Contract**: OpenClaw knows nothing about the internal routing logic, P2C strategy, or hedging mechanisms of `llb`. It simply sends standard OpenAI API requests.
2.  **Network Isolation**: Communication happens over a private Docker network `llm-mesh`.
3.  **Immutable Infrastructure**: `llb` containers are treated as immutable artifacts.

## Scaling

To add more compute capacity (throughput) to the system, **do not** modify the OpenClaw configuration.

Instead, scale the `llb` layer horizontally or, more commonly, register more Upstream LLM nodes into the shared Redis registry. `llb` will automatically discover new upstream nodes via Redis.

To scale the load balancer itself (if it becomes a bottleneck):
1.  Increase the replica count of the `load_balancer` service.
2.  Ensure they all share the same `redis` instance.

## Anti-Patterns

### ❌ Do NOT Import Python Code
**Strictly Forbidden:** Do not attempt to mount the `llb` source code into the OpenClaw container or import `load_balancer` Python modules directly.
- **Why?** This creates a tight coupling that breaks independent deployment and scaling. OpenClaw should be language-agnostic regarding its LLM provider.

### ❌ Do NOT Bypass the Load Balancer
Do not configure OpenClaw to talk directly to individual upstream providers (like vLLM or Ollama). Always route through `http://load_balancer:8000`.

## Quick Start

1.  Start the stack:
    ```bash
    docker compose up -d
    ```
2.  Verify connection:
    ```bash
    ./test_connection.sh
    ```
