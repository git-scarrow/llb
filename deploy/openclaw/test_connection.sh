#!/bin/bash

# Define the container name based on the docker-compose file
CONTAINER="openclaw_agent_platform"

echo "Testing Sidecar Connection: OpenClaw -> LLB"

# Ensure the stack is up
if ! docker ps --format '{{.Names}}' | grep -q "^${CONTAINER}$"; then
    echo "❌ Container $CONTAINER is not running."
    echo "Run: docker compose -f deploy/openclaw/docker-compose.yml up -d"
    exit 1
fi

# Run the check inside the container
# We use Node.js's built-in fetch because 'curl' might not be in node:20-slim
docker exec "$CONTAINER" node -e '
  const url = process.env.OPENAI_BASE_URL.replace("/v1", "/health");
  console.log(`Connecting to ${url}...`);
  fetch(url)
    .then(res => {
      if (res.ok) return res.json();
      throw new Error(`HTTP ${res.status}`);
    })
    .then(data => {
      console.log("✅ Success! Health response:", data);
      process.exit(0);
    })
    .catch(err => {
      console.error("❌ Failed:", err.message);
      process.exit(1);
    });
'
