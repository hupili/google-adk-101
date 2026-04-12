# Study: Agent Starter Pack (ASP) Local Deployment & UV Index Configuration

## 📅 Date
Sunday, April 12, 2026

## 🎯 Background
This study demonstrates how to solve key blockers when working with the Google Agent Development Kit (ADK) and the Agent Starter Pack (ASP):
1. **Local Development**: How to use a local source template instead of a remote alias.
2. **Dependency Resolution**: How to bypass restricted internal registries and use public PyPI.

## ✅ Key Patterns
- **Local Template Path**: Use `local@./path/to/template` with `agent-starter-pack create`.
- **Hermetic Dependencies**: Set `index-url = "https://pypi.org/simple"` in `pyproject.toml` and use `UV_NO_CONFIG=1` to ensure clean environment setup.
- **Platform Sanitization**: The provided `Makefile` filters out Windows-specific packages (`pywin32`) to prevent Cloud Build failures on Linux.

## 📦 The Study Pack
This directory provides a walkthrough for reproducing a minimal agent deployment.

### 1. The Source Template
Located in `../../sample_agents/minimal_agent/`. This follows the standard `python/agents/[name]/[package]` structure.

### 2. The Deployment Command
To deploy a new agent project using this local template, run the following from within this directory:

```bash
# Export the agent name
export AGENT_NAME=my-hello-world-agent

# Run the create command using the local source
UV_NO_CONFIG=1 uvx agent-starter-pack==0.15.4 create ${AGENT_NAME} \
    -d agent_engine \
    -ag \
    -a local@../../sample_agents/minimal_agent \
    --region us-central1 \
    --auto-approve
```

### 3. The Development & Deployment Workflow
Once the project is created, `cd` into the new folder:

```bash
cd ${AGENT_NAME}

# Step A: Install and Sync
# Uses public PyPI and ignores global config
make sync

# Step B: Deploy to Cloud
# Sanitizes dependencies and deploys to Vertex AI
make backend
```

### 4. Verification
After deployment, verify the agent using the provided script:

```bash
# Update the agent engine ID in the script if necessary
python verify_remote.py
```
