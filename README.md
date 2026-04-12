# 101 for Google ADK

This repo includes:

- CoLab to walk through basic concepts
- Off the shelf sample agents for quick experimentation

Notebooks:

1. QuickStart
2. Deploy to Agent Engine
3. Build Deep Research

## Study: Agent Starter Pack (ASP)

Walkthrough of deploying a minimal agent using the Agent Starter Pack (ASP).

- [ASP Local Deployment & UV Index Configuration](./study/agent-starter-pack/README.md)

## Wishlist

- Agent Engine Memory Bank. ( [apply for allowlist here](https://docs.google.com/forms/d/e/1FAIpQLSe6Qn0rQKK9Qa-zOzHI0YSmero9yXPb1ideJFtGXFqpvQL2sA/viewform) )
- A2A Samples
- Agent Engine deployment samples
- Agentspace integration samples
- Working with MCP
- Meta Agent

## Learning paths

```mermaid
graph TD
    subgraph "Start Here"
        A[QuickStart.ipynb]
    end

    subgraph "Sample Developer"
        B[ADK_CoLab_Boilerplate.ipynb]
    end

    subgraph "Applications"
        H[Build_Deep_Research.ipynb]
    end

    subgraph "Using Tools"
        C[Web_request_tool.ipynb]
        D[Artifacts_service.ipynb]
        E[GenMedia.ipynb]
    end

    subgraph "Communications"
        F[Builtin_A2A.ipynb]
        G[MCP_Barehand_JSON_RPC.ipynb]
    end

    subgraph "Deployment"
        I[Deploy_to_Agent_Engine.ipynb]
        J[Agentspace_integration.ipynb]
    end

    A --> H;
    A --> C;
    A --> F;
    A --> G;
    A --> I;
    C --> D;
    D --> E;
    I --> J;
```

