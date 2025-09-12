# 101 for Google ADK

This repo includes:

- CoLab to walk through basic concepts
- Off the shelf sample agents for quick experimentation

Notebooks:

1. QuickStart
2. Deploy to Agent Engine
3. Build Deep Research

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

    subgraph "Using Tools"
        C[Web_request_tool.ipynb]
        D[Artifacts_service.ipynb]
        E[GenMedia.ipynb]
    end

    subgraph "Agent to Agent Communication"
        F[Builtin_A2A.ipynb]
        G[MCP_Barehand_JSON_RPC.ipynb]
    end

    subgraph "Advanced Topics"
        H[Build_Deep_Research.ipynb]
        I[Deploy_to_Agent_Engine.ipynb]
        J[Agentspace_integration.ipynb]
    end

    A --> C;
    A --> D;
    A --> E;
    C --> F;
    D --> F;
    E --> F;
    F --> G;
    G --> H;
    H --> I;
    I --> J;
```

