"""Entry point for running the Memory MCP Server as a module.

Usage:
    python -m draagon_ai.mcp
    python -m draagon_ai.mcp.server
    python -m draagon_ai.mcp --qdrant-url http://localhost:6333
"""

from draagon_ai.mcp.server import main

if __name__ == "__main__":
    main()
