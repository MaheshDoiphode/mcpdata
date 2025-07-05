# Quick Start Guide

Get up and running with the Local MCP System in minutes.

## 1. Install and Index

```bash
cd mcpdata
pip install -e .
mcpdata /path/to/your/docs --workspace-name "My Docs"
```

## 2. Start MCP Server

```bash
cd mcp-global-server
pip install -r requirements.txt
python server.py
```

## 3. Configure AI Client

Add to your MCP client configuration:

```json
{
  "mcpServers": {
    "global-docs": {
      "command": "python",
      "args": ["C:\\path\\to\\mcpdata\\mcp-global-server\\server.py"]
    }
  }
}
```

### For GitHub Copilot

Create `C:\Users\%UserProfile%\AppData\Roaming\Code\User\mcp.json`:

```json
{
    "servers": {
        "global-docs": {
            "id": "global-docs",
            "name": "global-docs",
            "version": "1.0.0",
            "config": {
                "type": "stdio",
                "command": "C:\\Users\\%UserProfile%\\AppData\\Local\\Programs\\Python\\Python312\\python.exe",
                "args": [
                    "C:\\path\\to\\your\\mcpdata\\mcp-global-server\\server.py"
                ],
                "env": {
                    "MCP_REGISTRY_PATH": "C:\\Users\\%UserProfile%\\Documents\\mcpdata"
                }
            }
        }
    },
    "inputs": []
}
```

## 4. Start Searching

Your AI can now search across all your indexed content:

- "Search for authentication methods in the codebase"
- "Find documentation about database configuration"
- "Show me the user validation function"
- "List all my documentation workspaces"

## Examples

### Index Different Types of Content

```bash
# Documentation
mcpdata /docs/api --workspace-name "API Docs" --workspace-description "REST API documentation"

# Code repositories  
mcpdata /src/backend --workspace-name "Backend" --workspace-description "Core backend services"

# Mixed projects
mcpdata /project --workspace-name "Full Project" --workspace-description "Complete project with docs and code"
```

### Test the System

```bash
# Verify indexing worked
python -c "from mcpdata import query_directory; print(query_directory('/path/to/indexed/project', 'search term'))"

# Check global registry
python -c "from mcpdata.core.registry import get_global_registry; print(len(get_global_registry().workspaces))"
```

## Next Steps

- [Read the complete installation guide](INSTALLATION.md)
- [Configure advanced options](CONFIGURATION.md)
- [See more examples](EXAMPLES.md)
- [Troubleshoot issues](TROUBLESHOOTING.md)
- [Troubleshoot issues](TROUBLESHOOTING.md)
