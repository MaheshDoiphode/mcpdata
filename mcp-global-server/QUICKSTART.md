# Global MCP Server - Quick Start

## Setup

```bash
cd mcp-global-server
python setup_global_server.py --setup
```

This creates the central registry at `~/Documents/mcpdata/`

## Register Workspaces

```bash
# Register your documentation
python -m mcp /path/to/docs \
  --workspace-name "My Docs" \
  --workspace-description "Project documentation"

# Register more workspaces
python -m mcp /path/to/code \
  --workspace-name "Source Code" \
  --workspace-description "Application code"
```

## Start Server

```bash
python setup_global_server.py --start
```

## MCP Client Config

Add to your MCP client:

```json
{
  "mcpServers": {
    "global-docs": {
      "command": "python",
      "args": ["/path/to/mcp-global-server/__init__.py"]
    }
  }
}
```

## Available Tools

- **global_search** - Search across all workspaces
- **get_content** - Get detailed content from a workspace  
- **list_workspaces** - List all registered workspaces

## Commands

```bash
# Setup
python setup_global_server.py --setup

# Start/stop server
python setup_global_server.py --start
python setup_global_server.py --stop

# Check status
python setup_global_server.py --status
```

That's it! You can now search across all your documentation from one place.