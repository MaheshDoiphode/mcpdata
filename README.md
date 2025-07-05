# MCP Data - Documentation Indexing System

Fast indexing and searching for documentation and code repositories using the Model Context Protocol (MCP).

## Structure

```
mcpdata/
├── mcp/                    # Core indexing package
│   ├── core/              # Core functionality
│   ├── initializer.py     # Main initialization
│   ├── server.py          # Single workspace server
│   └── __init__.py        # Package interface
├── mcp-global-server/     # Global search server
│   ├── __init__.py        # Global MCP server
│   └── setup_global_server.py  # Setup script
└── setup.py              # Package installation
```

## Installation

```bash
pip install -e .
```

## Basic Usage

### Index a workspace

```bash
python -m mcp /path/to/your/docs
```

This creates `.mcpdata/` with search indices.

### Search a workspace

```python
from mcp import query_directory

results = query_directory("/path/to/docs", "your query")
print(results)
```

## Global Search (Optional)

### Setup

```bash
cd mcp-global-server
python setup_global_server.py --setup
```

### Register workspaces

```bash
python -m mcp /path/to/workspace1 \
  --workspace-name "API Docs" \
  --workspace-description "REST API documentation"
```

### Start global server

```bash
python setup_global_server.py --start
```

### MCP Client Config

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

## Features

- Fast indexing of documentation and code
- BM25 + semantic search
- Single workspace or global search
- MCP server integration
- Supports Markdown, code files, and more

## License

MIT