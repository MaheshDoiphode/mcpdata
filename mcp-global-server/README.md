# mcp-global-server - AI Documentation Search Tools

A Model Context Protocol (MCP) server that provides intelligent search tools for AI assistants across all your documentation workspaces indexed by mcpdata.

## Overview

**mcp-global-server** is a standalone MCP server that reads indexes created by mcpdata and exposes powerful search tools for AI assistants. It enables AI to search, retrieve, and explore documentation and code across multiple repositories from a single interface.

## Architecture

```
mcpdata/                    # Indexing tool (separate package)
├── Creates .mcpdata/ in each workspace
└── Manages ~/Documents/mcpdata/ central registry

mcp-global-server/         # This MCP server (provides AI tools)
├── Reads from central registry
├── Provides MCP tools for AI
└── Searches across all workspaces
```

**Key Separation:**
- **mcpdata**: Pure indexing tool, no MCP functionality
- **mcp-global-server**: Pure MCP server, reads mcpdata indexes

## Prerequisites

1. **Python 3.8+**
2. **mcpdata package** installed and configured
3. **At least one workspace** indexed with mcpdata
4. **MCP client** (like Claude Desktop, Cline, etc.)

## Installation

### Install Dependencies

```bash
cd mcp-global-server
pip install -r requirements.txt
```

### Manual Installation
```bash
pip install mcp fastmcp python-dotenv httpx
```

## Quick Start

### 1. Index Your Content (using mcpdata)

```bash
# First, install and use mcpdata
cd ../mcpdata
pip install -e .

# Index your documentation
mcp /path/to/your/docs \
  --workspace-name "API Documentation" \
  --workspace-description "REST API documentation and guides"

# Index your code
mcp /path/to/your/code \
  --workspace-name "Source Code" \
  --workspace-description "Main application source code"
```

### 2. Start the MCP Server

```bash
cd ../mcp-global-server
python server.py
```

### 3. Configure Your MCP Client

Add to your MCP client configuration (e.g., Claude Desktop):

```json
{
  "mcpServers": {
    "global-docs": {
      "command": "python",
      "args": ["C:\\path\\to\\localmcp\\mcp-global-server\\server.py"],
      "cwd": "C:\\path\\to\\localmcp\\mcp-global-server"
    }
  }
}
```

## MCP Tools for AI

### 🔍 `search_workspaces`

Search across all registered workspaces with intelligent query processing.

**AI Usage Examples:**
- "Search for authentication methods in the codebase"
- "Find documentation about database configuration" 
- "Look for error handling patterns in code"

**Parameters:**
- `query` (string): Search terms - AI can use natural language
- `workspace_id` (optional): Limit to specific workspace
- `max_results` (optional): Maximum results (default: 10)
- `search_type` (optional): "all", "code", "docs", "config"

**Features:**
- Multi-keyword search with relevance scoring
- Query expansion (removes "how to", "what is", etc.)
- Frequency and proximity scoring
- File type filtering

### 📄 `get_file_content`

Retrieve file content with smart metadata and outline.

**AI Usage Examples:**
- "Show me the user authentication file"
- "Get lines 50-100 from the config file"
- "Display the complete database setup file"

**Parameters:**
- `file_path` (string): Path to file
- `workspace_id` (optional): For relative path resolution
- `start_line` (optional): Starting line number
- `end_line` (optional): Ending line number

**Features:**
- **File outline** with functions, classes, headers (with line numbers)
- **Smart suggestions** for what sections to request next
- **File type detection** (code/docs/config)
- **Large file handling** with truncation warnings

### 📋 `list_workspaces`

List all registered workspaces with statistics.

**AI Usage Examples:**
- "What documentation workspaces do I have?"
- "Show me all my indexed projects"
- "List workspaces with their file counts"

**Parameters:**
- `include_stats` (optional): Include detailed statistics (default: true)

### 🏗️ `get_workspace_info`

Get detailed information about a specific workspace.

**AI Usage Examples:**
- "Tell me about the API documentation workspace"
- "Show files in the source code workspace"
- "What's the health status of my documentation?"

**Parameters:**
- `workspace_id` (string): Workspace ID
- `include_files` (optional): Include file list (default: false)

### ⚙️ `get_function_content`

Extract complete function or method content.

**AI Usage Examples:**
- "Show me the authenticate_user function"
- "Get the complete validate_config method"
- "Display the DatabaseManager class"

**Parameters:**
- `file_path` (string): Path to file
- `function_name` (string): Function/method name
- `workspace_id` (optional): For relative paths

## Configuration

### Environment Variables

Create `.env` file (optional):

```env
MCP_REGISTRY_PATH=C:\Users\YourName\Documents\mcpdata
```

**Default:** `~/Documents/mcpdata`

### Registry Structure

The server reads from:
```
~/Documents/mcpdata/
├── registry.json          # Workspace metadata
├── workspaces/            # Individual workspace data
├── global/                # Global search indices
└── backups/               # Registry backups
```

## AI Integration Examples

### Claude Desktop Configuration

```json
{
  "mcpServers": {
    "global-docs": {
      "command": "python",
      "args": ["C:\\Users\\YourName\\Desktop\\Projects\\localmcp\\mcp-global-server\\server.py"],
      "cwd": "C:\\Users\\YourName\\Desktop\\Projects\\localmcp\\mcp-global-server",
      "env": {
        "MCP_REGISTRY_PATH": "C:\\Users\\YourName\\Documents\\mcpdata"
      }
    }
  }
}
```

### Cline/Continue Configuration

```json
{
  "mcp": {
    "global-docs": {
      "command": ["python", "server.py"],
      "cwd": "/path/to/mcp-global-server"
    }
  }
}
```

## Real AI Conversation Examples

### Finding Code

**You:** "How does the user authentication work in the codebase?"

**AI Uses:** `search_workspaces("user authentication", search_type="code")`

**AI Gets:** List of authentication-related functions and classes

**AI Uses:** `get_function_content("src/auth.py", "authenticate_user")`

**AI Gets:** Complete function with implementation details

### Understanding Documentation

**You:** "What's the database configuration setup process?"

**AI Uses:** `search_workspaces("database configuration setup", search_type="docs")`

**AI Gets:** Relevant documentation sections

**AI Uses:** `get_file_content("docs/database-setup.md", start_line=45, end_line=80)`

**AI Gets:** Specific configuration steps

### Exploring Projects

**You:** "What documentation projects do I have?"

**AI Uses:** `list_workspaces(include_stats=true)`

**AI Gets:** Complete workspace list with file counts and health status

## Advanced Features

### Smart Query Processing

The server automatically improves AI queries:
- "How does authentication work?" → "authentication"
- "Show me database configuration files" → "database config"
- Removes question words, focuses on keywords

### Multi-Keyword Search

Handles complex queries intelligently:
- Scores each term individually
- Provides coverage bonuses for matching multiple terms
- Calculates proximity when terms appear close together

### File Outline Generation

When AI requests file content, it gets:
- Function and class definitions with line numbers
- Markdown headers with hierarchy
- TODO/FIXME comments
- File structure overview

## Troubleshooting

### Server Won't Start

1. **Check Python path**: Verify Python executable in MCP config
2. **Check mcpdata installation**: `pip list | grep mcpdata`
3. **Verify registry exists**: `~/Documents/mcpdata/registry.json`

### No Search Results

1. **List workspaces**: Ask AI to use `list_workspaces`
2. **Check workspace status**: Ensure workspaces show as "active"
3. **Re-index content**: Use `mcp /path --force` to re-index

### Import Errors

```bash
# Ensure mcpdata is installed
cd ../mcpdata
pip install -e .

# Check imports
python -c "from mcpdata.core.registry import CentralRegistry; print('OK')"
```

### File Path Issues

- Use absolute paths in MCP client configuration
- Ensure `cwd` is set correctly
- Check file permissions on registry directory

## Development

### Adding New Tools

1. Create function with `@mcp.tool()` decorator
2. Return JSON strings for all responses
3. Include proper error handling and suggestions
4. Add helpful metadata for AI decision-making

### Testing

```bash
# Test individual components
python -c "
from mcpdata.core.registry import get_global_registry
registry = get_global_registry()
print(f'Workspaces: {len(registry.list_workspaces())}')
"

# Test search
python -c "
from mcpdata.core.registry import get_global_registry
registry = get_global_registry()
results = registry.global_search('config')
print(f'Results: {len(results)}')
"
```

### Logging

Enable debug logging by modifying `server.py`:

```python
logging.basicConfig(level=logging.DEBUG)
```

## Security

- Only accesses files within registered workspaces
- No network access or external API calls
- All file access is logged
- Works entirely with local data

## Roadmap

- [ ] Relationship mapping between workspaces
- [ ] Cross-reference analysis
- [ ] Code dependency tracking
- [ ] Documentation freshness scoring
- [ ] Integration with more file types

## License

MIT License - see LICENSE file for details.

---

**Need Help?**
1. Check if mcpdata workspaces are properly indexed
2. Verify MCP client configuration
3. Test with simple queries first
4. Enable debug logging for troubleshooting