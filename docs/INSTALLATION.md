# Installation Guide

Complete installation guide for the Local MCP System.

## Prerequisites

- Python 3.8+
- Git

## Quick Installation

### 1. Install mcpdata

```bash
cd mcpdata
pip install -e .
```

### 2. Install MCP Server Dependencies

```bash
cd mcp-global-server
pip install -r requirements.txt
```

## Development Setup

### Full Development Installation

```bash
# Clone the repository
git clone <repository-url>
cd localmcp

# Install mcpdata in development mode
cd mcpdata
pip install -e .

# Install mcp-global-server dependencies
cd ../mcp-global-server
pip install -r requirements.txt

# Test the setup
cd ../mcpdata
mcpdata . --workspace-name "localmcp" --workspace-description "The localmcp project itself"

cd ../mcp-global-server
python server.py
```

## Verification

### Test mcpdata functionality
```bash
cd mcpdata
python -c "from mcpdata import query_directory; print('✅ mcpdata works')"

# Test global registry
python -c "from mcpdata.core.registry import get_global_registry; print('✅ Registry works')"

# Test server imports
cd ../mcp-global-server
python -c "from mcpdata.core.registry import CentralRegistry; print('✅ Server imports work')"
```

## Platform-Specific Notes

### Windows
- Use PowerShell or Command Prompt
- Ensure Python is in your PATH
- Consider using virtual environments

### macOS/Linux
- Use Terminal
- May need to use `python3` and `pip3`
- Ensure proper permissions for file operations

## Next Steps

After installation:
1. [Index your first project](QUICK-START.md)
2. [Configure your AI client](CONFIGURATION.md)
3. [Test the search functionality](EXAMPLES.md)
