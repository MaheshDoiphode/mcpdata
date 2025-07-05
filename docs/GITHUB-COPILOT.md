# GitHub Copilot Integration

Complete guide for setting up the Local MCP System with GitHub Copilot in VS Code.

## Prerequisites

1. **VS Code** with GitHub Copilot extension installed
2. **mcpdata** installed and at least one workspace indexed
3. **mcp-global-server** dependencies installed

## Setup Steps

### 1. Install and Index Content

```bash
# Clone the repository
git clone https://github.com/MaheshDoiphode/mcpdata.git
cd mcpdata

# Install mcpdata
cd mcpdata
pip install -e .

# Index your documentation
mcpdata /path/to/your/docs --workspace-name "My Documentation"

# Install MCP server dependencies
cd ../mcp-global-server
pip install -r requirements.txt
```

### 2. Create MCP Configuration

Create the MCP configuration file at:
`C:\Users\%UserProfile%\AppData\Roaming\Code\User\mcp.json`

### 3. Add Configuration Content

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

### 4. Customize Paths

Update the following paths in the configuration:

#### Python Executable
Find your Python installation:
```powershell
# In PowerShell
Get-Command python | Select-Object Source
```

Common locations:
- `C:\Users\%UserProfile%\AppData\Local\Programs\Python\Python312\python.exe`
- `C:\Users\%UserProfile%\AppData\Local\Microsoft\WindowsApps\python.exe`
- `C:\Python312\python.exe`

#### Project Path
Replace `C:\path\to\your\mcpdata` with your actual project location, for example:
- `C:\Users\%UserProfile%\Desktop\Projects\mcpdata`
- `C:\dev\mcpdata`

#### Registry Path
The registry path is typically:
- `C:\Users\%UserProfile%\Documents\mcpdata`

### 5. Test the Setup

#### Start the MCP Server
```bash
cd mcp-global-server
python server.py
```

You should see output indicating the server started successfully.

#### Test in VS Code
1. Restart VS Code
2. Open GitHub Copilot Chat
3. Try asking: "List my documentation workspaces"
4. Try asking: "Search for authentication in my docs"

## Common Issues

### Server Won't Start
- Verify Python path is correct
- Ensure `mcpdata` package is installed: `pip show mcpdata`
- Check registry path exists: `dir C:\Users\%UserProfile%\Documents\mcpdata`

### No Workspaces Found
- Verify you've indexed at least one directory with `mcpdata`
- Check registry location: The registry should be at `%UserProfile%\Documents\mcpdata\registry.json`

### Path Issues on Windows
- Use double backslashes (`\\`) in JSON configuration
- Ensure no trailing spaces in paths
- Use full paths, not relative paths

## Example Configuration for Different Setups

### Anaconda Python
```json
{
    "command": "C:\\Users\\%UserProfile%\\anaconda3\\python.exe"
}
```

### Virtual Environment
```json
{
    "command": "C:\\path\\to\\venv\\Scripts\\python.exe"
}
```

### Custom Python Installation
```json
{
    "command": "C:\\Python312\\python.exe"
}
```

## Verification Commands

Test your setup with these commands:

```bash
# Test mcpdata installation
python -c "import mcpdata; print('✅ mcpdata works')"

# Test registry access
python -c "from mcpdata.core.registry import get_global_registry; print(f'Registry has {len(get_global_registry().workspaces)} workspaces')"

# Test MCP server imports
cd mcp-global-server
python -c "from mcpdata.core.registry import CentralRegistry; print('✅ MCP server can import registry')"
```

## Advanced Configuration

### Multiple MCP Servers
You can add multiple MCP servers to your configuration:

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
                    "C:\\path\\to\\mcpdata\\mcp-global-server\\server.py"
                ],
                "env": {
                    "MCP_REGISTRY_PATH": "C:\\Users\\%UserProfile%\\Documents\\mcpdata"
                }
            }
        },
        "other-mcp-server": {
            "id": "other-server",
            "name": "other-server",
            "version": "1.0.0",
            "config": {
                "type": "stdio",
                "command": "python",
                "args": ["path/to/other/server.py"]
            }
        }
    },
    "inputs": []
}
```

### Environment Variables
You can customize behavior with environment variables:

```json
{
    "env": {
        "MCP_REGISTRY_PATH": "C:\\Users\\%UserProfile%\\Documents\\mcpdata",
        "MCP_LOG_LEVEL": "DEBUG",
        "MCP_CACHE_ENABLED": "true"
    }
}
```

## Support

For issues specific to GitHub Copilot integration:
1. Check the [Troubleshooting Guide](TROUBLESHOOTING.md)
2. Verify your configuration against this guide
3. Test the MCP server independently first
4. File an issue at: https://github.com/MaheshDoiphode/mcpdata/issues
