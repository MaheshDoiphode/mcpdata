# Local MCP System

A comprehensive system for local documentation indexing and AI-powered search through MCP (Model Context Protocol). Enable AI assistants to search across all your documentation and code repositories from a single interface.

## 🚀 Quick Start

### 1. Install and Index
```bash
cd mcpdata
pip install -e .
mcpdata /path/to/your/docs --workspace-name "My Docs"
```

### 2. Start MCP Server
```bash
cd mcp-global-server
pip install -r requirements.txt
python server.py
```

### 3. Configure AI Client
```json
{
  "mcpServers": {
    "global-docs": {
      "command": "python",
      "args": ["C:\\path\\to\\localmcp\\mcp-global-server\\server.py"]
    }
  }
}
```

## 🛠️ GitHub Copilot Configuration

To use with GitHub Copilot, add the MCP server configuration to your VS Code settings:

### Location
Create or edit: `C:\Users\%UserProfile%\AppData\Roaming\Code\User\mcp.json`

### Configuration
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
                    "C:\\path\\to\\your\\localmcp\\mcp-global-server\\server.py"
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

**Note:** Replace the paths with your actual Python installation and project location.

## 🎮 Examples

### Search Across Documentation and Code
**You:** "Search for mcp configuration"

![Search Results](docs/assets/search.png)

### Get Complete File Contents
**You:** "Get the contents of the initializer file"

![Get Contents](docs/assets/getContents.png)

## 🏗️ Architecture

**localmcp** consists of two components:

1. **[mcpdata](src/README.md)** - Fast indexing engine
2. **[mcp-global-server](mcp-global-server/README.md)** - MCP server for AI

**Flow:** Index files → Create searchable database → AI searches via MCP

## 📊 What You Can Search

- **Documentation**: `.md`, `.rst`, `.txt`, `.adoc`
- **Code**: `.py`, `.js`, `.ts`, `.java`, `.cpp`, `.rs`
- **Config**: `.json`, `.yaml`, `.yml`, `.toml`

## 🔗 Documentation

- **[Complete Installation Guide](docs/INSTALLATION.md)**
- **[GitHub Copilot Setup](docs/GITHUB-COPILOT.md)**
- **[Configuration Options](docs/CONFIGURATION.md)**
- **[Troubleshooting Guide](docs/TROUBLESHOOTING.md)**
- **[Contributing Guide](docs/CONTRIBUTING.md)**

## 🎯 Use Cases

- **Documentation Teams** - Search across multiple doc repositories
- **Development Teams** - Find code patterns across repositories
- **Mixed Projects** - Unified search for docs + code

## 📜 License

MIT License - see [LICENSE](LICENSE) file for details.

---

**Ready to get started?** Follow the [Quick Start](#-quick-start) above or read the [Complete Installation Guide](docs/INSTALLATION.md).
