# Configuration Guide

Comprehensive configuration options for the Local MCP System.

## mcpdata Configuration

### Default Configuration

mcpdata uses sensible defaults, but you can customize behavior with configuration files.

### Workspace Configuration

Create `.mcpdata/config.json` in your workspace:

```json
{
  "indexing": {
    "enable_embeddings": false,
    "chunk_size": 512,
    "parallel_workers": 4,
    "max_file_size": 10485760,
    "incremental_updates": true
  },
  "parsing": {
    "extract_functions": true,
    "extract_classes": true,
    "extract_headers": true,
    "include_comments": true
  },
  "search": {
    "enable_fuzzy_search": true,
    "max_results": 50,
    "relevance_threshold": 0.1
  },
  "storage": {
    "compression": "gzip",
    "backup_on_update": true
  },
  "ignored_patterns": [
    "node_modules/",
    ".git/",
    "__pycache__/",
    "*.tmp",
    "*.log"
  ],
  "file_types": {
    "documents": [".md", ".rst", ".txt", ".adoc"],
    "code": [".py", ".js", ".ts", ".java", ".cpp", ".rs", ".go"],
    "config": [".json", ".yaml", ".yml", ".toml", ".ini"],
    "notebooks": [".ipynb", ".rmd"]
  }
}
```

### Global Configuration

Edit `~/Documents/mcpdata/config.json`:

```json
{
  "registry": {
    "auto_backup": true,
    "backup_interval_hours": 24,
    "max_backups": 7
  }
}
```

## mcp-global-server Configuration

### Environment Variables

Create `.env` file in mcp-global-server/:

```env
MCP_REGISTRY_PATH=C:\Users\YourName\Documents\mcpdata
MCP_LOG_LEVEL=INFO
MCP_CACHE_ENABLED=true
```

### Registry Structure

The server reads from:
```
~/Documents/mcpdata/
├── registry.json          # Workspace metadata
├── workspaces/            # Individual workspace data
├── global/                # Global search indices
└── backups/               # Registry backups
```

## CLI Configuration Options

### mcpdata CLI Options

```bash
# Basic usage with options
mcpdata /path/to/project \
  --workspace-name "Project Name" \
  --workspace-description "Project description" \
  --parallel-workers 8 \
  --verbose \
  --no-embeddings

# Advanced options
mcpdata /path/to/project \
  --config custom-config.json \
  --force \
  --no-central-registry
```

### Common Option Combinations

#### Fast Indexing (No Embeddings)
```bash
mcpdata /project --no-embeddings --parallel-workers 8
```

#### Comprehensive Indexing (With Embeddings)
```bash
mcpdata /project --enable-embeddings --parallel-workers 4
```

#### Minimal Indexing (Basic Search Only)
```bash
mcpdata /project --config minimal-config.json
```

## File Type Configuration

### Adding Custom File Types

```json
{
  "file_types": {
    "documents": [".md", ".rst", ".txt", ".wiki"],
    "code": [".py", ".js", ".ts", ".rb", ".php"],
    "config": [".json", ".yaml", ".toml", ".env"],
    "custom": [".custom", ".special"]
  }
}
```

### Parser Customization

```json
{
  "parsers": {
    ".md": "markdown",
    ".py": "python",
    ".js": "javascript",
    ".custom": "text"
  }
}
```

## Performance Tuning

### For Large Projects

```json
{
  "indexing": {
    "parallel_workers": 8,
    "chunk_size": 1024,
    "enable_embeddings": false,
    "max_file_size": 5242880
  }
}
```

### For Memory-Constrained Systems

```json
{
  "indexing": {
    "parallel_workers": 2,
    "chunk_size": 256,
    "enable_embeddings": false
  }
}
```

### For High-Quality Search

```json
{
  "indexing": {
    "enable_embeddings": true,
    "chunk_size": 512,
    "parallel_workers": 4
  },
  "search": {
    "relevance_threshold": 0.3,
    "max_results": 20
  }
}
```

## Examples

See [EXAMPLES.md](EXAMPLES.md) for configuration examples in action.
