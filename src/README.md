# mcpdata - Fast Documentation & Code Indexing

Fast, efficient indexing and searching for documentation and code repositories. The core engine powering the Local MCP System.

## 🎯 Overview

**mcpdata** is a high-performance indexing system that creates searchable indexes of your documentation and code. It's designed to be fast, modular, and AI-friendly.

### What it does:
- 🚀 **Fast Indexing**: Parallel processing with smart content extraction
- 📁 **Multi-format Support**: Markdown, Python, JavaScript, and more
- 🔍 **Intelligent Search**: Full-text search with relevance ranking
- 🌐 **Central Registry**: Manages multiple workspaces from one location
- 🧠 **AI Ready**: Optimized for AI assistant integration

### Key Features:
- ⚡ Lightning-fast parallel processing
- 📊 Smart content chunking and extraction
- 🔄 Incremental updates (only re-index changed files)
- 💾 Persistent storage with optimized indexes
- 🎯 Context-aware search results
- 🌐 Cross-workspace discovery

## 🚀 Installation

### From Source
```bash
cd mcpdata
pip install -e .
```

### Verify Installation
```bash
mcpdata --help
```

## 📖 CLI Usage

### Basic Usage
```bash
# Index current directory
mcpdata .

# Index specific directory
mcpdata /path/to/your/project

# Index with workspace information
mcpdata /path/to/docs \
  --workspace-name "API Documentation" \
  --workspace-description "REST API documentation and guides"
```

### CLI Options

| Option | Description | Default |
|--------|-------------|---------|
| `--force` | Force re-initialization even if .mcpdata exists | `False` |
| `--config PATH` | Path to configuration file | `None` |
| `--verbose` | Enable verbose logging | `False` |
| `--no-embeddings` | Skip embedding generation | `False` |
| `--parallel-workers N` | Number of parallel workers | `4` |
| `--workspace-name NAME` | Name for the workspace | Auto-generated |
| `--workspace-description DESC` | Description for the workspace | Auto-generated |
| `--no-central-registry` | Skip central registry registration | `False` |

### Advanced Examples

```bash
# High-performance indexing with 8 workers
mcpdata /large/codebase --parallel-workers 8 --verbose

# Index without embeddings for faster processing
mcpdata /docs --no-embeddings --workspace-name "Quick Docs"

# Re-index existing directory
mcpdata /project --force

# Index without central registry (standalone)
mcpdata /private/docs --no-central-registry
```

## 🐍 Python API

### Quick Start
```python
from mcpdata import init_directory, query_directory

# Initialize directory
stats = init_directory("/path/to/project")
print(f"Indexed {stats['files_processed']} files")

# Query indexed directory
results = query_directory("/path/to/project", "authentication")
print(results)
```

### Core Classes

#### MCPInitializer
```python
from mcpdata import MCPInitializer
from pathlib import Path

# Initialize with custom options
initializer = MCPInitializer(
    root_path=Path("/path/to/project"),
    verbose=True,
    parallel_workers=8,
    workspace_name="My Project",
    workspace_description="Main project documentation"
)

# Run initialization
stats = initializer.initialize()
```

#### Search Engine
```python
from mcpdata.core.search import SearchEngine
from pathlib import Path

# Create search engine
search_engine = SearchEngine(Path("/path/to/project/.mcpdata"))

# Basic search
results = search_engine.search("authentication methods")

# Get optimized context for AI
context = search_engine.get_optimized_context(
    "how to configure database",
    max_results=10,
    include_code=True
)
```

#### Configuration Manager
```python
from mcpdata.core.config import ConfigManager

# Load configuration
config = ConfigManager.load_config("/path/to/config.json")

# Access configuration sections
indexing_config = config.indexing
search_config = config.search
```

## 📁 Directory Structure

After indexing, mcpdata creates a `.mcpdata` directory:

```
your-project/
├── .mcpdata/
│   ├── index/
│   │   ├── content.json        # Document sections and text
│   │   ├── symbols.json        # Code functions, classes, etc.
│   │   ├── search.json         # Search indices
│   │   └── embeddings.json     # Vector embeddings (if enabled)
│   ├── metadata/
│   │   ├── files.json          # File metadata and timestamps
│   │   ├── config.json         # Workspace configuration
│   │   └── stats.json          # Indexing statistics
│   └── cache/
│       ├── chunks.json         # Optimized content chunks
│       └── processed.json      # Processing cache
└── your-files...
```

## 🌐 Central Registry

mcpdata maintains a central registry at `~/Documents/mcpdata/`:

```
~/Documents/mcpdata/
├── registry.json              # All workspace metadata
├── workspaces/
│   ├── workspace1/
│   │   ├── metadata.json      # Workspace info
│   │   └── index_summary.json # Quick stats
│   └── workspace2/
│       └── ...
├── global/
│   ├── search_index.json      # Global search index
│   └── cross_refs.json        # Cross-workspace references
└── backups/
    └── registry_backup_*.json # Automatic backups
```

## ⚙️ Configuration

### Default Configuration
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

### Custom Configuration
```python
# Create config.json in your project
{
  "indexing": {
    "parallel_workers": 8,
    "enable_embeddings": true
  },
  "file_types": {
    "documents": [".md", ".txt", ".wiki"],
    "code": [".py", ".js", ".ts"]
  },
  "ignored_patterns": [
    "node_modules/",
    "build/",
    "dist/"
  ]
}
```

## 🔍 Supported File Types

### Documents
- **Markdown**: `.md`, `.markdown`
- **reStructuredText**: `.rst`
- **Plain Text**: `.txt`
- **AsciiDoc**: `.adoc`
- **Jupyter Notebooks**: `.ipynb`

### Code
- **Python**: `.py`
- **JavaScript/TypeScript**: `.js`, `.ts`, `.jsx`, `.tsx`
- **Java**: `.java`
- **C/C++**: `.c`, `.cpp`, `.h`, `.hpp`
- **Rust**: `.rs`
- **Go**: `.go`
- **Kotlin**: `.kt`

### Configuration
- **JSON**: `.json`
- **YAML**: `.yaml`, `.yml`
- **TOML**: `.toml`
- **INI**: `.ini`

### Web
- **HTML**: `.html`, `.htm`
- **CSS/SCSS**: `.css`, `.scss`

## 📊 Performance

### Indexing Performance
| Project Size | Files | Time | Memory |
|-------------|-------|------|---------|
| Small | <100 | ~5s | <50MB |
| Medium | 100-1000 | ~30s | <200MB |
| Large | 1000-5000 | ~2-5min | <500MB |
| Enterprise | 5000+ | ~10-30min | <1GB |

### Search Performance
- **Local search**: 10-50ms
- **Cross-workspace**: 50-200ms
- **With embeddings**: 100-500ms

### Optimization Tips
```bash
# Faster indexing
mcpdata /project --parallel-workers 8 --no-embeddings

# Smaller index size
mcpdata /project --config minimal-config.json

# Memory-efficient for large projects
mcpdata /project --parallel-workers 2
```

## 🛠️ Advanced Usage

### Programmatic Initialization
```python
from mcpdata import MCPInitializer
from pathlib import Path

# Custom initialization
initializer = MCPInitializer(
    root_path=Path("/path/to/project"),
    config_path="custom-config.json",
    enable_embeddings=True,
    parallel_workers=8,
    workspace_name="Production Docs",
    workspace_description="Production environment documentation"
)

# Get detailed stats
stats = initializer.initialize()
print(f"Files processed: {stats['files_processed']}")
print(f"Symbols extracted: {stats['symbols_extracted']}")
print(f"Sections created: {stats['sections_created']}")
```

### Custom Search Queries
```python
from mcpdata.core.search import SearchEngine
from pathlib import Path

engine = SearchEngine(Path("/project/.mcpdata"))

# Basic search
results = engine.search("authentication")

# Advanced search with filters
results = engine.search(
    query="database configuration",
    file_types=["python", "config"],
    max_results=20,
    include_context=True
)

# Get specific content
content = engine.get_file_content("config/database.py")
```

### Registry Management
```python
from mcpdata.core.registry import CentralRegistry

# Access central registry
registry = CentralRegistry()

# List all workspaces
workspaces = registry.list_workspaces()

# Get workspace info
info = registry.get_workspace_info("my-project")

# Search across all workspaces
results = registry.global_search("authentication methods")
```

## 🚨 Troubleshooting

### Common Issues

**`.mcpdata` directory not created:**
```bash
# Check permissions
ls -la /path/to/project

# Force re-initialization
mcpdata /path/to/project --force --verbose
```

**Slow indexing:**
```bash
# Increase parallel workers
mcpdata /path --parallel-workers 8

# Disable embeddings
mcpdata /path --no-embeddings

# Check file patterns
mcpdata /path --verbose
```

**Search returns no results:**
```python
# Check if index exists
from pathlib import Path
index_path = Path("/project/.mcpdata/index/content.json")
print(f"Index exists: {index_path.exists()}")

# Try simpler query
results = search_engine.search("simple term")
```

**Memory usage too high:**
```bash
# Reduce parallel workers
mcpdata /path --parallel-workers 2

# Use smaller chunk size in config
{
  "indexing": {
    "chunk_size": 256,
    "parallel_workers": 2
  }
}
```

### Debug Mode
```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Enable debug logging
from mcpdata import MCPInitializer
initializer = MCPInitializer(root_path=Path("/path"), verbose=True)
```

## 🧪 Testing

### Basic Functionality Test
```python
import tempfile
from pathlib import Path
from mcpdata import init_directory, query_directory

# Create test directory
with tempfile.TemporaryDirectory() as tmp_dir:
    tmp_path = Path(tmp_dir)

    # Create test file
    (tmp_path / "test.md").write_text("# Test\nThis is a test document.")

    # Index
    stats = init_directory(str(tmp_path))
    assert stats['files_processed'] > 0

    # Search
    results = query_directory(str(tmp_path), "test")
    assert len(results) > 0

    print("✅ Basic functionality works!")
```

### Performance Test
```python
import time
from mcpdata import init_directory

start_time = time.time()
stats = init_directory("/path/to/large/project")
end_time = time.time()

print(f"Indexed {stats['files_processed']} files in {end_time - start_time:.2f} seconds")
print(f"Performance: {stats['files_processed'] / (end_time - start_time):.2f} files/second")
```

## 🔧 Development

### Setting up Development Environment
```bash
# Clone and install in development mode
git clone <repo-url>
cd localmcp/mcpdata
pip install -e .

# Run tests
python -c "from mcpdata import init_directory; print('✅ Import works')"
```

### Contributing
1. Fork the repository
2. Create a feature branch
3. Make changes to the mcpdata package
4. Test thoroughly
5. Submit a pull request

### Code Structure
```
mcpdata/
├── __init__.py           # Main entry point and CLI
├── __main__.py           # Python -m mcpdata support
├── initializer.py        # MCPInitializer class
├── server.py             # MCPServer class
└── core/
    ├── __init__.py
    ├── models.py         # Data models
    ├── config.py         # Configuration management
    ├── parser.py         # File parsing
    ├── search.py         # Search engine
    ├── embeddings.py     # Vector embeddings
    ├── registry.py       # Central registry
    └── contextual_search.py  # Advanced search
```

## 📜 License

MIT License - see LICENSE file for details.

## 🎯 Next Steps

1. **Index your first project**: `mcpdata /path/to/project`
2. **Test search functionality**: Use Python API to search
3. **Set up central registry**: Let mcpdata manage multiple workspaces
4. **Integrate with MCP server**: Use with mcp-global-server for AI integration

---

**Ready to supercharge your documentation indexing?** Start with `mcpdata --help` and explore the possibilities! 🚀
