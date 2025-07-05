# Troubleshooting Guide

Common issues and solutions for the Local MCP System.

## 🚨 Common Issues

### Installation Problems

#### Python Version Issues
**Problem**: `mcpdata command not found` or import errors

**Solutions:**
```bash
# Check Python version
python --version

# Ensure Python 3.8+
python3 --version

# Reinstall in development mode
cd mcpdata
pip install -e .
```

#### Permission Errors
**Problem**: `Permission denied` when creating directories

**Solutions:**
```bash
# Run with elevated permissions (Windows)
# Or check directory permissions (Linux/macOS)
chmod 755 /path/to/directory
```

### Indexing Issues

#### Large File Problems
**Problem**: `Memory error` or very slow indexing

**Solutions:**
```bash
# Reduce parallel workers
mcpdata /project --parallel-workers 2

# Set file size limit
mcpdata /project --config limited-config.json
```

**limited-config.json:**
```json
{
  "indexing": {
    "max_file_size": 1048576,
    "parallel_workers": 2
  }
}
```

#### No Files Found
**Problem**: `No files found to index`

**Solutions:**
1. Check file patterns in configuration
2. Verify directory path is correct
3. Check ignored patterns aren't too broad

```bash
# Enable verbose mode to see what's happening
mcpdata /project --verbose
```

#### Registry Issues
**Problem**: `Cannot write to central registry`

**Solutions:**
```bash
# Check registry path
python -c "from mcpdata.core.registry import get_registry_path; print(get_registry_path())"

# Use without central registry
mcpdata /project --no-central-registry

# Clear corrupted registry
python -c "from mcpdata.core.registry import clear_registry; clear_registry()"
```

### Search Problems

#### No Search Results
**Problem**: Search returns empty results for known content

**Solutions:**
1. Verify indexing completed successfully
2. Check search terms and patterns
3. Try broader search terms

```python
# Debug search
from mcpdata import query_directory
results = query_directory("/project", "term", debug=True)
```

#### Slow Search Performance
**Problem**: Search takes too long

**Solutions:**
1. Reduce result limits
2. Use more specific search terms
3. Check if too many workspaces are registered

```json
{
  "search": {
    "max_results": 10,
    "relevance_threshold": 0.3
  }
}
```

### MCP Server Issues

#### Server Won't Start
**Problem**: `python server.py` fails

**Solutions:**
```bash
# Check dependencies
cd mcp-global-server
pip install -r requirements.txt

# Check Python path
python -c "import mcpdata; print('✅ mcpdata available')"

# Check registry
python -c "from mcpdata.core.registry import CentralRegistry; print('✅ Registry works')"
```

#### Client Connection Issues
**Problem**: MCP client can't connect to server

**Solutions:**
1. Verify server is running
2. Check MCP client configuration
3. Ensure correct Python path in configuration

```json
{
  "mcpServers": {
    "global-docs": {
      "command": "python",
      "args": ["C:\\full\\path\\to\\server.py"]
    }
  }
}
```

## 🔧 Debug Mode

### Enable Verbose Logging

```bash
# CLI with verbose output
mcpdata /project --verbose

# Environment variable
export MCP_LOG_LEVEL=DEBUG
python server.py
```

### Python API Debugging

```python
import logging
logging.basicConfig(level=logging.DEBUG)

from mcpdata import init_directory
stats = init_directory("/project", verbose=True)
```

## 🧪 Diagnostic Commands

### Check Installation

```bash
# Test basic functionality
python -c "from mcpdata import init_directory; print('✅ Basic import works')"
python -c "from mcpdata.core.registry import get_global_registry; print('✅ Registry works')"
```

### Check Registry Status

```python
from mcpdata.core.registry import get_global_registry

registry = get_global_registry()
print(f"Registry path: {registry.registry_path}")
print(f"Workspaces: {len(registry.workspaces)}")
for ws in registry.workspaces:
    print(f"  - {ws.name}: {ws.path}")
```

### Test Search Functionality

```python
from mcpdata import query_directory

# Test local search
results = query_directory("/project", "test")
print(f"Found {len(results)} results")

# Test global search
from mcpdata.core.search import SearchEngine
engine = SearchEngine.create_global()
results = engine.search("test")
print(f"Global search: {len(results)} results")
```

## 🛠️ Recovery Procedures

### Reset Registry

```python
from mcpdata.core.registry import CentralRegistry
import os

# Backup current registry
registry = CentralRegistry()
registry.backup()

# Clear and reinitialize
registry.clear()
registry.initialize()
```

### Rebuild Indexes

```bash
# Force rebuild single workspace
mcpdata /project --force

# Rebuild all workspaces
python -c "
from mcpdata.core.registry import get_global_registry
registry = get_global_registry()
for ws in registry.workspaces:
    print(f'Rebuilding {ws.name}...')
    # Would need to call mcpdata for each workspace
"
```

### Clean Corrupted Data

```bash
# Remove corrupted .mcpdata directory
rm -rf /project/.mcpdata

# Reindex
mcpdata /project --workspace-name "Rebuilt Project"
```

## 🆘 Getting Help

### Log Collection

```bash
# Collect debug information
mcpdata /project --verbose > debug.log 2>&1

# Include system information
python -c "
import sys, platform
print(f'Python: {sys.version}')
print(f'Platform: {platform.platform()}')
print(f'Architecture: {platform.architecture()}')
"
```

### Common Error Messages

#### `ModuleNotFoundError: No module named 'mcpdata'`
**Solution**: Install mcpdata in development mode
```bash
cd mcpdata
pip install -e .
```

#### `PermissionError: [Errno 13] Permission denied`
**Solution**: Check directory permissions or run with appropriate privileges

#### `JSONDecodeError: Expecting value`
**Solution**: Registry file is corrupted, reset registry:
```python
from mcpdata.core.registry import clear_registry
clear_registry()
```

#### `ConnectionError: Cannot connect to MCP server`
**Solution**: Check server is running and MCP client configuration

### When to File a Bug Report

File a bug report if you encounter:
- Crashes or unhandled exceptions
- Data corruption or loss
- Features not working as documented

Include in your report:
- Python version and platform
- Complete error messages and stack traces
- Configuration files used
- Steps to reproduce the issue
- Debug logs (with sensitive information removed)

## 📞 Support Resources

- [Installation Guide](INSTALLATION.md) - Setup help
- [Configuration Guide](CONFIGURATION.md) - Configuration options
- [Examples](EXAMPLES.md) - Working examples
- GitHub Issues - Bug reports and feature requests
