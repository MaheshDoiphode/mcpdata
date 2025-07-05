# Contributing Guide

Guidelines for contributing to the Local MCP System.

## 🤝 Getting Started

### Development Environment Setup

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
```

### Development Guidelines

#### Code Style
- Follow PEP 8 for Python code
- Use meaningful variable and function names
- Add docstrings for all public functions and classes
- Keep functions focused and small

#### Component Separation
- **mcpdata**: Focus on indexing performance and accuracy
- **mcp-global-server**: Focus on AI usability and response quality
- Maintain clear separation between indexing and serving
- Avoid tight coupling between components

## 🔧 Development Workflow

### 1. Fork and Branch

```bash
# Fork the repository on GitHub
# Clone your fork
git clone https://github.com/yourusername/localmcp.git
cd localmcp

# Create feature branch
git checkout -b feature-name
```

### 2. Make Changes

Focus your changes on one of the two main components:

#### For mcpdata changes:
- Work in `mcpdata/` directory
- Test indexing functionality thoroughly
- Verify performance with various project sizes
- Test both CLI and Python API

#### For mcp-global-server changes:
- Work in `mcp-global-server/` directory
- Test MCP tool functionality
- Verify AI assistant integration
- Test cross-workspace search

### 3. Testing

#### Test mcpdata functionality:
```bash
cd mcpdata

# Basic functionality test
python -c "from mcpdata import init_directory; print('✅ Import works')"

# Test with sample project
mcpdata . --workspace-name "test" --verbose

# Test search
python -c "from mcpdata import query_directory; print(query_directory('.', 'test'))"
```

#### Test mcp-global-server:
```bash
cd mcp-global-server

# Test server imports
python -c "from mcpdata.core.registry import CentralRegistry; print('✅ Server imports work')"

# Start server (test startup)
python server.py &
sleep 5
kill %1
```

### 4. Documentation

Update relevant documentation:
- Update README files for user-facing changes
- Add examples to `docs/EXAMPLES.md`
- Update configuration documentation if adding new options
- Include inline code documentation

### 5. Submit Pull Request

```bash
# Commit changes
git add .
git commit -m "Descriptive commit message"

# Push to your fork
git push origin feature-name

# Create pull request on GitHub
```

## 📝 Contribution Types

### Bug Fixes
- Fix crashes or errors
- Improve error handling
- Fix performance issues
- Correct documentation errors

### New Features
- Add new file type support
- Improve search algorithms
- Add new MCP tools
- Enhance AI integration

### Performance Improvements
- Optimize indexing speed
- Reduce memory usage
- Improve search response times
- Optimize storage efficiency

### Documentation
- Improve installation guides
- Add usage examples
- Create troubleshooting guides
- Write API documentation

## 🧪 Testing Guidelines

### Required Tests

#### For mcpdata changes:
- Test indexing with various file types
- Verify search functionality
- Test with different project sizes
- Check central registry functionality
- Test CLI options

#### For mcp-global-server changes:
- Test all MCP tools
- Verify cross-workspace search
- Test AI assistant integration
- Check error handling

### Test Scenarios

#### Basic Functionality
```python
import tempfile
from pathlib import Path
from mcpdata import init_directory, query_directory

# Create test directory
with tempfile.TemporaryDirectory() as tmp_dir:
    tmp_path = Path(tmp_dir)
    
    # Create test files
    (tmp_path / "test.md").write_text("# Test\nThis is a test document.")
    (tmp_path / "code.py").write_text("def test_function():\n    return True")
    
    # Index
    stats = init_directory(str(tmp_path))
    assert stats['files_processed'] >= 2
    
    # Search
    results = query_directory(str(tmp_path), "test")
    assert len(results) > 0
    
    print("✅ Basic functionality works!")
```

#### Performance Test
```python
import time
from mcpdata import init_directory

start_time = time.time()
stats = init_directory("/path/to/test/project")
end_time = time.time()

performance = stats['files_processed'] / (end_time - start_time)
print(f"Performance: {performance:.2f} files/second")

# Should be reasonable for project size
assert performance > 1.0  # At least 1 file per second
```

## 📋 Code Review Checklist

### Before Submitting
- [ ] Code follows project style guidelines
- [ ] All tests pass
- [ ] No breaking changes (or clearly documented)
- [ ] Documentation updated
- [ ] Performance impact considered
- [ ] Error handling included
- [ ] Logging added where appropriate

### For Reviewers
- [ ] Code is readable and well-documented
- [ ] Changes are focused and logical
- [ ] Tests cover new functionality
- [ ] Performance impact is acceptable
- [ ] No security vulnerabilities introduced
- [ ] Documentation is accurate and complete

## 🏗️ Architecture Guidelines

### mcpdata Architecture
```
mcpdata/
├── __init__.py           # Main entry point and CLI
├── __main__.py           # Python -m mcpdata support
├── initializer.py        # MCPInitializer class
├── server.py             # MCPServer class
└── core/
    ├── models.py         # Data models
    ├── config.py         # Configuration management
    ├── parser.py         # File parsing
    ├── search.py         # Search engine
    ├── embeddings.py     # Vector embeddings
    ├── registry.py       # Central registry
    └── contextual_search.py  # Advanced search
```

### Key Principles
- **Modularity**: Each component has a clear responsibility
- **Performance**: Optimize for speed and memory efficiency
- **Extensibility**: Easy to add new file types and features
- **Reliability**: Robust error handling and recovery
- **Usability**: Clear APIs and good documentation

## 🔄 Release Process

### Version Management
- Use semantic versioning (MAJOR.MINOR.PATCH)
- Update version in `setup.py` files
- Tag releases in git
- Update CHANGELOG.md

### Release Checklist
- [ ] All tests pass
- [ ] Documentation updated
- [ ] Version numbers updated
- [ ] CHANGELOG.md updated
- [ ] Git tag created
- [ ] Release notes written

## 🆘 Getting Help

### Development Questions
- Check existing issues and discussions
- Review documentation and examples
- Ask questions in GitHub discussions

### Code Review
- Be respectful and constructive
- Focus on the code, not the person
- Explain reasoning for suggestions
- Be open to feedback

### Community
- Follow the code of conduct
- Help newcomers get started
- Share knowledge and experiences
- Participate in discussions

## 📜 License

By contributing to this project, you agree that your contributions will be licensed under the MIT License.

---

**Thank you for contributing to the Local MCP System!** 🎉

Your contributions help make documentation search better for everyone.
