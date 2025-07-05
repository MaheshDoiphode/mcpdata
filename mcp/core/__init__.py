"""
MCP Core Package

Core components for the MCP (Model Context Protocol) indexing system.
Provides models, configuration, parsing, search, and embedding functionality.
"""

__version__ = "1.0.0"

# Core data models
from .models import (
    FileMetadata,
    DocumentSection,
    CodeSymbol,
    SearchIndex,
    QueryResult
)

# Configuration management
from .config import (
    MCPConfig,
    ConfigManager,
    IndexingConfig,
    SearchConfig,
    LoggingConfig
)

# Content parsing
from .parser import (
    BaseParser,
    ParserFactory,
    MarkdownParser,
    RestructuredTextParser,
    PythonParser,
    JavaScriptParser,
    ContentPreprocessor
)

# Search functionality
from .search import (
    SearchEngine,
    CachedSearchEngine,
    SearchIndexBuilder,
    QueryPreprocessor,
    BM25Scorer,
    SemanticSearcher,
    HybridRanker,
    SearchConfig as SearchEngineConfig
)

# Embedding management
from .embeddings import (
    EmbeddingManager,
    EmbeddingGenerator,
    EmbeddingStorage,
    EmbeddingConfig,
    EmbeddingModel,
    TextChunker
)

# Export all public classes and functions
__all__ = [
    # Models
    'FileMetadata',
    'DocumentSection',
    'CodeSymbol',
    'SearchIndex',
    'QueryResult',

    # Configuration
    'MCPConfig',
    'ConfigManager',
    'IndexingConfig',
    'SearchConfig',
    'LoggingConfig',

    # Parsing
    'BaseParser',
    'ParserFactory',
    'MarkdownParser',
    'RestructuredTextParser',
    'PythonParser',
    'JavaScriptParser',
    'ContentPreprocessor',

    # Search
    'SearchEngine',
    'CachedSearchEngine',
    'SearchIndexBuilder',
    'QueryPreprocessor',
    'BM25Scorer',
    'SemanticSearcher',
    'HybridRanker',
    'SearchEngineConfig',

    # Embeddings
    'EmbeddingManager',
    'EmbeddingGenerator',
    'EmbeddingStorage',
    'EmbeddingConfig',
    'EmbeddingModel',
    'TextChunker'
]

# Package metadata
PACKAGE_NAME = "mcp-core"
DESCRIPTION = "Core components for MCP indexing system"
AUTHOR = "MCP Development Team"

def get_version():
    """Get package version"""
    return __version__

def get_package_info():
    """Get package information"""
    return {
        'name': PACKAGE_NAME,
        'version': __version__,
        'description': DESCRIPTION,
        'author': AUTHOR
    }
