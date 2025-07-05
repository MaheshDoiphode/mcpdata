#!/usr/bin/env python3
"""
MCP Data Initialization System
Fast, modular indexing for Model Context Protocol servers
"""

__version__ = "1.0.0"
__author__ = "MCP Team"

import sys
import argparse
from pathlib import Path

# Core imports
from .core.models import (
    FileMetadata,
    DocumentSection,
    CodeSymbol,
    SearchIndex,
    QueryResult
)

from .core.config import (
    MCPConfig,
    ConfigManager,
    IndexingConfig,
    SearchConfig,
    LoggingConfig
)

from .core.parser import (
    ParserFactory,
    MarkdownParser,
    RestructuredTextParser,
    PythonParser,
    JavaScriptParser,
    ContentPreprocessor
)

from .core.search import (
    SearchEngine,
    CachedSearchEngine,
    SearchIndexBuilder,
    QueryPreprocessor,
    BM25Scorer,
    SemanticSearcher,
    HybridRanker
)

from .core.embeddings import (
    EmbeddingManager,
    EmbeddingGenerator,
    EmbeddingStorage,
    EmbeddingConfig
)


# Main initializer class
from .initializer import MCPInitializer

# Server implementation
from .server import MCPServer

# Export main classes
__all__ = [
    # Core models
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

    # Embeddings
    'EmbeddingManager',
    'EmbeddingGenerator',
    'EmbeddingStorage',
    'EmbeddingConfig',

    # Main classes
    'MCPInitializer',
    'MCPServer'
]


def main():
    """Main entry point for CLI usage"""
    parser = argparse.ArgumentParser(
        description="Initialize MCP data directory for fast document and code indexing"
    )
    parser.add_argument(
        'path',
        nargs='?',
        default='.',
        help='Path to initialize (default: current directory)'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force re-initialization even if .mcpdata exists'
    )
    parser.add_argument(
        '--config',
        type=str,
        help='Path to configuration file'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    parser.add_argument(
        '--no-embeddings',
        action='store_true',
        help='Skip embedding generation'
    )
    parser.add_argument(
        '--parallel-workers',
        type=int,
        default=4,
        help='Number of parallel workers for processing'
    )
    parser.add_argument(
        '--workspace-name',
        type=str,
        help='Name for the workspace (for central registry)'
    )
    parser.add_argument(
        '--workspace-description',
        type=str,
        help='Description for the workspace (for central registry)'
    )
    parser.add_argument(
        '--no-central-registry',
        action='store_true',
        help='Skip central registry registration'
    )

    args = parser.parse_args()

    # Convert path to Path object
    root_path = Path(args.path).resolve()

    if not root_path.exists():
        print(f"❌ Error: Path '{root_path}' does not exist")
        sys.exit(1)

    # Check if .mcpdata exists
    mcp_data_path = root_path / '.mcpdata'
    if mcp_data_path.exists() and not args.force:
        print(f"⚠️  .mcpdata already exists in {root_path}")
        print("Use --force to re-initialize")
        sys.exit(1)

    try:
        # Initialize MCP system
        initializer = MCPInitializer(
            root_path=root_path,
            config_path=args.config,
            verbose=args.verbose,
            enable_embeddings=not args.no_embeddings,
            parallel_workers=args.parallel_workers,
            register_with_central=not args.no_central_registry,
            workspace_name=args.workspace_name,
            workspace_description=args.workspace_description
        )

        # Run initialization
        stats = initializer.initialize()

        # Print success message
        print("\n✅ MCP initialization completed!")
        print(f"📊 Processed {stats['files_processed']} files")
        print(f"📄 Parsed {stats['documents_parsed']} documents with {stats['sections_created']} sections")
        print(f"🔍 Extracted {stats['symbols_extracted']} code symbols")

        if stats.get('embeddings_generated', 0) > 0:
            print(f"🧠 Generated {stats['embeddings_generated']} embeddings")

        print("🚀 Ready for fast MCP queries!")

    except KeyboardInterrupt:
        print("\n❌ Initialization cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error during initialization: {e}")
        sys.exit(1)


def init_directory(path: str, **kwargs) -> dict:
    """
    Programmatic interface for initializing MCP data directory

    Args:
        path: Directory path to initialize
        **kwargs: Additional configuration options

    Returns:
        Dictionary with initialization statistics
    """
    root_path = Path(path).resolve()

    initializer = MCPInitializer(root_path=root_path, **kwargs)
    return initializer.initialize()


def query_directory(path: str, query: str, **kwargs) -> str:
    """
    Programmatic interface for querying MCP-indexed directory

    Args:
        path: Directory path with .mcpdata
        query: Search query
        **kwargs: Additional query options

    Returns:
        Formatted context string
    """
    root_path = Path(path).resolve()
    mcp_data_path = root_path / '.mcpdata'

    if not mcp_data_path.exists():
        raise FileNotFoundError(f"No .mcpdata found in {root_path}. Run initialization first.")

    search_engine = SearchEngine(mcp_data_path)
    return search_engine.get_optimized_context(query, **kwargs)


def start_server(path: str, **kwargs) -> 'MCPServer':
    """
    Start MCP server for the given directory

    Args:
        path: Directory path with .mcpdata
        **kwargs: Server configuration options

    Returns:
        MCPServer instance
    """
    root_path = Path(path).resolve()
    mcp_data_path = root_path / '.mcpdata'

    if not mcp_data_path.exists():
        raise FileNotFoundError(f"No .mcpdata found in {root_path}. Run initialization first.")

    server = MCPServer(mcp_data_path, **kwargs)
    return server


if __name__ == "__main__":
    main()
