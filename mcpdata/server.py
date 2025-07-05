"""
MCP Server - Production-ready server for Model Context Protocol
"""

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor
import threading

from .core.models import QueryResult
from .core.config import ConfigManager, SearchConfig
from .core.search import SearchEngine, CachedSearchEngine
from .core.embeddings import EmbeddingManager


@dataclass
class QueryRequest:
    """Request structure for MCP queries"""
    query: str
    max_tokens: Optional[int] = None
    limit: Optional[int] = None
    search_type: str = 'hybrid'  # 'keyword', 'semantic', 'hybrid'
    include_metadata: bool = False
    format: str = 'text'  # 'text', 'json', 'structured'


@dataclass
class QueryResponse:
    """Response structure for MCP queries"""
    content: str
    metadata: Dict[str, Any]
    performance: Dict[str, float]
    results_count: int
    query_id: Optional[str] = None


@dataclass
class ServerStats:
    """Server performance statistics"""
    total_queries: int = 0
    total_query_time: float = 0.0
    average_query_time: float = 0.0
    cache_hit_rate: float = 0.0
    active_connections: int = 0
    uptime: float = 0.0
    last_query_time: Optional[float] = None


class MCPServer:
    """Production-ready MCP server"""

    def __init__(self, mcp_data_path: Path, config: Optional[SearchConfig] = None,
                 enable_cache: bool = True, cache_size: int = 100,
                 max_workers: int = 4):
        """
        Initialize MCP server

        Args:
            mcp_data_path: Path to .mcpdata directory
            config: Search configuration
            enable_cache: Enable query result caching
            cache_size: Maximum cache size
            max_workers: Maximum worker threads
        """
        self.mcp_data_path = Path(mcp_data_path)
        self.config = config or SearchConfig()
        self.max_workers = max_workers
        self.start_time = time.time()

        # Initialize logging
        self.logger = self._setup_logging()

        # Initialize search engine
        if enable_cache:
            self.search_engine = CachedSearchEngine(
                self.mcp_data_path, self.config, cache_size
            )
        else:
            self.search_engine = SearchEngine(self.mcp_data_path, self.config)

        # Load indices on startup
        self._load_indices()

        # Thread pool for concurrent processing
        self.executor = ThreadPoolExecutor(max_workers=max_workers)

        # Server state
        self.is_running = False
        self.stats = ServerStats()
        self._stats_lock = threading.Lock()

        # Active connections tracking
        self.active_connections = set()
        self._connections_lock = threading.Lock()

        self.logger.info("MCP Server initialized successfully")

    def _setup_logging(self) -> logging.Logger:
        """Setup server logging"""
        logger = logging.getLogger('mcp_server')
        logger.setLevel(logging.INFO)

        # Clear existing handlers
        logger.handlers.clear()

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # File handler
        try:
            log_dir = self.mcp_data_path / 'logs'
            log_dir.mkdir(exist_ok=True)

            file_handler = logging.FileHandler(log_dir / 'server.log')
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        except Exception as e:
            logger.warning(f"Could not setup file logging: {e}")

        return logger

    def _load_indices(self):
        """Load search indices on startup"""
        try:
            if not self.search_engine.load_indices():
                raise RuntimeError("Failed to load search indices")
            self.logger.info("Search indices loaded successfully")
        except Exception as e:
            self.logger.error(f"Failed to load indices: {e}")
            raise

    def _update_stats(self, query_time: float):
        """Update server statistics"""
        with self._stats_lock:
            self.stats.total_queries += 1
            self.stats.total_query_time += query_time
            self.stats.average_query_time = (
                self.stats.total_query_time / self.stats.total_queries
            )
            self.stats.last_query_time = time.time()

            # Update cache stats if available
            if hasattr(self.search_engine, 'get_cache_stats'):
                cache_stats = self.search_engine.get_cache_stats()
                total_requests = cache_stats['cache_hits'] + cache_stats['cache_misses']
                if total_requests > 0:
                    self.stats.cache_hit_rate = cache_stats['cache_hits'] / total_requests

    def _add_connection(self, connection_id: str):
        """Track active connection"""
        with self._connections_lock:
            self.active_connections.add(connection_id)
            self.stats.active_connections = len(self.active_connections)

    def _remove_connection(self, connection_id: str):
        """Remove active connection"""
        with self._connections_lock:
            self.active_connections.discard(connection_id)
            self.stats.active_connections = len(self.active_connections)

    def _format_response(self, results: List[QueryResult], request: QueryRequest,
                        query_time: float) -> QueryResponse:
        """Format query results into response"""
        if request.format == 'json':
            content = self._format_json_response(results)
        elif request.format == 'structured':
            content = self._format_structured_response(results)
        else:
            content = self._format_text_response(results, request.max_tokens or 4000)

        metadata = {
            'results_found': len(results),
            'search_type': request.search_type,
            'query_processed': request.query,
            'timestamp': time.time()
        }

        if request.include_metadata:
            metadata['detailed_results'] = [
                {
                    'section_id': r.section_id,
                    'title': r.title,
                    'score': r.score,
                    'match_type': r.match_type,
                    'file_path': r.file_path,
                    'line_range': r.line_range
                } for r in results
            ]

        performance = {
            'query_time': query_time,
            'results_count': len(results),
            'server_uptime': time.time() - self.start_time
        }

        return QueryResponse(
            content=content,
            metadata=metadata,
            performance=performance,
            results_count=len(results)
        )

    def _format_text_response(self, results: List[QueryResult], max_tokens: int) -> str:
        """Format results as optimized text for LLM consumption"""
        if not results:
            return "No relevant content found for your query."

        context_parts = []
        current_tokens = 0

        for result in results:
            # Estimate tokens (rough: 4 chars per token)
            result_tokens = len(result.content) // 4

            if current_tokens + result_tokens <= max_tokens:
                # Add section with proper formatting
                section_content = f"## {result.title}\n\n{result.content}"
                if result.file_path != 'unknown':
                    section_content += f"\n\n*Source: {result.file_path}*"

                context_parts.append(section_content)
                current_tokens += result_tokens
            else:
                # Try to fit a truncated version
                remaining_tokens = max_tokens - current_tokens
                if remaining_tokens > 100:
                    truncated_content = result.content[:remaining_tokens * 4]
                    section_content = f"## {result.title}\n\n{truncated_content}..."
                    if result.file_path != 'unknown':
                        section_content += f"\n\n*Source: {result.file_path}*"
                    context_parts.append(section_content)
                break

        return "\n\n---\n\n".join(context_parts)

    def _format_json_response(self, results: List[QueryResult]) -> str:
        """Format results as JSON"""
        json_results = []
        for result in results:
            json_results.append({
                'section_id': result.section_id,
                'title': result.title,
                'content': result.content,
                'score': result.score,
                'match_type': result.match_type,
                'file_path': result.file_path,
                'line_range': list(result.line_range)
            })

        return json.dumps(json_results, indent=2, ensure_ascii=False)

    def _format_structured_response(self, results: List[QueryResult]) -> str:
        """Format results in a structured format"""
        if not results:
            return "No results found."

        sections = []
        for i, result in enumerate(results, 1):
            section = [
                f"### Result {i}: {result.title}",
                f"**Relevance Score:** {result.score:.3f}",
                f"**Match Type:** {result.match_type}",
                f"**Source:** {result.file_path}",
                "",
                result.content,
                ""
            ]
            sections.append("\n".join(section))

        return "\n---\n\n".join(sections)

    async def query_async(self, request: QueryRequest, connection_id: str = None) -> QueryResponse:
        """
        Asynchronous query processing

        Args:
            request: Query request object
            connection_id: Optional connection identifier

        Returns:
            QueryResponse object
        """
        start_time = time.time()

        # Track connection
        if connection_id:
            self._add_connection(connection_id)

        try:
            # Run query in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(
                self.executor,
                self._execute_query,
                request
            )

            query_time = time.time() - start_time
            self._update_stats(query_time)

            response = self._format_response(results, request, query_time)

            self.logger.info(
                f"Query processed: '{request.query[:50]}...' "
                f"in {query_time:.3f}s, {len(results)} results"
            )

            return response

        except Exception as e:
            self.logger.error(f"Query failed: {e}")
            query_time = time.time() - start_time
            return QueryResponse(
                content=f"Error processing query: {str(e)}",
                metadata={'error': str(e), 'timestamp': time.time()},
                performance={'query_time': query_time, 'error': True},
                results_count=0
            )

        finally:
            if connection_id:
                self._remove_connection(connection_id)

    def query_sync(self, request: QueryRequest) -> QueryResponse:
        """
        Synchronous query processing

        Args:
            request: Query request object

        Returns:
            QueryResponse object
        """
        start_time = time.time()

        try:
            results = self._execute_query(request)
            query_time = time.time() - start_time
            self._update_stats(query_time)

            response = self._format_response(results, request, query_time)

            self.logger.info(
                f"Query processed: '{request.query[:50]}...' "
                f"in {query_time:.3f}s, {len(results)} results"
            )

            return response

        except Exception as e:
            self.logger.error(f"Query failed: {e}")
            query_time = time.time() - start_time
            return QueryResponse(
                content=f"Error processing query: {str(e)}",
                metadata={'error': str(e), 'timestamp': time.time()},
                performance={'query_time': query_time, 'error': True},
                results_count=0
            )

    def _execute_query(self, request: QueryRequest) -> List[QueryResult]:
        """Execute the actual query"""
        max_tokens = request.max_tokens or self.config.default_token_limit
        limit = request.limit or self.config.max_results

        if request.search_type == 'keyword':
            # Keyword-only search
            return self.search_engine._keyword_search(
                self.search_engine.preprocessor.preprocess_query(request.query)
            )[:limit]
        elif request.search_type == 'semantic':
            # Semantic-only search
            return self.search_engine._semantic_search(request.query)[:limit]
        else:
            # Hybrid search (default)
            return self.search_engine.query(request.query, max_tokens, limit)

    def get_stats(self) -> Dict[str, Any]:
        """Get server statistics"""
        with self._stats_lock:
            stats_dict = asdict(self.stats)
            stats_dict['uptime'] = time.time() - self.start_time

            # Add search engine stats
            if hasattr(self.search_engine, 'get_performance_stats'):
                engine_stats = self.search_engine.get_performance_stats()
                stats_dict['search_engine'] = engine_stats

            # Add cache stats if available
            if hasattr(self.search_engine, 'get_cache_stats'):
                cache_stats = self.search_engine.get_cache_stats()
                stats_dict['cache'] = cache_stats

            return stats_dict

    def health_check(self) -> Dict[str, Any]:
        """Perform health check"""
        health = {
            'status': 'healthy',
            'timestamp': time.time(),
            'uptime': time.time() - self.start_time,
            'active_connections': self.stats.active_connections,
            'indices_loaded': True,
            'last_query': self.stats.last_query_time
        }

        try:
            # Test query to ensure system is responsive
            test_request = QueryRequest(query="test", max_tokens=100, limit=1)
            start_time = time.time()
            self._execute_query(test_request)
            health['response_time'] = time.time() - start_time
        except Exception as e:
            health['status'] = 'unhealthy'
            health['error'] = str(e)

        return health

    def reload_indices(self) -> bool:
        """Reload search indices"""
        try:
            self.logger.info("Reloading search indices")
            if self.search_engine.load_indices():
                self.logger.info("Indices reloaded successfully")
                return True
            else:
                self.logger.error("Failed to reload indices")
                return False
        except Exception as e:
            self.logger.error(f"Error reloading indices: {e}")
            return False

    def start(self):
        """Start the server"""
        self.is_running = True
        self.logger.info("MCP Server started")

    def stop(self):
        """Stop the server"""
        self.is_running = False
        self.executor.shutdown(wait=True)
        self.logger.info("MCP Server stopped")

    def __enter__(self):
        """Context manager entry"""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.stop()


class MCPServerManager:
    """Manager for multiple MCP server instances"""

    def __init__(self):
        self.servers: Dict[str, MCPServer] = {}
        self.logger = logging.getLogger('mcp_server_manager')

    def create_server(self, name: str, mcp_data_path: Path, **kwargs) -> MCPServer:
        """Create and register a new server instance"""
        if name in self.servers:
            raise ValueError(f"Server '{name}' already exists")

        server = MCPServer(mcp_data_path, **kwargs)
        self.servers[name] = server
        self.logger.info(f"Created server '{name}' for {mcp_data_path}")
        return server

    def get_server(self, name: str) -> Optional[MCPServer]:
        """Get server instance by name"""
        return self.servers.get(name)

    def remove_server(self, name: str) -> bool:
        """Remove server instance"""
        if name in self.servers:
            server = self.servers[name]
            server.stop()
            del self.servers[name]
            self.logger.info(f"Removed server '{name}'")
            return True
        return False

    def list_servers(self) -> List[str]:
        """List all server names"""
        return list(self.servers.keys())

    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all servers"""
        return {name: server.get_stats() for name, server in self.servers.items()}

    def shutdown_all(self):
        """Shutdown all servers"""
        for name, server in self.servers.items():
            server.stop()
            self.logger.info(f"Shutdown server '{name}'")
        self.servers.clear()


# Convenience functions for quick usage
def create_server(mcp_data_path: Union[str, Path], **kwargs) -> MCPServer:
    """Create a new MCP server instance"""
    return MCPServer(Path(mcp_data_path), **kwargs)


def quick_query(mcp_data_path: Union[str, Path], query: str, **kwargs) -> str:
    """Quick query function for simple usage"""
    server = MCPServer(Path(mcp_data_path))

    request = QueryRequest(query=query, **kwargs)
    response = server.query_sync(request)

    server.stop()
    return response.content


async def quick_query_async(mcp_data_path: Union[str, Path], query: str, **kwargs) -> str:
    """Async quick query function"""
    server = MCPServer(Path(mcp_data_path))

    try:
        request = QueryRequest(query=query, **kwargs)
        response = await server.query_async(request)
        return response.content
    finally:
        server.stop()


# Global server manager instance
_server_manager = MCPServerManager()

def get_global_manager() -> MCPServerManager:
    """Get global server manager instance"""
    return _server_manager
