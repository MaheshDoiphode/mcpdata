#!/usr/bin/env python3
"""
Global MCP Server for Documentation Search
Provides tools for searching across all registered workspaces
"""

import asyncio
import os
import json
import sys
from typing import Any, Dict, List, Optional
from pathlib import Path
import logging
from dotenv import load_dotenv

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Install with: pip install mcp fastmcp
try:
    from mcp.server.fastmcp import FastMCP
except ImportError:
    print("ERROR: Missing mcp package!")
    print("Install with: pip install mcp fastmcp")
    sys.exit(1)

# Add mcpdata to path for imports
current_dir = Path(__file__).parent
mcpdata_path = current_dir.parent / "mcpdata"
sys.path.insert(0, str(mcpdata_path))

try:
    from mcpdata.core.registry import CentralRegistry, get_global_registry
    from mcpdata.core.search import SearchEngine
except ImportError as e:
    print(f"ERROR: Could not import mcpdata modules: {e}")
    print("Make sure mcpdata is properly installed")
    sys.exit(1)

# Configuration
REGISTRY_PATH = os.getenv("MCP_REGISTRY_PATH", str(Path.home() / "Documents" / "mcpdata"))

# Initialize MCP server
mcp = FastMCP("Global Documentation MCP Server")


class GlobalDocumentationClient:
    def __init__(self):
        self.registry_path = Path(REGISTRY_PATH)
        self.registry = None
        self._load_registry()

        # Cache for search engines
        self.search_engines = {}

    def _load_registry(self):
        """Load the central registry"""
        try:
            self.registry = CentralRegistry(self.registry_path)
            logger.info(f"Loaded registry from {self.registry_path}")
        except Exception as e:
            logger.error(f"Failed to load registry: {e}")
            self.registry = None

    def _get_search_engine(self, workspace_id: str):
        """Get or create search engine for workspace"""
        if workspace_id in self.search_engines:
            return self.search_engines[workspace_id]

        workspace = self.registry.get_workspace(workspace_id)
        if not workspace:
            return None

        workspace_path = Path(workspace.path)
        mcpdata_path = workspace_path / '.mcpdata'

        if not mcpdata_path.exists():
            return None

        try:
            search_engine = SearchEngine(mcpdata_path)
            self.search_engines[workspace_id] = search_engine
            return search_engine
        except Exception as e:
            logger.error(f"Failed to create search engine for {workspace_id}: {e}")
            return None

    def _read_file_content(self, file_path: str) -> List[str]:
        """Read file content as lines"""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                return f.readlines()
        except Exception as e:
            logger.error(f"Failed to read file {file_path}: {e}")
            return []


# Initialize client
global_client = GlobalDocumentationClient()


@mcp.tool()
async def search_workspaces(query: str, workspace_id: str = None, max_results: int = 10, search_type: str = "all") -> str:
    """
    Search across registered workspaces for documentation and code.

    IMPORTANT SEARCH STRATEGY:
    - This tool works best with SHORT, KEYWORD-BASED queries (2-4 key terms)
    - Convert natural language questions into technical terms and concepts
    - Avoid full sentence questions or natural language phrases

    QUERY TRANSFORMATION EXAMPLES:
    ❌ Bad: "how mcp data configs work"
    ✅ Good: "mcp config" or "mcp data" or "mcp configuration"

    ❌ Bad: "what is the best way to implement user authentication"
    ✅ Good: "user authentication" or "auth implementation"

    ❌ Bad: "how do I troubleshoot database connection issues"
    ✅ Good: "database connection" or "connection troubleshooting"

    INSTRUCTIONS FOR AI:
    1. Extract 2-4 key technical terms from the user's question
    2. Use specific terminology, abbreviations, and concepts
    3. Prefer nouns over verbs and questions
    4. If no results, try individual key terms separately

    Args:
        query: 2-4 key technical terms (NOT full sentences or questions)
        workspace_id: Optional workspace ID to limit search scope
        max_results: Maximum number of results (default: 10)
        search_type: Type of content to search - "all", "code", "docs", "config"

    Returns:
        JSON string with search results including file paths, line numbers, and context
    """
    try:
        if not global_client.registry:
            return json.dumps({
                'error': 'Registry not available',
                'query': query
            }, indent=2)

        logger.info(f"Searching for: '{query}', workspace: {workspace_id}, type: {search_type}")

        # Process and expand query for better results
        processed_query = _process_search_query(query)
        logger.info(f"Processed query: '{processed_query}'")

        results = []

        if workspace_id:
            # Search specific workspace
            workspaces = [global_client.registry.get_workspace(workspace_id)]
            if not workspaces[0]:
                return json.dumps({
                    'error': f'Workspace not found: {workspace_id}',
                    'query': query
                }, indent=2)
        else:
            # Search all workspaces
            workspaces = global_client.registry.list_workspaces()

        # Perform global search first for quick discovery
        if not workspace_id:
            global_results = global_client.registry.global_search(processed_query, max_results)

            for result in global_results:
                workspace = global_client.registry.get_workspace(result.workspace_id)

                # Filter by search type
                if search_type != "all":
                    file_ext = Path(result.file_path).suffix.lower()

                    if search_type == "code" and file_ext not in ['.py', '.js', '.java', '.cpp', '.rs', '.go', '.kt']:
                        continue
                    elif search_type == "docs" and file_ext not in ['.md', '.rst', '.txt']:
                        continue
                    elif search_type == "config" and file_ext not in ['.json', '.yaml', '.yml', '.toml', '.ini']:
                        continue

                results.append({
                    'workspace_name': result.workspace_name,
                    'workspace_id': result.workspace_id,
                    'file_path': result.file_path,
                    'title': result.title,
                    'content_preview': result.content_preview,
                    'section_type': result.section_type,
                    'relevance_score': result.relevance_score,
                    'keywords': result.keywords[:5],  # Limit keywords
                    'workspace_path': workspace.path if workspace else None
                })

        # If no global results or specific workspace, use detailed search
        if not results:
            for workspace in workspaces:
                if not workspace:
                    continue

                search_engine = global_client._get_search_engine(workspace.id)
                if not search_engine:
                    continue

                try:
                    # Use basic search from the search engine
                    context = search_engine.get_optimized_context(processed_query, max_results=5)

                    if context and len(context.strip()) > 50:  # Has meaningful content
                        results.append({
                            'workspace_name': workspace.name,
                            'workspace_id': workspace.id,
                            'file_path': 'multiple_files',
                            'title': f'Search results in {workspace.name}',
                            'content_preview': context[:300] + "...",
                            'section_type': 'mixed',
                            'relevance_score': 0.7,
                            'keywords': query.split(),
                            'workspace_path': workspace.path
                        })
                except Exception as e:
                    logger.warning(f"Search failed for workspace {workspace.id}: {e}")
                    continue

        result_data = {
            'query': query,
            'processed_query': processed_query,
            'search_type': search_type,
            'total_results': len(results),
            'workspaces_searched': len(workspaces),
            'results': results[:max_results]
        }

        if not results:
            result_data['message'] = f"No results found for '{query}'"
            result_data['suggestions'] = [
                f"Try keywords instead: {_suggest_keywords(query)}",
                "Use specific terms: 'config database' not 'database configuration'",
                "Try technical terms: 'MCPInitializer' for classes, 'authenticate' for auth code",
                "Check workspace names with list_workspaces",
                f"Try different search_type: {_suggest_search_type(query)}"
            ]

        logger.info(f"Returning {len(results)} results")
        return json.dumps(result_data, indent=2)

    except Exception as e:
        error_msg = f"Error searching workspaces: {str(e)}"
        logger.error(error_msg)
        return json.dumps({
            'error': error_msg,
            'query': query,
            'workspace_id': workspace_id
        }, indent=2)


@mcp.tool()
async def get_file_content(file_path: str, workspace_id: str = None, start_line: int = None, end_line: int = None) -> str:
    """
    Get content from a specific file with metadata and outline for better AI context.

    Args:
        file_path: Path to the file (can be relative to workspace or absolute)
        workspace_id: Optional workspace ID to resolve relative paths
        start_line: Optional starting line number (1-based)
        end_line: Optional ending line number (1-based)

    Returns:
        JSON string with file content, metadata, outline, and helpful suggestions
    """
    try:
        resolved_path = file_path
        workspace_info = None

        # If workspace_id provided, resolve relative path
        if workspace_id and global_client.registry:
            workspace = global_client.registry.get_workspace(workspace_id)
            if workspace:
                workspace_info = {
                    'name': workspace.name,
                    'description': workspace.description,
                    'path': workspace.path
                }

                if not Path(file_path).is_absolute():
                    resolved_path = str(Path(workspace.path) / file_path)

        if not Path(resolved_path).exists():
            return json.dumps({
                'error': f'File not found: {resolved_path}',
                'file_path': file_path,
                'workspace_id': workspace_id
            }, indent=2)

        # Read file content
        lines = global_client._read_file_content(resolved_path)

        if not lines:
            return json.dumps({
                'error': f'Could not read file: {resolved_path}',
                'file_path': file_path
            }, indent=2)

        total_lines = len(lines)

        # Generate file outline
        file_outline = _generate_file_outline(lines, resolved_path)

        # Determine file type
        file_ext = Path(resolved_path).suffix.lower()
        file_type = _determine_file_type(file_ext)

        # Apply line range if specified
        if start_line is not None or end_line is not None:
            start_idx = max(0, (start_line - 1) if start_line else 0)
            end_idx = min(total_lines, end_line if end_line else total_lines)

            extracted_lines = lines[start_idx:end_idx]
            line_numbers = list(range(start_idx + 1, end_idx + 1))

            result = {
                'file_path': file_path,
                'resolved_path': resolved_path,
                'file_type': file_type,
                'total_lines': total_lines,
                'requested_range': f"{start_line or 1}-{end_line or total_lines}",
                'actual_range': f"{start_idx + 1}-{end_idx}",
                'content': [line.rstrip() for line in extracted_lines],
                'line_numbers': line_numbers,
                'outline': file_outline,
                'workspace': workspace_info,
                'suggestions': [
                    f"File has {total_lines} total lines",
                    "Use start_line and end_line for other sections",
                    f"File contains {len(file_outline)} major elements"
                ]
            }
        else:
            # Return full file (with size warning for large files)
            if total_lines > 500:
                result = {
                    'file_path': file_path,
                    'resolved_path': resolved_path,
                    'file_type': file_type,
                    'total_lines': total_lines,
                    'warning': f'Large file ({total_lines} lines). Consider using line range.',
                    'content': [line.rstrip() for line in lines[:500]],
                    'truncated': True,
                    'outline': file_outline,
                    'workspace': workspace_info,
                    'suggestions': [
                        f"File truncated - showing first 500 of {total_lines} lines",
                        "Use start_line and end_line for specific sections",
                        f"File outline shows {len(file_outline)} major elements",
                        "Use get_function_content for specific functions"
                    ]
                }
            else:
                result = {
                    'file_path': file_path,
                    'resolved_path': resolved_path,
                    'file_type': file_type,
                    'total_lines': total_lines,
                    'content': [line.rstrip() for line in lines],
                    'truncated': False,
                    'outline': file_outline,
                    'workspace': workspace_info,
                    'suggestions': [
                        f"Complete file with {total_lines} lines",
                        f"File contains {len(file_outline)} major elements"
                    ]
                }

        return json.dumps(result, indent=2)

    except Exception as e:
        error_msg = f"Error reading file: {str(e)}"
        logger.error(error_msg)
        return json.dumps({
            'error': error_msg,
            'file_path': file_path,
            'workspace_id': workspace_id
        }, indent=2)


@mcp.tool()
async def list_workspaces(include_stats: bool = True) -> str:
    """
    List all registered workspaces with their information.

    Args:
        include_stats: Whether to include detailed statistics for each workspace

    Returns:
        JSON string with workspace list and metadata
    """
    try:
        if not global_client.registry:
            return json.dumps({
                'error': 'Registry not available'
            }, indent=2)

        workspaces = global_client.registry.list_workspaces()
        workspace_list = []

        for workspace in workspaces:
            workspace_info = {
                'id': workspace.id,
                'name': workspace.name,
                'description': workspace.description,
                'path': workspace.path,
                'project_type': workspace.project_type,
                'status': workspace.status,
                'tags': workspace.tags,
                'created_at': workspace.created_at,
                'last_indexed': workspace.last_indexed
            }

            if include_stats:
                workspace_info.update({
                    'file_count': workspace.file_count,
                    'document_count': workspace.document_count,
                    'code_symbol_count': workspace.code_symbol_count,
                    'embedding_count': workspace.embedding_count,
                    'index_size_mb': workspace.index_size_mb,
                    'health_score': workspace.health_score
                })

            workspace_list.append(workspace_info)

        # Get registry statistics
        registry_stats = global_client.registry.get_stats()

        result = {
            'total_workspaces': len(workspace_list),
            'registry_stats': {
                'total_documents': registry_stats.total_documents,
                'total_code_symbols': registry_stats.total_code_symbols,
                'total_size_mb': registry_stats.total_size_mb,
                'health_status': registry_stats.health_status
            },
            'workspaces': workspace_list
        }

        return json.dumps(result, indent=2)

    except Exception as e:
        error_msg = f"Error listing workspaces: {str(e)}"
        logger.error(error_msg)
        return json.dumps({
            'error': error_msg
        }, indent=2)


@mcp.tool()
async def get_workspace_info(workspace_id: str, include_files: bool = False) -> str:
    """
    Get detailed information about a specific workspace.

    Args:
        workspace_id: ID of the workspace to get information for
        include_files: Whether to include file list in the response

    Returns:
        JSON string with detailed workspace information
    """
    try:
        if not global_client.registry:
            return json.dumps({
                'error': 'Registry not available',
                'workspace_id': workspace_id
            }, indent=2)

        workspace = global_client.registry.get_workspace(workspace_id)
        if not workspace:
            return json.dumps({
                'error': f'Workspace not found: {workspace_id}',
                'workspace_id': workspace_id
            }, indent=2)

        result = {
            'id': workspace.id,
            'name': workspace.name,
            'description': workspace.description,
            'path': workspace.path,
            'project_type': workspace.project_type,
            'status': workspace.status,
            'health_score': workspace.health_score,
            'tags': workspace.tags,
            'created_at': workspace.created_at,
            'updated_at': workspace.updated_at,
            'last_indexed': workspace.last_indexed,
            'statistics': {
                'file_count': workspace.file_count,
                'document_count': workspace.document_count,
                'code_symbol_count': workspace.code_symbol_count,
                'embedding_count': workspace.embedding_count,
                'index_size_mb': workspace.index_size_mb
            }
        }

        # Get related workspaces
        relationships = global_client.registry.get_related_workspaces(workspace_id)
        if relationships:
            result['relationships'] = {}
            for related_id, rel_types in relationships.items():
                related_workspace = global_client.registry.get_workspace(related_id)
                result['relationships'][related_id] = {
                    'name': related_workspace.name if related_workspace else 'Unknown',
                    'relationship_types': rel_types
                }

        # Optionally include file list
        if include_files:
            search_engine = global_client._get_search_engine(workspace_id)
            if search_engine:
                try:
                    # Try to get file metadata from the search engine
                    file_metadata = getattr(search_engine, 'file_metadata', {})
                    result['files'] = list(file_metadata.keys())[:50]  # Limit to 50 files
                    if len(file_metadata) > 50:
                        result['files_truncated'] = True
                        result['total_files'] = len(file_metadata)
                except Exception as e:
                    result['files_error'] = f"Could not load file list: {e}"

        return json.dumps(result, indent=2)

    except Exception as e:
        error_msg = f"Error getting workspace info: {str(e)}"
        logger.error(error_msg)
        return json.dumps({
            'error': error_msg,
            'workspace_id': workspace_id
        }, indent=2)


@mcp.tool()
async def get_function_content(file_path: str, function_name: str, workspace_id: str = None) -> str:
    """
    Extract the complete content of a specific function from a file.

    Args:
        file_path: Path to the file containing the function
        function_name: Name of the function to extract
        workspace_id: Optional workspace ID to resolve relative paths

    Returns:
        JSON string with function content and metadata
    """
    try:
        resolved_path = file_path

        # Resolve path if workspace provided
        if workspace_id and global_client.registry:
            workspace = global_client.registry.get_workspace(workspace_id)
            if workspace and not Path(file_path).is_absolute():
                resolved_path = str(Path(workspace.path) / file_path)

        if not Path(resolved_path).exists():
            return json.dumps({
                'error': f'File not found: {resolved_path}',
                'file_path': file_path,
                'function_name': function_name
            }, indent=2)

        lines = global_client._read_file_content(resolved_path)
        if not lines:
            return json.dumps({
                'error': f'Could not read file: {resolved_path}',
                'file_path': file_path,
                'function_name': function_name
            }, indent=2)

        # Find function start
        func_start = None
        for i, line in enumerate(lines):
            # Look for function definition (supports multiple languages)
            if any(pattern in line for pattern in [f"def {function_name}(", f"function {function_name}(", f"fun {function_name}("]):
                func_start = i
                break

        if func_start is None:
            return json.dumps({
                'error': f'Function "{function_name}" not found in file',
                'file_path': file_path,
                'function_name': function_name,
                'suggestion': 'Check function name spelling and make sure it exists in the file'
            }, indent=2)

        # Find function end by indentation
        base_indent = len(lines[func_start]) - len(lines[func_start].lstrip())
        func_end = len(lines)

        for i in range(func_start + 1, len(lines)):
            line = lines[i]
            if line.strip():  # Skip empty lines
                current_indent = len(line) - len(line.lstrip())
                if current_indent <= base_indent:
                    # Check if this is another function/class definition
                    if any(keyword in line for keyword in ['def ', 'function ', 'fun ', 'class ']):
                        func_end = i
                        break

        # Extract function content
        function_lines = lines[func_start:func_end]

        result = {
            'file_path': file_path,
            'resolved_path': resolved_path,
            'function_name': function_name,
            'start_line': func_start + 1,
            'end_line': func_end,
            'line_count': len(function_lines),
            'content': [line.rstrip() for line in function_lines]
        }

        return json.dumps(result, indent=2)

    except Exception as e:
        error_msg = f"Error extracting function: {str(e)}"
        logger.error(error_msg)
        return json.dumps({
            'error': error_msg,
            'file_path': file_path,
            'function_name': function_name
        }, indent=2)


def _process_search_query(query: str) -> str:
    """Process search query to extract key terms and improve search"""
    # Remove common question words and phrases
    stop_phrases = [
        "how does", "how do", "how to", "what is", "what are", "where is", "where are",
        "why does", "why do", "when does", "when do", "can you", "please", "show me",
        "find", "search for", "look for", "get", "retrieve"
    ]

    processed = query.lower()

    # Remove stop phrases
    for phrase in stop_phrases:
        processed = processed.replace(phrase, " ")

    # Extract meaningful keywords
    words = [word.strip() for word in processed.split() if len(word.strip()) > 2]

    # Remove common stop words
    stop_words = {"the", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "from"}
    keywords = [word for word in words if word not in stop_words]

    # Return processed query or original if no meaningful keywords
    return " ".join(keywords) if keywords else query


def _suggest_keywords(query: str) -> str:
    """Suggest better keywords based on query"""
    keyword_map = {
        "configuration": "config",
        "initialize": "init setup",
        "authentication": "auth login",
        "database": "db data",
        "documentation": "docs readme",
        "function": "def method",
        "class": "class object",
        "error": "error exception",
        "setup": "init setup config",
        "work": "function method",
        "works": "function method"
    }

    suggestions = []
    words = query.lower().split()

    for word in words:
        if word in keyword_map:
            suggestions.append(keyword_map[word])
        elif len(word) > 3:
            suggestions.append(word)

    return " ".join(suggestions[:3]) if suggestions else "config setup init"


def _suggest_search_type(query: str) -> str:
    """Suggest appropriate search type based on query"""
    query_lower = query.lower()

    if any(word in query_lower for word in ["function", "class", "method", "code", "implement"]):
        return "code"
    elif any(word in query_lower for word in ["config", "setting", "setup", "json", "yaml"]):
        return "config"
    elif any(word in query_lower for word in ["docs", "documentation", "readme", "guide", "how"]):
        return "docs"
    else:
        return "all"


def _determine_file_type(file_ext: str) -> str:
    """Determine file type from extension"""
    code_exts = {'.py', '.js', '.ts', '.java', '.cpp', '.c', '.h', '.rs', '.go', '.kt'}
    doc_exts = {'.md', '.rst', '.txt', '.adoc'}
    config_exts = {'.json', '.yaml', '.yml', '.toml', '.ini', '.conf'}

    if file_ext in code_exts:
        return "code"
    elif file_ext in doc_exts:
        return "documentation"
    elif file_ext in config_exts:
        return "configuration"
    else:
        return "unknown"


def _generate_file_outline(lines: List[str], file_path: str) -> List[Dict[str, Any]]:
    """Generate file outline showing structure"""
    outline = []
    file_ext = Path(file_path).suffix.lower()

    for i, line in enumerate(lines, 1):
        line_stripped = line.strip()

        if file_ext == '.py':
            # Python functions and classes
            if line_stripped.startswith('class ') and ':' in line:
                class_name = line_stripped.split('class ')[1].split('(')[0].split(':')[0].strip()
                outline.append({
                    'type': 'class',
                    'name': class_name,
                    'line': i,
                    'content': line_stripped
                })
            elif line_stripped.startswith('def ') and ':' in line:
                func_name = line_stripped.split('def ')[1].split('(')[0].strip()
                outline.append({
                    'type': 'function',
                    'name': func_name,
                    'line': i,
                    'content': line_stripped
                })
        elif file_ext in ['.js', '.ts']:
            # JavaScript/TypeScript functions and classes
            if 'function ' in line or 'const ' in line and '=>' in line:
                outline.append({
                    'type': 'function',
                    'name': _extract_js_function_name(line_stripped),
                    'line': i,
                    'content': line_stripped
                })
            elif line_stripped.startswith('class '):
                class_name = line_stripped.split('class ')[1].split(' ')[0].split('{')[0].strip()
                outline.append({
                    'type': 'class',
                    'name': class_name,
                    'line': i,
                    'content': line_stripped
                })
        elif file_ext in ['.md', '.rst']:
            # Markdown/RST headers
            if line_stripped.startswith('#'):
                level = len(line_stripped) - len(line_stripped.lstrip('#'))
                title = line_stripped.lstrip('#').strip()
                outline.append({
                    'type': 'header',
                    'name': title,
                    'level': level,
                    'line': i,
                    'content': line_stripped
                })

        # Generic patterns for any file type
        if 'TODO' in line_stripped or 'FIXME' in line_stripped:
            outline.append({
                'type': 'todo',
                'name': line_stripped,
                'line': i,
                'content': line_stripped
            })

    return outline[:50]  # Limit to 50 items


def _extract_js_function_name(line: str) -> str:
    """Extract function name from JavaScript line"""
    try:
        if 'function ' in line:
            return line.split('function ')[1].split('(')[0].strip()
        elif 'const ' in line and '=>' in line:
            return line.split('const ')[1].split('=')[0].strip()
        elif 'let ' in line and '=>' in line:
            return line.split('let ')[1].split('=')[0].strip()
        else:
            return "anonymous"
    except:
        return "unknown"


if __name__ == "__main__":
    print("Starting Global Documentation MCP Server...")
    print(f"Registry path: {REGISTRY_PATH}")

    # Verify registry exists
    if not Path(REGISTRY_PATH).exists():
        print(f"Warning: Registry path does not exist: {REGISTRY_PATH}")
        print("Make sure to initialize some workspaces first with the mcpdata tool")

    # Start the server
    mcp.run()
