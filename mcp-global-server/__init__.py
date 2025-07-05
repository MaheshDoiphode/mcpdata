#!/usr/bin/env python3
"""
Global MCP Server for Documentation Indexing
Provides global search and content retrieval across all registered workspaces
"""

__version__ = "1.0.0"
__author__ = "MCP Team"

import sys
import json
import asyncio
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import asdict

# MCP imports
try:
    from mcp.server.models import InitializationOptions
    from mcp.server import NotificationOptions, Server
    from mcp.types import (
        Resource, Tool, TextContent, ImageContent, EmbeddedResource,
        LoggingLevel, CallToolResult, GetResourceResult, ListResourcesResult,
        ListToolsResult
    )
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False

# Add mcpdata to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "mcp"))

# Local imports
from mcp.core.registry import CentralRegistry, get_global_registry
from mcp.core.search import SearchEngine
from mcp.core.models import QueryResult

# Server configuration
SERVER_NAME = "mcp-global-docs"
SERVER_VERSION = "1.0.0"
DEFAULT_PORT = 8001

# Initialize logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class GlobalMCPServer:
    """Global MCP Server for documentation indexing"""

    def __init__(self, registry_path: Optional[Path] = None):
        """Initialize the global MCP server"""
        if not MCP_AVAILABLE:
            raise ImportError("MCP library not available. Please install: pip install mcp")

        self.registry_path = registry_path
        self.central_registry = CentralRegistry(registry_path) if registry_path else get_global_registry()
        self.server = Server(SERVER_NAME)
        self.search_engines: Dict[str, SearchEngine] = {}

        # Setup MCP server handlers
        self._setup_handlers()

    def _setup_handlers(self):
        """Setup MCP server handlers"""

        @self.server.list_tools()
        async def handle_list_tools() -> ListToolsResult:
            """List available tools"""
            tools = [
                Tool(
                    name="global_search",
                    description="Search across all registered workspaces",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Search query"
                            },
                            "max_results": {
                                "type": "integer",
                                "description": "Maximum number of results",
                                "default": 20
                            }
                        },
                        "required": ["query"]
                    }
                ),
                Tool(
                    name="get_content",
                    description="Get detailed content from a specific workspace",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "workspace_id": {
                                "type": "string",
                                "description": "Workspace ID"
                            },
                            "query": {
                                "type": "string",
                                "description": "Content query"
                            }
                        },
                        "required": ["workspace_id", "query"]
                    }
                ),
                Tool(
                    name="list_workspaces",
                    description="List all registered workspaces",
                    inputSchema={
                        "type": "object",
                        "properties": {}
                    }
                )
            ]
            return ListToolsResult(tools=tools)

        @self.server.call_tool()
        async def handle_call_tool(name: str, arguments: dict) -> CallToolResult:
            """Handle tool calls"""
            try:
                if name == "global_search":
                    return await self._handle_global_search(arguments)
                elif name == "get_content":
                    return await self._handle_get_content(arguments)
                elif name == "list_workspaces":
                    return await self._handle_list_workspaces(arguments)
                else:
                    raise ValueError(f"Unknown tool: {name}")
            except Exception as e:
                logger.error(f"Error handling tool {name}: {e}")
                return CallToolResult(
                    content=[
                        TextContent(
                            type="text",
                            text=f"Error: {str(e)}"
                        )
                    ]
                )

    async def _handle_global_search(self, arguments: dict) -> CallToolResult:
        """Handle global search across all workspaces"""
        query = arguments["query"]
        max_results = arguments.get("max_results", 20)

        # Perform global search
        results = self.central_registry.global_search(query, max_results)

        if not results:
            return CallToolResult(
                content=[
                    TextContent(
                        type="text",
                        text=f"No results found for query: {query}"
                    )
                ]
            )

        # Format response
        response_text = f"# Global Search Results for: {query}\n\n"
        response_text += f"Found {len(results)} results\n\n"

        for i, result in enumerate(results, 1):
            response_text += f"## {i}. {result.title}\n"
            response_text += f"**Workspace:** {result.workspace_name}\n"
            response_text += f"**File:** {result.file_path}\n"
            response_text += f"**Type:** {result.section_type}\n"
            response_text += f"**Score:** {result.relevance_score:.2f}\n"
            response_text += f"**Preview:** {result.content_preview}\n\n"

        return CallToolResult(
            content=[
                TextContent(
                    type="text",
                    text=response_text
                )
            ]
        )

    async def _handle_get_content(self, arguments: dict) -> CallToolResult:
        """Handle detailed content retrieval from a workspace"""
        workspace_id = arguments["workspace_id"]
        query = arguments["query"]

        workspace = self.central_registry.get_workspace(workspace_id)
        if not workspace:
            raise ValueError(f"Workspace not found: {workspace_id}")

        # Get workspace search engine
        search_engine = await self._get_workspace_search_engine(workspace)

        # Perform detailed search
        try:
            context = search_engine.get_optimized_context(query)

            response_text = f"# Content from {workspace.name}\n\n"
            response_text += f"**Query:** {query}\n"
            response_text += f"**Workspace:** {workspace.name}\n\n"
            response_text += "## Content\n\n"
            response_text += context

            return CallToolResult(
                content=[
                    TextContent(
                        type="text",
                        text=response_text
                    )
                ]
            )
        except Exception as e:
            logger.error(f"Error retrieving content from workspace {workspace_id}: {e}")
            return CallToolResult(
                content=[
                    TextContent(
                        type="text",
                        text=f"Error retrieving content: {str(e)}"
                    )
                ]
            )

    async def _handle_list_workspaces(self, arguments: dict) -> CallToolResult:
        """Handle workspace listing"""
        workspaces = self.central_registry.list_workspaces()

        response_text = f"# Registered Workspaces\n\n"
        response_text += f"Found {len(workspaces)} workspaces\n\n"

        for workspace in workspaces:
            response_text += f"## {workspace.name}\n"
            response_text += f"**ID:** {workspace.id}\n"
            response_text += f"**Description:** {workspace.description}\n"
            response_text += f"**Type:** {workspace.project_type}\n"
            response_text += f"**Documents:** {workspace.document_count}\n"
            response_text += f"**Status:** {workspace.status}\n\n"

        return CallToolResult(
            content=[
                TextContent(
                    type="text",
                    text=response_text
                )
            ]
        )

    async def _get_workspace_search_engine(self, workspace) -> SearchEngine:
        """Get or create a search engine for a workspace"""
        if workspace.id not in self.search_engines:
            workspace_path = Path(workspace.path)
            mcpdata_path = workspace_path / '.mcpdata'

            if not mcpdata_path.exists():
                raise FileNotFoundError(f"No .mcpdata found in {workspace_path}")

            search_engine = SearchEngine(mcpdata_path)
            self.search_engines[workspace.id] = search_engine

        return self.search_engines[workspace.id]

    async def run(self, transport_type: str = "stdio"):
        """Run the server"""
        logger.info(f"Starting {SERVER_NAME} v{SERVER_VERSION}")

        if transport_type == "stdio":
            async with self.server.run_stdio() as streams:
                await self.server.run(
                    streams[0], streams[1],
                    InitializationOptions(
                        server_name=SERVER_NAME,
                        server_version=SERVER_VERSION,
                        capabilities=self.server.get_capabilities(
                            notification_options=NotificationOptions(),
                            experimental_capabilities={}
                        )
                    )
                )
        else:
            raise ValueError(f"Unsupported transport type: {transport_type}")


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description="Global MCP Server for Documentation")
    parser.add_argument(
        "--registry-path",
        type=str,
        help="Path to central registry (default: ~/Documents/mcpdata)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Create server
    registry_path = Path(args.registry_path) if args.registry_path else None
    server = GlobalMCPServer(registry_path)

    # Run server
    try:
        asyncio.run(server.run())
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Server error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
