#!/usr/bin/env python3
"""
Test the knowledge graph integration with the MCP server
"""

import sys
import json
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from mcpdata.server import QueryRequest, MCPServer
from mcpdata.core.knowledge_graph import get_knowledge_graph
from mcpdata.core.config import ConfigManager


def test_knowledge_graph_integration():
    """Test the knowledge graph integration with the MCP server"""
    
    # Create a temporary directory for testing
    test_dir = Path("/tmp/mcpdata_test")
    test_dir.mkdir(exist_ok=True)
    
    # Create test configuration
    config_manager = ConfigManager(test_dir / "config.json")
    config = config_manager.load_config()
    
    # Create test server
    server = MCPServer(test_dir, config.search)
    
    # Test query with knowledge graph tracking
    request = QueryRequest(
        query="test query",
        search_type="keyword",
        max_tokens=100,
        connection_id="test_connection"
    )
    
    print("Testing synchronous query with knowledge graph tracking...")
    try:
        response = server.query_sync(request)
        print(f"Response received with request_id: {response.request_id}")
        print(f"Response content length: {len(response.content)}")
        
        # Test getting the request flow
        knowledge_graph = get_knowledge_graph()
        if response.request_id:
            flow = knowledge_graph.get_request_flow(response.request_id)
            print(f"Request flow tracked: {json.dumps(flow, indent=2)}")
        
        # Test tool usage patterns
        patterns = knowledge_graph.get_tool_usage_patterns()
        print(f"Tool usage patterns: {json.dumps(patterns, indent=2)}")
        
        print("✓ Knowledge graph integration test passed!")
        return True
        
    except Exception as e:
        print(f"✗ Knowledge graph integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_knowledge_graph_integration()
    sys.exit(0 if success else 1)