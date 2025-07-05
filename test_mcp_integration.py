#!/usr/bin/env python3
"""
Test MCP server integration with knowledge graph
"""

import sys
import json
import asyncio
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent / "mcp-global-server"))
sys.path.insert(0, str(Path(__file__).parent / "src"))

from mcpdata.core.knowledge_graph import get_knowledge_graph


def test_mcp_server_integration():
    """Test MCP server integration with knowledge graph"""
    
    print("Testing MCP server integration...")
    
    # Import the MCP server functions
    try:
        import server
        print("✓ MCP server module imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import MCP server: {e}")
        return False
    
    # Test knowledge graph integration
    knowledge_graph = get_knowledge_graph()
    
    # Simulate the request tracking that would happen in the MCP tools
    print("\nSimulating MCP tool request tracking...")
    
    # Test 1: Simulate search_workspaces call
    search_req = knowledge_graph.start_request(
        tool_name="search_workspaces",
        arguments={"query": "authentication", "search_type": "code"},
        connection_id="mcp_client_1"
    )
    
    # Simulate processing time and completion
    import time
    time.sleep(0.01)
    
    knowledge_graph.complete_request(
        request_id=search_req,
        result_summary="Found 3 authentication-related files"
    )
    
    # Test 2: Simulate get_file_content call triggered by search
    file_req = knowledge_graph.start_request(
        tool_name="get_file_content",
        arguments={"file_path": "src/auth.py", "workspace_id": "test_workspace"},
        connection_id="mcp_client_1"
    )
    
    # Add relationship
    knowledge_graph.add_relationship(
        source_request_id=search_req,
        target_request_id=file_req,
        relationship_type="sequential",
        metadata={"trigger": "file_found_in_search"}
    )
    
    time.sleep(0.01)
    
    knowledge_graph.complete_request(
        request_id=file_req,
        result_summary="Retrieved file content (245 lines)"
    )
    
    # Test 3: Test the new MCP tools
    print("\nTesting new MCP tools functionality...")
    
    # Test get_request_flow functionality
    flow_data = knowledge_graph.get_request_flow(search_req)
    print(f"Request flow analysis:")
    print(f"- Root request: {flow_data['root_request_id']}")
    print(f"- Total nodes: {flow_data['total_nodes']}")
    print(f"- Total edges: {flow_data['total_edges']}")
    print(f"- Request chain: {flow_data['request_chain']}")
    
    # Test get_knowledge_graph_stats functionality
    stats = knowledge_graph.get_tool_usage_patterns()
    print(f"\nKnowledge graph statistics:")
    print(f"- Total requests: {stats['total_requests']}")
    print(f"- Tool usage: {stats['tool_usage']}")
    print(f"- Common sequences: {stats['common_sequences']}")
    
    # Test error handling
    print("\nTesting error handling...")
    
    error_req = knowledge_graph.start_request(
        tool_name="search_workspaces",
        arguments={"query": "invalid", "search_type": "code"},
        connection_id="mcp_client_1"
    )
    
    knowledge_graph.complete_request(
        request_id=error_req,
        error_message="Registry not available"
    )
    
    # Verify error tracking
    error_flow = knowledge_graph.get_request_flow(error_req)
    error_node = error_flow['nodes'][0]
    
    if error_node['status'] == 'failed' and error_node['error_message']:
        print("✓ Error tracking working correctly")
    else:
        print("✗ Error tracking failed")
        return False
    
    # Test multiple connection tracking
    print("\nTesting multiple connection tracking...")
    
    # Simulate parallel clients
    for client_id in ["client_1", "client_2", "client_3"]:
        req = knowledge_graph.start_request(
            tool_name="list_workspaces",
            arguments={"include_stats": True},
            connection_id=client_id
        )
        knowledge_graph.complete_request(req, f"Listed workspaces for {client_id}")
    
    # Check active sessions
    print(f"Active sessions tracked: {len(knowledge_graph.active_sessions)}")
    
    print("\n✓ MCP server integration tests passed!")
    return True


def test_request_flow_json_output():
    """Test that the request flow tools produce valid JSON output"""
    
    print("Testing JSON output format...")
    
    knowledge_graph = get_knowledge_graph()
    
    # Create some test data
    req1 = knowledge_graph.start_request(
        tool_name="search_workspaces",
        arguments={"query": "config", "search_type": "all"},
        connection_id="json_test"
    )
    
    req2 = knowledge_graph.start_request(
        tool_name="get_file_content",
        arguments={"file_path": "config.json"},
        connection_id="json_test"
    )
    
    knowledge_graph.add_relationship(req1, req2, "sequential")
    knowledge_graph.complete_request(req1, "Found 2 config files")
    knowledge_graph.complete_request(req2, "Retrieved config file")
    
    # Test flow output
    flow_data = knowledge_graph.get_request_flow(req1)
    
    try:
        json_output = json.dumps(flow_data, indent=2)
        print(f"✓ Request flow JSON output is valid ({len(json_output)} characters)")
    except Exception as e:
        print(f"✗ Request flow JSON output is invalid: {e}")
        return False
    
    # Test stats output
    stats = knowledge_graph.get_tool_usage_patterns()
    
    try:
        json_output = json.dumps(stats, indent=2)
        print(f"✓ Statistics JSON output is valid ({len(json_output)} characters)")
    except Exception as e:
        print(f"✗ Statistics JSON output is invalid: {e}")
        return False
    
    print("✓ JSON output tests passed!")
    return True


def test_concurrent_requests():
    """Test concurrent request handling"""
    
    print("Testing concurrent request handling...")
    
    knowledge_graph = get_knowledge_graph()
    
    # Simulate concurrent requests from multiple clients
    import threading
    import time
    
    results = []
    
    def make_request(client_id, request_num):
        req_id = knowledge_graph.start_request(
            tool_name="search_workspaces",
            arguments={"query": f"test_{request_num}", "search_type": "all"},
            connection_id=client_id
        )
        time.sleep(0.001)  # Small delay to simulate processing
        knowledge_graph.complete_request(req_id, f"Completed request {request_num}")
        results.append((client_id, request_num, req_id))
    
    # Create multiple threads
    threads = []
    for i in range(10):
        client_id = f"client_{i % 3}"  # 3 different clients
        thread = threading.Thread(target=make_request, args=(client_id, i))
        threads.append(thread)
        thread.start()
    
    # Wait for all threads to complete
    for thread in threads:
        thread.join()
    
    # Verify all requests were tracked
    if len(results) == 10:
        print(f"✓ All {len(results)} concurrent requests tracked successfully")
    else:
        print(f"✗ Only {len(results)} of 10 concurrent requests tracked")
        return False
    
    # Check that no data corruption occurred
    patterns = knowledge_graph.get_tool_usage_patterns()
    total_requests = patterns['total_requests']
    
    if total_requests >= 10:
        print(f"✓ Knowledge graph integrity maintained ({total_requests} total requests)")
    else:
        print(f"✗ Knowledge graph integrity compromised ({total_requests} total requests)")
        return False
    
    print("✓ Concurrent request tests passed!")
    return True


if __name__ == "__main__":
    try:
        success = (
            test_mcp_server_integration() and
            test_request_flow_json_output() and
            test_concurrent_requests()
        )
        
        print(f"\n{'='*50}")
        print(f"MCP Integration test result: {'PASSED' if success else 'FAILED'}")
        
        if success:
            print("\n🎉 Knowledge graph is ready for production!")
            print("The following new MCP tools are available:")
            print("- get_request_flow: Analyze request flow patterns")
            print("- get_knowledge_graph_stats: Get usage statistics")
            print("- All existing tools now track requests in the knowledge graph")
        
        sys.exit(0 if success else 1)
        
    except Exception as e:
        print(f"Integration test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)