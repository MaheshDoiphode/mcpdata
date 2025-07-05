#!/usr/bin/env python3
"""
Test the knowledge graph functionality with request flow tracking
"""

import sys
import json
import time
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from mcpdata.core.knowledge_graph import get_knowledge_graph


def test_knowledge_graph_functionality():
    """Test the knowledge graph functionality"""
    
    print("Testing knowledge graph functionality...")
    
    knowledge_graph = get_knowledge_graph()
    
    # Test 1: Start a request
    print("\n1. Testing request tracking...")
    request_id_1 = knowledge_graph.start_request(
        tool_name="search_workspaces",
        arguments={"query": "test", "search_type": "all"},
        connection_id="test_connection"
    )
    print(f"Started request 1: {request_id_1}")
    
    # Test 2: Start a related request
    request_id_2 = knowledge_graph.start_request(
        tool_name="get_file_content",
        arguments={"file_path": "/test/file.py"},
        connection_id="test_connection"
    )
    print(f"Started request 2: {request_id_2}")
    
    # Test 3: Add a relationship
    knowledge_graph.add_relationship(
        source_request_id=request_id_1,
        target_request_id=request_id_2,
        relationship_type="sequential",
        metadata={"trigger": "file_found_in_search"}
    )
    print("Added relationship between requests")
    
    # Test 4: Complete the requests
    knowledge_graph.complete_request(
        request_id=request_id_1,
        result_summary="Found 3 files matching the search"
    )
    print("Completed request 1")
    
    time.sleep(0.1)  # Small delay to show different timestamps
    
    knowledge_graph.complete_request(
        request_id=request_id_2,
        result_summary="Retrieved file content (150 lines)"
    )
    print("Completed request 2")
    
    # Test 5: Test parallel requests
    print("\n2. Testing parallel requests...")
    parallel_requests = []
    for i in range(3):
        req_id = knowledge_graph.start_request(
            tool_name="get_file_content",
            arguments={"file_path": f"/test/file{i}.py"},
            connection_id="test_connection"
        )
        parallel_requests.append(req_id)
        knowledge_graph.complete_request(req_id, f"Read file {i}")
    
    print(f"Created {len(parallel_requests)} parallel requests")
    
    # Test 6: Get request flow
    print("\n3. Testing request flow analysis...")
    flow_data = knowledge_graph.get_request_flow(request_id_1)
    print(f"Request flow for {request_id_1}:")
    print(json.dumps(flow_data, indent=2))
    
    # Test 7: Get tool usage patterns
    print("\n4. Testing tool usage patterns...")
    patterns = knowledge_graph.get_tool_usage_patterns()
    print("Tool usage patterns:")
    print(json.dumps(patterns, indent=2))
    
    # Test 8: Test search patterns
    print("\n5. Testing search patterns...")
    search_patterns = knowledge_graph.get_tool_usage_patterns(tool_name="search_workspaces")
    print("Search-specific patterns:")
    print(json.dumps(search_patterns, indent=2))
    
    print("\n✓ All knowledge graph tests passed!")
    return True


def test_knowledge_graph_persistence():
    """Test knowledge graph data persistence"""
    
    print("\nTesting knowledge graph persistence...")
    
    knowledge_graph = get_knowledge_graph()
    
    # Save current state
    knowledge_graph.save_data()
    print("Saved knowledge graph data")
    
    # Test cleanup
    cleaned_count = knowledge_graph.cleanup_old_data(max_age_days=0)
    print(f"Cleaned up {cleaned_count} old requests")
    
    print("✓ Persistence tests passed!")
    return True


def test_request_flow_simulation():
    """Simulate a realistic request flow scenario"""
    
    print("\nSimulating realistic request flow...")
    
    knowledge_graph = get_knowledge_graph()
    
    # Simulate AI asking: "How does authentication work in this codebase?"
    # This would trigger multiple MCP tool calls
    
    # 1. First, search for authentication-related code
    search_req = knowledge_graph.start_request(
        tool_name="search_workspaces",
        arguments={"query": "authentication", "search_type": "code"},
        connection_id="ai_session_1"
    )
    
    # 2. Then get specific files found in search
    file_requests = []
    for i, file_path in enumerate(["src/auth.py", "src/user.py", "src/middleware.py"]):
        file_req = knowledge_graph.start_request(
            tool_name="get_file_content",
            arguments={"file_path": file_path},
            connection_id="ai_session_1"
        )
        file_requests.append(file_req)
        
        # Link to search request
        knowledge_graph.add_relationship(
            source_request_id=search_req,
            target_request_id=file_req,
            relationship_type="sequential",
            metadata={"trigger": "file_found_in_search", "file_index": i}
        )
    
    # 3. Then get specific function content
    func_req = knowledge_graph.start_request(
        tool_name="get_function_content",
        arguments={"file_path": "src/auth.py", "function_name": "authenticate_user"},
        connection_id="ai_session_1"
    )
    
    # Link to file request
    knowledge_graph.add_relationship(
        source_request_id=file_requests[0],
        target_request_id=func_req,
        relationship_type="sequential",
        metadata={"trigger": "function_analysis"}
    )
    
    # Complete all requests
    knowledge_graph.complete_request(search_req, "Found 5 authentication-related files")
    for i, file_req in enumerate(file_requests):
        knowledge_graph.complete_request(file_req, f"Retrieved file content ({100 + i * 50} lines)")
    knowledge_graph.complete_request(func_req, "Retrieved function implementation")
    
    # Analyze the flow
    flow_data = knowledge_graph.get_request_flow(search_req)
    print(f"Complete request flow analysis:")
    print(f"- Total nodes: {flow_data['total_nodes']}")
    print(f"- Total edges: {flow_data['total_edges']}")
    print(f"- Request chain length: {len(flow_data['request_chain'])}")
    print(f"- Parallel requests: {len(flow_data['parallel_requests'])}")
    
    # Show timeline
    print("\nRequest timeline:")
    for item in flow_data['flow_timeline']:
        print(f"  {item['timestamp']:.3f}: {item['tool_name']} - {item['status']}")
    
    print("✓ Request flow simulation passed!")
    return True


if __name__ == "__main__":
    try:
        success = (
            test_knowledge_graph_functionality() and
            test_knowledge_graph_persistence() and
            test_request_flow_simulation()
        )
        print(f"\n{'='*50}")
        print(f"Overall test result: {'PASSED' if success else 'FAILED'}")
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)