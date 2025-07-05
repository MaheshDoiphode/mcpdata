#!/usr/bin/env python3
"""
Simple test of the knowledge graph implementation
"""

import sys
import json
import time
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from mcpdata.core.knowledge_graph import get_knowledge_graph


def main():
    """Main test function"""
    
    print("=== Knowledge Graph Implementation Test ===\n")
    
    # Initialize knowledge graph
    knowledge_graph = get_knowledge_graph()
    print("✓ Knowledge graph initialized successfully")
    
    # Test 1: Basic request tracking
    print("\n1. Testing basic request tracking...")
    
    req1 = knowledge_graph.start_request(
        tool_name="search_workspaces", 
        arguments={"query": "authentication", "search_type": "code"},
        connection_id="test_session"
    )
    print(f"   Started request: {req1}")
    
    time.sleep(0.01)
    knowledge_graph.complete_request(req1, "Found 5 authentication files")
    print(f"   Completed request: {req1}")
    
    # Test 2: Request relationships
    print("\n2. Testing request relationships...")
    
    req2 = knowledge_graph.start_request(
        tool_name="get_file_content",
        arguments={"file_path": "src/auth.py"},
        connection_id="test_session"
    )
    
    knowledge_graph.add_relationship(
        source_request_id=req1,
        target_request_id=req2,
        relationship_type="sequential",
        metadata={"trigger": "file_found_in_search"}
    )
    
    time.sleep(0.01)
    knowledge_graph.complete_request(req2, "Retrieved 150 lines of code")
    print(f"   Added relationship: {req1} -> {req2}")
    
    # Test 3: Request flow analysis
    print("\n3. Testing request flow analysis...")
    
    flow = knowledge_graph.get_request_flow(req1)
    print(f"   Flow analysis for {req1}:")
    print(f"   - Total nodes: {flow['total_nodes']}")
    print(f"   - Total edges: {flow['total_edges']}")
    print(f"   - Request chain: {flow['request_chain']}")
    
    # Test 4: Usage patterns
    print("\n4. Testing usage patterns...")
    
    patterns = knowledge_graph.get_tool_usage_patterns()
    print(f"   Tool usage patterns:")
    print(f"   - Total requests: {patterns['total_requests']}")
    print(f"   - Tools used: {list(patterns['tool_usage'].keys())}")
    print(f"   - Common sequences: {list(patterns['common_sequences'].keys())}")
    
    # Test 5: Parallel request simulation
    print("\n5. Testing parallel request simulation...")
    
    # Create multiple requests quickly to simulate parallel execution
    parallel_reqs = []
    for i in range(3):
        req = knowledge_graph.start_request(
            tool_name="get_file_content",
            arguments={"file_path": f"src/module{i}.py"},
            connection_id="test_session"
        )
        parallel_reqs.append(req)
        
        # Link to original search
        knowledge_graph.add_relationship(
            source_request_id=req1,
            target_request_id=req,
            relationship_type="parallel",
            metadata={"trigger": "parallel_file_retrieval"}
        )
    
    # Complete parallel requests
    for i, req in enumerate(parallel_reqs):
        knowledge_graph.complete_request(req, f"Retrieved module {i}")
    
    print(f"   Created {len(parallel_reqs)} parallel requests")
    
    # Test 6: Updated flow analysis
    print("\n6. Testing updated flow analysis...")
    
    updated_flow = knowledge_graph.get_request_flow(req1)
    print(f"   Updated flow analysis:")
    print(f"   - Total nodes: {updated_flow['total_nodes']}")
    print(f"   - Total edges: {updated_flow['total_edges']}")
    print(f"   - Parallel groups: {len(updated_flow['parallel_requests'])}")
    
    # Test 7: Error handling
    print("\n7. Testing error handling...")
    
    error_req = knowledge_graph.start_request(
        tool_name="search_workspaces",
        arguments={"query": "invalid"},
        connection_id="test_session"
    )
    
    knowledge_graph.complete_request(
        request_id=error_req,
        error_message="Registry not available"
    )
    
    error_flow = knowledge_graph.get_request_flow(error_req)
    error_node = error_flow['nodes'][0]
    
    if error_node['status'] == 'failed':
        print(f"   ✓ Error tracking works correctly")
    else:
        print(f"   ✗ Error tracking failed")
        return False
    
    # Test 8: JSON serialization
    print("\n8. Testing JSON serialization...")
    
    try:
        json_output = json.dumps(updated_flow, indent=2)
        print(f"   ✓ JSON serialization works ({len(json_output)} characters)")
    except Exception as e:
        print(f"   ✗ JSON serialization failed: {e}")
        return False
    
    # Test 9: Data persistence
    print("\n9. Testing data persistence...")
    
    try:
        knowledge_graph.save_data()
        print(f"   ✓ Data saved successfully")
    except Exception as e:
        print(f"   ✗ Data save failed: {e}")
        return False
    
    # Final summary
    print("\n=== Test Summary ===")
    final_patterns = knowledge_graph.get_tool_usage_patterns()
    print(f"Total requests tracked: {final_patterns['total_requests']}")
    print(f"Tools used: {list(final_patterns['tool_usage'].keys())}")
    print(f"Error rate: {sum(stats['errors'] for stats in final_patterns['error_rates'].values())} errors")
    
    print("\n✅ All tests passed! Knowledge graph is working correctly.")
    
    # Show sample output for documentation
    print("\n=== Sample Output ===")
    print("Request flow analysis:")
    print(json.dumps(updated_flow, indent=2)[:500] + "...")
    
    return True


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)