# Knowledge Graph for Request Flow Tracking

The knowledge graph feature tracks how AI requests flow through the MCP server system, enabling analysis of request patterns, dependencies, and performance metrics.

## Overview

When AI assistants interact with the MCP server, they typically make multiple related requests:
1. Search for relevant content
2. Get specific files based on search results  
3. Extract functions or sections from files
4. Analyze related workspaces

The knowledge graph captures these request flows to provide insights into:
- Sequential request chains (A → B → C)
- Parallel request patterns (A → [B, C, D])
- Tool usage patterns and performance
- Error rates and debugging information

## Architecture

### Core Components

**RequestNode**: Represents a single request
```python
@dataclass
class RequestNode:
    request_id: str
    tool_name: str
    arguments: Dict[str, Any]
    timestamp: float
    duration: Optional[float]
    status: str  # pending, completed, failed
    result_summary: Optional[str]
    workspace_id: Optional[str]
    connection_id: Optional[str]
    error_message: Optional[str]
```

**RequestEdge**: Represents relationships between requests
```python
@dataclass
class RequestEdge:
    source_request_id: str
    target_request_id: str
    relationship_type: str  # sequential, parallel, triggered_by, depends_on
    timestamp: float
    metadata: Dict[str, Any]
```

**KnowledgeGraph**: Main class managing the graph
- Thread-safe request tracking
- Relationship management
- Flow analysis and pattern detection
- Data persistence and cleanup

## New MCP Tools

### get_request_flow

Analyze the complete request flow for a specific request.

```bash
# Get flow for specific request
get_request_flow(request_id="abc123", max_depth=10)

# Get recent flows and patterns
get_request_flow()
```

**Returns:**
```json
{
  "root_request_id": "abc123",
  "total_nodes": 5,
  "total_edges": 4,
  "flow_timeline": [
    {
      "request_id": "abc123",
      "tool_name": "search_workspaces",
      "timestamp": 1677123456.789,
      "duration": 0.15,
      "status": "completed",
      "arguments": {"query": "authentication", "search_type": "code"}
    }
  ],
  "request_chain": ["abc123", "def456", "ghi789"],
  "parallel_requests": [["jkl012", "mno345"]],
  "nodes": [...],
  "edges": [...]
}
```

### get_knowledge_graph_stats

Get comprehensive statistics about request patterns and tool usage.

```bash
get_knowledge_graph_stats()
```

**Returns:**
```json
{
  "tracking_summary": {
    "total_requests": 1247,
    "total_relationships": 890,
    "active_sessions": 3
  },
  "tool_usage": {
    "search_workspaces": 345,
    "get_file_content": 567,
    "get_function_content": 123
  },
  "performance": {
    "avg_duration_by_tool": {
      "search_workspaces": 0.25,
      "get_file_content": 0.12
    },
    "error_rates": {
      "search_workspaces": {"total": 345, "errors": 12, "error_rate": 0.035}
    }
  },
  "request_patterns": {
    "common_sequences": {
      "search_workspaces -> get_file_content": 234,
      "get_file_content -> get_function_content": 89
    }
  }
}
```

## Request Flow Examples

### Example 1: Authentication Analysis Flow

**User Query**: "How does authentication work in this codebase?"

**AI Request Flow**:
1. `search_workspaces(query="authentication", search_type="code")`
2. `get_file_content(file_path="src/auth.py")` ← Sequential from #1
3. `get_file_content(file_path="src/user.py")` ← Sequential from #1  
4. `get_function_content(file_path="src/auth.py", function_name="authenticate_user")` ← Sequential from #2

**Knowledge Graph Output**:
```json
{
  "request_chain": ["search_1", "file_1", "function_1"],
  "parallel_requests": [["file_1", "file_2"]],
  "flow_timeline": [
    {"tool_name": "search_workspaces", "timestamp": 1677123456.1},
    {"tool_name": "get_file_content", "timestamp": 1677123456.3},
    {"tool_name": "get_file_content", "timestamp": 1677123456.3},
    {"tool_name": "get_function_content", "timestamp": 1677123456.5}
  ]
}
```

### Example 2: Configuration Discovery Flow

**User Query**: "Show me the database configuration"

**AI Request Flow**:
1. `search_workspaces(query="database config", search_type="config")`
2. `get_file_content(file_path="config/database.yml")` ← Sequential from #1
3. `search_workspaces(query="database connection", search_type="code")` ← Parallel exploration
4. `get_file_content(file_path="src/db.py")` ← Sequential from #3

## Integration

### Automatic Tracking

All existing MCP tools automatically track requests:

```python
# In search_workspaces tool
knowledge_graph = get_knowledge_graph()
request_id = knowledge_graph.start_request(
    tool_name="search_workspaces",
    arguments={"query": query, "search_type": search_type},
    workspace_id=workspace_id
)

# ... process request ...

knowledge_graph.complete_request(
    request_id=request_id,
    result_summary=f"Found {len(results)} results"
)
```

### Relationship Detection

```python
# Automatic relationship when one request triggers another
if parent_request_id:
    knowledge_graph.add_relationship(
        source_request_id=parent_request_id,
        target_request_id=request_id,
        relationship_type="sequential",
        metadata={"trigger": "file_found_in_search"}
    )
```

## Use Cases

### For AI Assistants
- Understand request flow patterns for better question answering
- Identify when to make parallel vs sequential requests
- Learn from successful request chains

### For Developers  
- Debug MCP tool usage patterns
- Optimize tool performance based on usage data
- Understand user interaction patterns

### For System Administrators
- Monitor MCP server performance
- Track error patterns and failures
- Capacity planning based on usage trends

## Configuration

The knowledge graph is automatically initialized and requires no configuration. Data is stored in:
```
~/Documents/mcpdata/knowledge_graph/
├── nodes.json     # Request nodes
├── edges.json     # Relationships  
└── flows.json     # Complete flows
```

### Cleanup

Old request data is automatically cleaned up:
```python
# Clean up requests older than 30 days
knowledge_graph.cleanup_old_data(max_age_days=30)
```

## Performance

- **Thread-safe**: Handles concurrent requests safely
- **Memory efficient**: Automatic cleanup of old data
- **Fast queries**: In-memory indices for quick analysis
- **Minimal overhead**: < 1ms per request tracking

## Future Enhancements

- Visual request flow diagrams
- Machine learning for request prediction
- Advanced pattern recognition
- Integration with monitoring systems
- Export to external analytics tools