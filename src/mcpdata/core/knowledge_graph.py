"""
Knowledge Graph for tracking MCP request flows

This module implements a knowledge graph to track how AI requests flow through
the MCP server system, enabling analysis of request patterns and dependencies.
"""

import json
import time
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass, asdict, field
from datetime import datetime
from collections import defaultdict, deque
from threading import Lock


@dataclass
class RequestNode:
    """Represents a single request in the knowledge graph"""
    request_id: str
    tool_name: str
    arguments: Dict[str, Any]
    timestamp: float
    duration: Optional[float] = None
    status: str = "pending"  # pending, completed, failed
    result_summary: Optional[str] = None
    workspace_id: Optional[str] = None
    connection_id: Optional[str] = None
    error_message: Optional[str] = None
    
    def __post_init__(self):
        if not self.request_id:
            # Generate unique request ID
            content = f"{self.tool_name}:{self.timestamp}:{id(self)}"
            self.request_id = hashlib.md5(content.encode()).hexdigest()[:12]


@dataclass
class RequestEdge:
    """Represents a relationship between requests"""
    source_request_id: str
    target_request_id: str
    relationship_type: str  # 'sequential', 'parallel', 'triggered_by', 'depends_on'
    timestamp: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RequestFlow:
    """Represents a complete flow of requests"""
    flow_id: str
    session_id: Optional[str]
    start_time: float
    end_time: Optional[float]
    nodes: List[RequestNode]
    edges: List[RequestEdge]
    total_duration: Optional[float] = None
    status: str = "active"  # active, completed, failed
    
    def __post_init__(self):
        if not self.flow_id:
            # Generate unique flow ID
            content = f"{self.session_id}:{self.start_time}:{len(self.nodes)}"
            self.flow_id = hashlib.md5(content.encode()).hexdigest()[:12]


class KnowledgeGraph:
    """Knowledge graph for tracking MCP request flows"""
    
    def __init__(self, storage_path: Optional[Path] = None):
        """Initialize the knowledge graph"""
        self.storage_path = storage_path or Path.home() / "Documents" / "mcpdata" / "knowledge_graph"
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # In-memory storage
        self.nodes: Dict[str, RequestNode] = {}
        self.edges: List[RequestEdge] = []
        self.flows: Dict[str, RequestFlow] = {}
        self.active_sessions: Dict[str, List[str]] = defaultdict(list)  # session_id -> request_ids
        
        # Thread safety
        self._lock = Lock()
        
        # Load existing data
        self._load_data()
    
    def start_request(self, tool_name: str, arguments: Dict[str, Any], 
                     connection_id: Optional[str] = None,
                     workspace_id: Optional[str] = None) -> str:
        """Start tracking a new request"""
        with self._lock:
            node = RequestNode(
                request_id="",  # Will be generated in __post_init__
                tool_name=tool_name,
                arguments=arguments.copy(),
                timestamp=time.time(),
                workspace_id=workspace_id,
                connection_id=connection_id
            )
            
            self.nodes[node.request_id] = node
            
            # Track in active session
            if connection_id:
                self.active_sessions[connection_id].append(node.request_id)
            
            return node.request_id
    
    def complete_request(self, request_id: str, result_summary: Optional[str] = None,
                        error_message: Optional[str] = None) -> None:
        """Mark a request as completed"""
        with self._lock:
            if request_id not in self.nodes:
                return
            
            node = self.nodes[request_id]
            node.duration = time.time() - node.timestamp
            node.status = "failed" if error_message else "completed"
            node.result_summary = result_summary
            node.error_message = error_message
    
    def add_relationship(self, source_request_id: str, target_request_id: str,
                        relationship_type: str, metadata: Dict[str, Any] = None) -> None:
        """Add a relationship between two requests"""
        with self._lock:
            if source_request_id not in self.nodes or target_request_id not in self.nodes:
                return
            
            edge = RequestEdge(
                source_request_id=source_request_id,
                target_request_id=target_request_id,
                relationship_type=relationship_type,
                timestamp=time.time(),
                metadata=metadata or {}
            )
            
            self.edges.append(edge)
    
    def get_request_flow(self, request_id: str, max_depth: int = 10) -> Dict[str, Any]:
        """Get the complete flow for a request"""
        with self._lock:
            if request_id not in self.nodes:
                return {"error": f"Request {request_id} not found"}
            
            # Build flow graph using BFS
            visited = set()
            queue = deque([request_id])
            flow_nodes = []
            flow_edges = []
            
            while queue and len(flow_nodes) < max_depth:
                current_id = queue.popleft()
                if current_id in visited:
                    continue
                
                visited.add(current_id)
                if current_id in self.nodes:
                    flow_nodes.append(self.nodes[current_id])
                
                # Find connected nodes
                for edge in self.edges:
                    if edge.source_request_id == current_id and edge.target_request_id not in visited:
                        queue.append(edge.target_request_id)
                        flow_edges.append(edge)
                    elif edge.target_request_id == current_id and edge.source_request_id not in visited:
                        queue.append(edge.source_request_id)
                        flow_edges.append(edge)
            
            # Build flow visualization
            flow_data = {
                "root_request_id": request_id,
                "total_nodes": len(flow_nodes),
                "total_edges": len(flow_edges),
                "flow_timeline": self._build_timeline(flow_nodes, flow_edges),
                "request_chain": self._build_request_chain(request_id, flow_nodes, flow_edges),
                "parallel_requests": self._identify_parallel_requests(flow_nodes, flow_edges),
                "nodes": [asdict(node) for node in flow_nodes],
                "edges": [asdict(edge) for edge in flow_edges]
            }
            
            return flow_data
    
    def get_tool_usage_patterns(self, tool_name: Optional[str] = None) -> Dict[str, Any]:
        """Get usage patterns for tools"""
        with self._lock:
            patterns = {
                "total_requests": len(self.nodes),
                "tool_usage": defaultdict(int),
                "common_sequences": defaultdict(int),
                "parallel_patterns": defaultdict(int),
                "avg_duration_by_tool": defaultdict(list),
                "error_rates": defaultdict(lambda: {"total": 0, "errors": 0})
            }
            
            # Analyze nodes
            for node in self.nodes.values():
                if tool_name and node.tool_name != tool_name:
                    continue
                
                patterns["tool_usage"][node.tool_name] += 1
                
                if node.duration:
                    patterns["avg_duration_by_tool"][node.tool_name].append(node.duration)
                
                patterns["error_rates"][node.tool_name]["total"] += 1
                if node.status == "failed":
                    patterns["error_rates"][node.tool_name]["errors"] += 1
            
            # Calculate averages
            for tool, durations in patterns["avg_duration_by_tool"].items():
                if durations:
                    patterns["avg_duration_by_tool"][tool] = sum(durations) / len(durations)
            
            # Calculate error rates
            for tool, stats in patterns["error_rates"].items():
                if stats["total"] > 0:
                    patterns["error_rates"][tool]["error_rate"] = stats["errors"] / stats["total"]
            
            # Analyze sequences
            for edge in self.edges:
                if edge.relationship_type == "sequential":
                    source_tool = self.nodes.get(edge.source_request_id)
                    target_tool = self.nodes.get(edge.target_request_id)
                    if source_tool and target_tool:
                        sequence = f"{source_tool.tool_name} -> {target_tool.tool_name}"
                        patterns["common_sequences"][sequence] += 1
            
            return dict(patterns)
    
    def _build_timeline(self, nodes: List[RequestNode], edges: List[RequestEdge]) -> List[Dict[str, Any]]:
        """Build a timeline of requests"""
        timeline = []
        
        # Sort nodes by timestamp
        sorted_nodes = sorted(nodes, key=lambda x: x.timestamp)
        
        for node in sorted_nodes:
            timeline.append({
                "request_id": node.request_id,
                "tool_name": node.tool_name,
                "timestamp": node.timestamp,
                "duration": node.duration,
                "status": node.status,
                "arguments": node.arguments
            })
        
        return timeline
    
    def _build_request_chain(self, root_id: str, nodes: List[RequestNode], 
                           edges: List[RequestEdge]) -> List[str]:
        """Build a sequential chain of requests"""
        chain = [root_id]
        
        # Follow sequential edges
        current_id = root_id
        visited = {root_id}
        
        while True:
            next_id = None
            for edge in edges:
                if (edge.source_request_id == current_id and 
                    edge.relationship_type == "sequential" and 
                    edge.target_request_id not in visited):
                    next_id = edge.target_request_id
                    break
            
            if next_id:
                chain.append(next_id)
                visited.add(next_id)
                current_id = next_id
            else:
                break
        
        return chain
    
    def _identify_parallel_requests(self, nodes: List[RequestNode], 
                                  edges: List[RequestEdge]) -> List[List[str]]:
        """Identify parallel request groups"""
        parallel_groups = []
        
        # Group by timestamp windows (requests within 100ms are considered parallel)
        timestamp_groups = defaultdict(list)
        
        for node in nodes:
            time_bucket = int(node.timestamp * 10)  # 100ms buckets
            timestamp_groups[time_bucket].append(node.request_id)
        
        # Filter groups with more than one request
        for group in timestamp_groups.values():
            if len(group) > 1:
                parallel_groups.append(group)
        
        return parallel_groups
    
    def _load_data(self) -> None:
        """Load knowledge graph data from storage"""
        try:
            nodes_file = self.storage_path / "nodes.json"
            edges_file = self.storage_path / "edges.json"
            flows_file = self.storage_path / "flows.json"
            
            if nodes_file.exists():
                with open(nodes_file, 'r') as f:
                    nodes_data = json.load(f)
                    self.nodes = {
                        node_id: RequestNode(**node_data)
                        for node_id, node_data in nodes_data.items()
                    }
            
            if edges_file.exists():
                with open(edges_file, 'r') as f:
                    edges_data = json.load(f)
                    self.edges = [RequestEdge(**edge_data) for edge_data in edges_data]
            
            if flows_file.exists():
                with open(flows_file, 'r') as f:
                    flows_data = json.load(f)
                    self.flows = {
                        flow_id: RequestFlow(**flow_data)
                        for flow_id, flow_data in flows_data.items()
                    }
        
        except Exception as e:
            # If loading fails, start with empty data
            pass
    
    def save_data(self) -> None:
        """Save knowledge graph data to storage"""
        try:
            with self._lock:
                # Save nodes
                nodes_file = self.storage_path / "nodes.json"
                with open(nodes_file, 'w') as f:
                    nodes_data = {node_id: asdict(node) for node_id, node in self.nodes.items()}
                    json.dump(nodes_data, f, indent=2)
                
                # Save edges
                edges_file = self.storage_path / "edges.json"
                with open(edges_file, 'w') as f:
                    edges_data = [asdict(edge) for edge in self.edges]
                    json.dump(edges_data, f, indent=2)
                
                # Save flows
                flows_file = self.storage_path / "flows.json"
                with open(flows_file, 'w') as f:
                    flows_data = {flow_id: asdict(flow) for flow_id, flow in self.flows.items()}
                    json.dump(flows_data, f, indent=2)
        
        except Exception as e:
            # Silently fail on save errors
            pass
    
    def cleanup_old_data(self, max_age_days: int = 30) -> int:
        """Clean up old request data"""
        with self._lock:
            cutoff_time = time.time() - (max_age_days * 24 * 3600)
            
            # Find old nodes
            old_request_ids = [
                request_id for request_id, node in self.nodes.items()
                if node.timestamp < cutoff_time
            ]
            
            # Remove old nodes
            for request_id in old_request_ids:
                del self.nodes[request_id]
            
            # Remove edges with old nodes
            self.edges = [
                edge for edge in self.edges
                if (edge.source_request_id not in old_request_ids and 
                    edge.target_request_id not in old_request_ids)
            ]
            
            # Clean up active sessions
            for session_id, request_ids in list(self.active_sessions.items()):
                self.active_sessions[session_id] = [
                    rid for rid in request_ids if rid not in old_request_ids
                ]
                if not self.active_sessions[session_id]:
                    del self.active_sessions[session_id]
            
            return len(old_request_ids)


# Global knowledge graph instance
_knowledge_graph = None


def get_knowledge_graph() -> KnowledgeGraph:
    """Get the global knowledge graph instance"""
    global _knowledge_graph
    if _knowledge_graph is None:
        _knowledge_graph = KnowledgeGraph()
    return _knowledge_graph