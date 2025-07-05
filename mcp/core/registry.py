"""
Central Registry System for MCP Documentation Indexing
Handles workspace registration, global search, and cross-workspace relationships
"""

import json
import time
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
from collections import defaultdict


@dataclass
class WorkspaceMetadata:
    """Metadata for a registered workspace"""
    id: str
    name: str
    description: str
    path: str
    created_at: str
    updated_at: str
    version: str
    project_type: str
    file_count: int
    document_count: int
    code_symbol_count: int
    embedding_count: int
    tags: List[str]
    relationships: Dict[str, List[str]]  # workspace_id -> list of relationship types
    status: str  # 'active', 'inactive', 'error'
    last_indexed: Optional[str] = None
    index_size_mb: float = 0.0
    health_score: float = 1.0

    def __post_init__(self):
        if not self.tags:
            self.tags = []
        if not self.relationships:
            self.relationships = {}


@dataclass
class GlobalSearchMetadata:
    """Lightweight metadata for global search"""
    workspace_id: str
    workspace_name: str
    section_id: str
    title: str
    content_preview: str  # First 200 chars
    file_path: str
    section_type: str  # 'document', 'code', 'config'
    keywords: List[str]
    relevance_score: float = 0.0


@dataclass
class RegistryConfig:
    """Configuration for the central registry"""
    registry_version: str = "1.0.0"
    max_workspaces: int = 100
    max_content_preview_length: int = 200
    global_search_index_size_mb: float = 50.0
    relationship_types: List[str] = None
    auto_cleanup_days: int = 30
    backup_retention_days: int = 7

    def __post_init__(self):
        if self.relationship_types is None:
            self.relationship_types = [
                "imports", "references", "depends_on", "extends",
                "implements", "shares_concepts", "related_docs"
            ]


@dataclass
class RegistryStats:
    """Statistics for the central registry"""
    total_workspaces: int = 0
    active_workspaces: int = 0
    total_documents: int = 0
    total_code_symbols: int = 0
    total_embeddings: int = 0
    total_size_mb: float = 0.0
    last_updated: str = ""
    health_status: str = "healthy"


class CentralRegistry:
    """Central registry manager for MCP workspaces"""

    def __init__(self, registry_path: Optional[Path] = None):
        """Initialize the central registry"""
        if registry_path is None:
            # Default to ~/Documents/mcpdata/
            home = Path.home()
            registry_path = home / "Documents" / "mcpdata"

        self.registry_path = Path(registry_path)
        self.registry_file = self.registry_path / "registry.json"
        self.workspaces_dir = self.registry_path / "workspaces"
        self.global_dir = self.registry_path / "global"
        self.backup_dir = self.registry_path / "backups"

        # Create directories if they don't exist
        for dir_path in [self.registry_path, self.workspaces_dir, self.global_dir, self.backup_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

        self.config = RegistryConfig()
        self._workspaces: Dict[str, WorkspaceMetadata] = {}
        self._global_search_data: List[GlobalSearchMetadata] = []
        self._load_registry()

    def _generate_workspace_id(self, workspace_path: str) -> str:
        """Generate a unique workspace ID"""
        path_hash = hashlib.md5(workspace_path.encode()).hexdigest()[:8]
        timestamp = str(int(time.time()))[-6:]
        return f"ws_{path_hash}_{timestamp}"

    def _load_registry(self) -> None:
        """Load registry from disk"""
        if self.registry_file.exists():
            try:
                with open(self.registry_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                # Load workspaces
                workspaces_data = data.get("workspaces", {})
                for ws_id, ws_data in workspaces_data.items():
                    self._workspaces[ws_id] = WorkspaceMetadata(**ws_data)

                # Load config if present
                if "config" in data:
                    config_data = data["config"]
                    self.config = RegistryConfig(**config_data)

                # Load global search data
                global_search_file = self.global_dir / "search_metadata.json"
                if global_search_file.exists():
                    with open(global_search_file, 'r', encoding='utf-8') as f:
                        search_data = json.load(f)
                        self._global_search_data = [
                            GlobalSearchMetadata(**item) for item in search_data
                        ]

            except Exception as e:
                print(f"Warning: Could not load registry: {e}")
                print("Starting with empty registry")

    def _save_registry(self) -> None:
        """Save registry to disk"""
        try:
            # Prepare data for serialization
            registry_data = {
                "version": self.config.registry_version,
                "created_at": datetime.now().isoformat(),
                "config": asdict(self.config),
                "workspaces": {
                    ws_id: asdict(ws_meta) for ws_id, ws_meta in self._workspaces.items()
                },
                "stats": asdict(self.get_stats())
            }

            # Create backup if registry exists
            if self.registry_file.exists():
                import time
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                backup_name = f"registry_backup_{timestamp}.json"
                backup_path = self.backup_dir / backup_name

                # Handle existing backup files
                counter = 1
                while backup_path.exists():
                    backup_name = f"registry_backup_{timestamp}_{counter}.json"
                    backup_path = self.backup_dir / backup_name
                    counter += 1

                # Copy instead of rename to avoid file system issues
                import shutil
                shutil.copy2(self.registry_file, backup_path)

            # Save registry
            with open(self.registry_file, 'w', encoding='utf-8') as f:
                json.dump(registry_data, f, indent=2, ensure_ascii=False)

            # Save global search data
            global_search_file = self.global_dir / "search_metadata.json"
            with open(global_search_file, 'w', encoding='utf-8') as f:
                search_data = [asdict(item) for item in self._global_search_data]
                json.dump(search_data, f, indent=2, ensure_ascii=False)

            # Cleanup old backups
            self._cleanup_old_backups()

        except Exception as e:
            raise RuntimeError(f"Could not save registry: {e}")

    def _cleanup_old_backups(self) -> None:
        """Clean up old backup files"""
        try:
            backup_files = list(self.backup_dir.glob("registry_backup_*.json"))
            backup_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)

            # Keep only the most recent backups
            for old_backup in backup_files[self.config.backup_retention_days:]:
                old_backup.unlink()
        except Exception:
            pass  # Ignore cleanup errors

    def register_workspace(self, workspace_path: str, name: str, description: str,
                           project_type: str = "auto", tags: List[str] = None,
                           force: bool = False) -> str:
        """Register a new workspace"""
        workspace_path = str(Path(workspace_path).resolve())

        # Check if workspace already exists
        existing_ws = self.find_workspace_by_path(workspace_path)
        if existing_ws and not force:
            raise ValueError(f"Workspace already registered: {existing_ws.name}")

        # Generate workspace ID
        if existing_ws and force:
            workspace_id = existing_ws.id
        else:
            workspace_id = self._generate_workspace_id(workspace_path)

        # Check workspace limits
        if len(self._workspaces) >= self.config.max_workspaces and not existing_ws:
            raise ValueError(f"Maximum workspace limit reached: {self.config.max_workspaces}")

        # Create workspace metadata
        now = datetime.now().isoformat()
        workspace_meta = WorkspaceMetadata(
            id=workspace_id,
            name=name,
            description=description,
            path=workspace_path,
            created_at=existing_ws.created_at if existing_ws else now,
            updated_at=now,
            version="1.0.0",
            project_type=project_type,
            file_count=0,
            document_count=0,
            code_symbol_count=0,
            embedding_count=0,
            tags=tags or [],
            relationships={},
            status="active"
        )

        # Register workspace
        self._workspaces[workspace_id] = workspace_meta

        # Create workspace directory
        workspace_dir = self.workspaces_dir / workspace_id
        workspace_dir.mkdir(exist_ok=True)

        # Save workspace metadata
        workspace_meta_file = workspace_dir / "metadata.json"
        with open(workspace_meta_file, 'w', encoding='utf-8') as f:
            json.dump(asdict(workspace_meta), f, indent=2, ensure_ascii=False)

        # Save registry
        self._save_registry()

        return workspace_id

    def update_workspace_stats(self, workspace_id: str, stats: Dict[str, Any]) -> None:
        """Update workspace statistics"""
        if workspace_id not in self._workspaces:
            raise ValueError(f"Workspace not found: {workspace_id}")

        workspace = self._workspaces[workspace_id]

        # Update stats
        workspace.file_count = stats.get("file_count", workspace.file_count)
        workspace.document_count = stats.get("document_count", workspace.document_count)
        workspace.code_symbol_count = stats.get("code_symbol_count", workspace.code_symbol_count)
        workspace.embedding_count = stats.get("embedding_count", workspace.embedding_count)
        workspace.index_size_mb = stats.get("index_size_mb", workspace.index_size_mb)
        workspace.last_indexed = datetime.now().isoformat()
        workspace.updated_at = datetime.now().isoformat()

        # Update status
        workspace.status = "active"
        workspace.health_score = stats.get("health_score", 1.0)

        # Save registry
        self._save_registry()

    def update_global_search_data(self, workspace_id: str, search_metadata: List[GlobalSearchMetadata]) -> None:
        """Update global search data for a workspace"""
        if workspace_id not in self._workspaces:
            raise ValueError(f"Workspace not found: {workspace_id}")

        # Remove existing data for this workspace
        self._global_search_data = [
            item for item in self._global_search_data
            if item.workspace_id != workspace_id
        ]

        # Add new data
        self._global_search_data.extend(search_metadata)

        # Save registry
        self._save_registry()

    def find_workspace_by_path(self, workspace_path: str) -> Optional[WorkspaceMetadata]:
        """Find workspace by path"""
        workspace_path = str(Path(workspace_path).resolve())
        for workspace in self._workspaces.values():
            if workspace.path == workspace_path:
                return workspace
        return None

    def get_workspace(self, workspace_id: str) -> Optional[WorkspaceMetadata]:
        """Get workspace by ID"""
        return self._workspaces.get(workspace_id)

    def list_workspaces(self, include_inactive: bool = False) -> List[WorkspaceMetadata]:
        """List all workspaces"""
        workspaces = list(self._workspaces.values())
        if not include_inactive:
            workspaces = [ws for ws in workspaces if ws.status == "active"]
        return workspaces

    def remove_workspace(self, workspace_id: str) -> None:
        """Remove a workspace from the registry"""
        if workspace_id not in self._workspaces:
            raise ValueError(f"Workspace not found: {workspace_id}")

        # Remove from registry
        del self._workspaces[workspace_id]

        # Remove global search data
        self._global_search_data = [
            item for item in self._global_search_data
            if item.workspace_id != workspace_id
        ]

        # Remove workspace directory
        workspace_dir = self.workspaces_dir / workspace_id
        if workspace_dir.exists():
            import shutil
            shutil.rmtree(workspace_dir)

        # Save registry
        self._save_registry()

    def global_search(self, query: str, max_results: int = 20) -> List[GlobalSearchMetadata]:
        """Perform global search across all workspaces"""
        query_lower = query.lower()
        results = []

        for item in self._global_search_data:
            score = 0.0

            # Title match (higher weight)
            if query_lower in item.title.lower():
                score += 10.0

            # Content preview match
            if query_lower in item.content_preview.lower():
                score += 5.0

            # Keywords match
            for keyword in item.keywords:
                if query_lower in keyword.lower():
                    score += 3.0

            # File path match
            if query_lower in item.file_path.lower():
                score += 2.0

            # Workspace name match
            if query_lower in item.workspace_name.lower():
                score += 1.0

            if score > 0:
                item.relevance_score = score
                results.append(item)

        # Sort by relevance score
        results.sort(key=lambda x: x.relevance_score, reverse=True)

        return results[:max_results]

    def get_stats(self) -> RegistryStats:
        """Get registry statistics"""
        total_workspaces = len(self._workspaces)
        active_workspaces = sum(1 for ws in self._workspaces.values() if ws.status == "active")
        total_documents = sum(ws.document_count for ws in self._workspaces.values())
        total_code_symbols = sum(ws.code_symbol_count for ws in self._workspaces.values())
        total_embeddings = sum(ws.embedding_count for ws in self._workspaces.values())
        total_size_mb = sum(ws.index_size_mb for ws in self._workspaces.values())

        # Determine health status
        health_scores = [ws.health_score for ws in self._workspaces.values() if ws.status == "active"]
        avg_health = sum(health_scores) / len(health_scores) if health_scores else 1.0

        health_status = "healthy"
        if avg_health < 0.8:
            health_status = "warning"
        elif avg_health < 0.5:
            health_status = "critical"

        return RegistryStats(
            total_workspaces=total_workspaces,
            active_workspaces=active_workspaces,
            total_documents=total_documents,
            total_code_symbols=total_code_symbols,
            total_embeddings=total_embeddings,
            total_size_mb=total_size_mb,
            last_updated=datetime.now().isoformat(),
            health_status=health_status
        )

    def add_workspace_relationship(self, workspace_id: str, related_workspace_id: str,
                                   relationship_type: str) -> None:
        """Add a relationship between workspaces"""
        if workspace_id not in self._workspaces:
            raise ValueError(f"Workspace not found: {workspace_id}")
        if related_workspace_id not in self._workspaces:
            raise ValueError(f"Related workspace not found: {related_workspace_id}")
        if relationship_type not in self.config.relationship_types:
            raise ValueError(f"Invalid relationship type: {relationship_type}")

        workspace = self._workspaces[workspace_id]
        if related_workspace_id not in workspace.relationships:
            workspace.relationships[related_workspace_id] = []

        if relationship_type not in workspace.relationships[related_workspace_id]:
            workspace.relationships[related_workspace_id].append(relationship_type)

        # Save registry
        self._save_registry()

    def get_related_workspaces(self, workspace_id: str) -> Dict[str, List[str]]:
        """Get workspaces related to the given workspace"""
        if workspace_id not in self._workspaces:
            raise ValueError(f"Workspace not found: {workspace_id}")

        return self._workspaces[workspace_id].relationships.copy()

    def cleanup_inactive_workspaces(self) -> List[str]:
        """Clean up inactive workspaces"""
        removed_workspaces = []
        cutoff_time = datetime.now().timestamp() - (self.config.auto_cleanup_days * 24 * 3600)

        for workspace_id, workspace in list(self._workspaces.items()):
            if workspace.status == "inactive":
                try:
                    updated_time = datetime.fromisoformat(workspace.updated_at).timestamp()
                    if updated_time < cutoff_time:
                        self.remove_workspace(workspace_id)
                        removed_workspaces.append(workspace_id)
                except Exception:
                    pass  # Skip if timestamp parsing fails

        return removed_workspaces


# Global registry instance
_global_registry: Optional[CentralRegistry] = None


def get_global_registry() -> CentralRegistry:
    """Get the global registry instance"""
    global _global_registry
    if _global_registry is None:
        _global_registry = CentralRegistry()
    return _global_registry


def reset_global_registry() -> None:
    """Reset the global registry instance"""
    global _global_registry
    _global_registry = None
