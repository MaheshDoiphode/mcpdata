"""
Core data models for MCP indexing system
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from collections import Counter


@dataclass
class SearchIndex:
    """Advanced search index structure"""
    inverted_index: Dict[str, List[str]]
    term_frequencies: Dict[str, Counter]
    document_frequencies: Dict[str, int]
    section_lengths: Dict[str, int]
    total_sections: int


@dataclass
class QueryResult:
    """Search result with relevance score"""
    section_id: str
    title: str
    content: str
    score: float
    match_type: str  # 'keyword', 'semantic', 'hybrid'
    file_path: str
    line_range: Tuple[int, int]


@dataclass
class FileMetadata:
    """File metadata for tracking changes"""
    path: str
    size: int
    modified: float
    file_type: str
    hash: str
    language: Optional[str] = None
    encoding: str = 'utf-8'


@dataclass
class DocumentSection:
    """Document section with hierarchical structure"""
    id: str
    title: str
    level: int
    start_line: int
    end_line: int
    content: str
    parent_id: Optional[str] = None
    children: List[str] = None
    file_path: Optional[str] = None

    def __post_init__(self):
        if self.children is None:
            self.children = []


@dataclass
class CodeSymbol:
    """Code symbol (function, class, etc.)"""
    name: str
    type: str  # 'function', 'class', 'variable', 'import'
    file_path: str
    line_number: int
    column: int
    scope: str
    signature: str
    docstring: Optional[str] = None
