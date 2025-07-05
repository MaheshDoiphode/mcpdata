"""
Basic Search Functionality for MCP Data Indexing

This module provides simple search capabilities for the mcpdata indexing system.
The complex search functionality is handled by the separate mcp-global-server.
"""

import re
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class BasicSearchResult:
    """Simple search result for internal mcpdata use"""
    content: str
    file_path: str
    line_number: int
    score: float
    match_type: str = "basic"


class BasicQueryProcessor:
    """Simple query processing for mcpdata"""

    def __init__(self):
        self.stop_words = {
            'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
            'by', 'from', 'is', 'are', 'was', 'were', 'be', 'been', 'being'
        }

    def process_query(self, query: str) -> List[str]:
        """Extract search terms from query"""
        # Simple term extraction
        terms = re.findall(r'\b\w+\b', query.lower())

        # Remove stop words and short terms
        terms = [term for term in terms if term not in self.stop_words and len(term) > 2]

        return terms


class BasicSearchEngine:
    """Simple search engine for mcpdata internal use"""

    def __init__(self, mcp_data_path: Path):
        self.mcp_data_path = Path(mcp_data_path)
        self.query_processor = BasicQueryProcessor()

    def search_content(self, query: str, max_results: int = 10) -> List[BasicSearchResult]:
        """Basic content search"""
        terms = self.query_processor.process_query(query)

        if not terms:
            return []

        results = []

        # Load content index if it exists
        try:
            content_index = self._load_content_index()

            for content_id, content_data in content_index.items():
                score = self._calculate_basic_score(content_data, terms)

                if score > 0:
                    result = BasicSearchResult(
                        content=content_data.get('content', '')[:200] + "...",
                        file_path=content_data.get('file_path', ''),
                        line_number=1,
                        score=score
                    )
                    results.append(result)

        except Exception as e:
            logger.warning(f"Search error: {e}")

        # Sort by score and return top results
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:max_results]

    def _calculate_basic_score(self, content_data: Dict[str, Any], terms: List[str]) -> float:
        """Calculate basic relevance score"""
        content = f"{content_data.get('title', '')} {content_data.get('content', '')}"
        content_lower = content.lower()

        score = 0.0

        for term in terms:
            if term in content_lower:
                # Simple frequency-based scoring
                count = content_lower.count(term)
                score += count * 0.1

        return min(score, 1.0)

    def _load_content_index(self) -> Dict[str, Any]:
        """Load content index"""
        try:
            content_path = self.mcp_data_path / 'index' / 'content.json'
            if content_path.exists():
                with open(content_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            logger.warning(f"Could not load content index: {e}")

        return {}


def create_basic_search(mcp_data_path: Path) -> BasicSearchEngine:
    """Create a basic search engine for mcpdata"""
    return BasicSearchEngine(mcp_data_path)
