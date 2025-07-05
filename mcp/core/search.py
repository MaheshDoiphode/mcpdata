"""
Advanced search module for MCP indexing system
"""

import re
import math
import time
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from collections import Counter, defaultdict
from dataclasses import dataclass

from .models import SearchIndex, QueryResult, DocumentSection, CodeSymbol

try:
    import numpy as np
    from sentence_transformers import SentenceTransformer
    NUMPY_AVAILABLE = True
except ImportError:
    np = None
    SentenceTransformer = None
    NUMPY_AVAILABLE = False

# Type hints compatibility
if NUMPY_AVAILABLE:
    from typing import TYPE_CHECKING
    if TYPE_CHECKING:
        import numpy.typing as npt
        NDArray = npt.NDArray
else:
    NDArray = 'numpy.ndarray'


@dataclass
class SearchConfig:
    """Configuration for search behavior"""
    bm25_k1: float = 1.2
    bm25_b: float = 0.75
    semantic_threshold: float = 0.3
    keyword_boost: float = 2.0
    title_boost: float = 1.5
    short_content_boost: float = 1.2
    max_results: int = 50
    default_token_limit: int = 4000


class QueryPreprocessor:
    """Utility class for query preprocessing"""

    STOP_WORDS = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'by', 'how', 'what', 'where', 'when', 'why', 'is', 'are',
        'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does',
        'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can',
        'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we',
        'they', 'me', 'him', 'her', 'us', 'them'
    }

    @staticmethod
    def preprocess_query(query: str) -> List[str]:
        """Preprocess query for better search"""
        # Convert to lowercase and remove special characters
        query = re.sub(r'[^\w\s]', ' ', query.lower())
        query = re.sub(r'\s+', ' ', query).strip()

        # Split into terms
        terms = query.split()

        # Remove stop words and short terms
        return [term for term in terms if len(term) > 2 and term not in QueryPreprocessor.STOP_WORDS]

    @staticmethod
    def expand_query(query_terms: List[str]) -> List[str]:
        """Expand query with synonyms and related terms"""
        # Simple expansion - in production, you'd use a thesaurus or word embeddings
        expansions = {
            'install': ['setup', 'installation', 'configure'],
            'config': ['configuration', 'settings', 'setup'],
            'api': ['interface', 'endpoint', 'service'],
            'error': ['exception', 'bug', 'issue', 'problem'],
            'test': ['testing', 'spec', 'validate'],
            'doc': ['documentation', 'guide', 'manual'],
            'build': ['compile', 'make', 'generate'],
            'run': ['execute', 'start', 'launch']
        }

        expanded_terms = set(query_terms)
        for term in query_terms:
            if term in expansions:
                expanded_terms.update(expansions[term])

        return list(expanded_terms)


class BM25Scorer:
    """BM25 scoring implementation"""

    def __init__(self, config: SearchConfig):
        self.k1 = config.bm25_k1
        self.b = config.bm25_b

    def calculate_score(self, query_terms: List[str], section_id: str,
                       search_index: SearchIndex) -> float:
        """Calculate BM25 relevance score"""
        if section_id not in search_index.term_frequencies:
            return 0.0

        tf = search_index.term_frequencies[section_id]
        doc_len = search_index.section_lengths[section_id]
        avg_doc_len = sum(search_index.section_lengths.values()) / len(search_index.section_lengths)

        score = 0.0
        for term in query_terms:
            if term in tf:
                # Term frequency component
                tf_component = tf[term] * (self.k1 + 1) / (
                    tf[term] + self.k1 * (1 - self.b + self.b * doc_len / avg_doc_len)
                )

                # Inverse document frequency component
                df = search_index.document_frequencies.get(term, 0)
                if df > 0:
                    idf = math.log((search_index.total_sections - df + 0.5) / (df + 0.5))
                    score += idf * tf_component

        return score


class SemanticSearcher:
    """Semantic search using embeddings"""

    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2"):
        self.model = None
        self.embedding_model_name = embedding_model
        self._load_model()

    def _load_model(self):
        """Load sentence transformer model"""
        if SentenceTransformer is None:
            print("Warning: sentence-transformers not installed. Semantic search disabled.")
            return

        try:
            self.model = SentenceTransformer(self.embedding_model_name)
        except Exception as e:
            print(f"Warning: Could not load embedding model {self.embedding_model_name}: {e}")

    def search(self, query: str, embeddings: 'numpy.ndarray',
               section_ids: List[str], threshold: float = 0.3) -> List[Tuple[str, float]]:
        """Perform semantic search"""
        if self.model is None or embeddings is None or not NUMPY_AVAILABLE:
            return []

        try:
            # Generate query embedding
            query_embedding = self.model.encode([query])

            # Calculate cosine similarity
            similarities = np.dot(embeddings, query_embedding.T).flatten()

            # Create results above threshold
            results = []
            for i, similarity in enumerate(similarities):
                if i < len(section_ids) and similarity > threshold:
                    results.append((section_ids[i], float(similarity)))

            return sorted(results, key=lambda x: x[1], reverse=True)

        except Exception as e:
            print(f"Warning: Semantic search failed: {e}")
            return []


class HybridRanker:
    """Hybrid ranking combining keyword and semantic search"""

    def __init__(self, config: SearchConfig):
        self.config = config

    def rank_results(self, keyword_results: List[QueryResult],
                    semantic_results: List[QueryResult],
                    query_terms: List[str]) -> List[QueryResult]:
        """Combine and rank keyword and semantic results"""
        # Combine results
        all_results = {}

        # Add keyword results with boost
        for result in keyword_results:
            all_results[result.section_id] = result
            result.score *= self.config.keyword_boost

        # Add semantic results
        for result in semantic_results:
            if result.section_id in all_results:
                # Combine scores for sections found in both
                existing = all_results[result.section_id]
                existing.score = (existing.score + result.score) / 2
                existing.match_type = 'hybrid'
            else:
                all_results[result.section_id] = result

        # Apply additional ranking factors
        for result in all_results.values():
            self._apply_ranking_boosts(result, query_terms)

        return sorted(all_results.values(), key=lambda x: x.score, reverse=True)

    def _apply_ranking_boosts(self, result: QueryResult, query_terms: List[str]):
        """Apply additional ranking factors"""
        # Boost exact title matches
        if any(term in result.title.lower() for term in query_terms):
            result.score *= self.config.title_boost

        # Boost shorter, more focused sections
        if len(result.content) < 500:
            result.score *= self.config.short_content_boost

        # Boost based on section level (higher level = more important)
        if hasattr(result, 'level'):
            level_boost = 1.0 + (0.1 * (4 - getattr(result, 'level', 3)))
            result.score *= level_boost


class SearchEngine:
    """Main search engine orchestrating all search functionality"""

    def __init__(self, mcp_data_path: Path, config: SearchConfig = None):
        self.mcp_data_path = Path(mcp_data_path)
        self.config = config or SearchConfig()

        # Initialize components
        self.preprocessor = QueryPreprocessor()
        self.bm25_scorer = BM25Scorer(self.config)
        self.semantic_searcher = SemanticSearcher()
        self.hybrid_ranker = HybridRanker(self.config)

        # Cached indices
        self._content_index = None
        self._search_index = None
        self._embeddings = None
        self._section_ids = None

        # Performance metrics
        self.query_count = 0
        self.total_query_time = 0.0

    def load_indices(self) -> bool:
        """Load search indices from disk"""
        try:
            index_dir = Path(self.mcp_data_path) / 'index'

            # Load content index
            content_idx_path = index_dir / 'content.idx'
            with open(content_idx_path, 'r', encoding='utf-8') as f:
                self._content_index = json.load(f)

            # Load search index
            search_idx_path = index_dir / 'search.idx'
            with open(search_idx_path, 'r', encoding='utf-8') as f:
                search_data = json.load(f)

                # Reconstruct SearchIndex
                self._search_index = SearchIndex(
                    inverted_index=search_data['inverted_index'],
                    term_frequencies={k: Counter(v) for k, v in search_data['term_frequencies'].items()},
                    document_frequencies=search_data['document_frequencies'],
                    section_lengths=search_data['section_lengths'],
                    total_sections=search_data['total_sections']
                )

            # Load embeddings
            vectors_dir = Path(self.mcp_data_path) / 'vectors'
            embeddings_path = vectors_dir / 'doc_embeddings.bin'
            if embeddings_path.exists() and NUMPY_AVAILABLE:
                self._embeddings = np.load(str(embeddings_path), allow_pickle=True)
                self._section_ids = list(self._content_index['sections'].keys())

            return True

        except Exception as e:
            print(f"Error loading search indices: {e}")
            return False

    def query(self, query: str, max_tokens: int = None, limit: int = None) -> List[QueryResult]:
        """Execute hybrid search query"""
        start_time = time.time()

        # Set defaults
        if max_tokens is None:
            max_tokens = self.config.default_token_limit
        if limit is None:
            limit = self.config.max_results

        # Ensure indices are loaded
        if not self._content_index or not self._search_index:
            if not self.load_indices():
                return []

        # Preprocess query
        query_terms = self.preprocessor.preprocess_query(query)
        if not query_terms:
            return []

        # Expand query (optional)
        expanded_terms = self.preprocessor.expand_query(query_terms)

        # Perform keyword search
        keyword_results = self._keyword_search(expanded_terms)

        # Perform semantic search
        semantic_results = self._semantic_search(query)

        # Combine and rank results
        final_results = self.hybrid_ranker.rank_results(
            keyword_results, semantic_results, query_terms
        )

        # Apply limit
        final_results = final_results[:limit]

        # Update performance metrics
        query_time = time.time() - start_time
        self.query_count += 1
        self.total_query_time += query_time

        return final_results

    def _keyword_search(self, query_terms: List[str]) -> List[QueryResult]:
        """Perform keyword search using BM25"""
        results = []

        # Find candidate sections
        candidate_sections = set()
        for term in query_terms:
            if term in self._search_index.inverted_index:
                candidate_sections.update(self._search_index.inverted_index[term])

        # Calculate BM25 scores
        for section_id in candidate_sections:
            if section_id in self._content_index['sections']:
                score = self.bm25_scorer.calculate_score(
                    query_terms, section_id, self._search_index
                )

                if score > 0:
                    section_data = self._content_index['sections'][section_id]

                    results.append(QueryResult(
                        section_id=section_id,
                        title=section_data['title'],
                        content=section_data['content'],
                        score=score,
                        match_type='keyword',
                        file_path=section_data.get('file_path', 'unknown'),
                        line_range=(0, 0)  # Will be filled from metadata if available
                    ))

        return sorted(results, key=lambda x: x.score, reverse=True)

    def _semantic_search(self, query: str) -> List[QueryResult]:
        """Perform semantic search using embeddings"""
        if not self._embeddings or not self._section_ids or not NUMPY_AVAILABLE:
            return []

        # Get semantic matches
        semantic_matches = self.semantic_searcher.search(
            query, self._embeddings, self._section_ids, self.config.semantic_threshold
        )

        # Convert to QueryResult objects
        results = []
        for section_id, score in semantic_matches:
            if section_id in self._content_index['sections']:
                section_data = self._content_index['sections'][section_id]

                results.append(QueryResult(
                    section_id=section_id,
                    title=section_data['title'],
                    content=section_data['content'],
                    score=score,
                    match_type='semantic',
                    file_path=section_data.get('file_path', 'unknown'),
                    line_range=(0, 0)
                ))

        return results

    def get_optimized_context(self, query: str, max_tokens: int = None) -> str:
        """Get optimized context for MCP server response"""
        if max_tokens is None:
            max_tokens = self.config.default_token_limit

        results = self.query(query, max_tokens)

        if not results:
            return "No relevant content found."

        # Build context within token limit
        context_parts = []
        current_tokens = 0

        for result in results:
            # Estimate tokens (rough: 4 chars per token)
            result_tokens = len(result.content) // 4

            if current_tokens + result_tokens <= max_tokens:
                context_parts.append(f"## {result.title}\n{result.content}")
                current_tokens += result_tokens
            else:
                # Truncate last section if needed
                remaining_tokens = max_tokens - current_tokens
                if remaining_tokens > 100:  # Only add if meaningful content fits
                    truncated_content = result.content[:remaining_tokens * 4]
                    context_parts.append(f"## {result.title}\n{truncated_content}...")
                break

        return "\n\n".join(context_parts)

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        avg_query_time = self.total_query_time / self.query_count if self.query_count > 0 else 0

        return {
            'total_queries': self.query_count,
            'total_query_time': self.total_query_time,
            'average_query_time': avg_query_time,
            'indices_loaded': bool(self._content_index and self._search_index),
            'semantic_search_available': bool(self._embeddings is not None),
            'total_sections': len(self._content_index['sections']) if self._content_index else 0
        }


class SearchIndexBuilder:
    """Builder for creating search indices"""

    @staticmethod
    def build_inverted_index(sections: List[DocumentSection]) -> SearchIndex:
        """Build advanced inverted index for full-text search"""
        inverted_index = defaultdict(list)
        term_frequencies = defaultdict(Counter)
        document_frequencies = defaultdict(int)
        section_lengths = {}

        for section in sections:
            # Extract and clean text
            text = f"{section.title} {section.content}".lower()
            text = re.sub(r'[^\w\s]', ' ', text)
            text = re.sub(r'\s+', ' ', text)

            words = text.split()
            section_lengths[section.id] = len(words)

            # Count term frequencies
            word_counts = Counter(words)
            term_frequencies[section.id] = word_counts

            # Build inverted index
            for word in set(words):
                if len(word) > 2:  # Filter out very short words
                    inverted_index[word].append(section.id)
                    document_frequencies[word] += 1

        return SearchIndex(
            inverted_index=dict(inverted_index),
            term_frequencies=dict(term_frequencies),
            document_frequencies=dict(document_frequencies),
            section_lengths=section_lengths,
            total_sections=len(sections)
        )

    @staticmethod
    def create_search_metadata(sections: List[DocumentSection]) -> Dict[str, Any]:
        """Create metadata for search operations"""
        return {
            section.id: {
                'title': section.title,
                'level': section.level,
                'file_path': section.file_path or 'unknown',
                'line_range': [section.start_line, section.end_line],
                'parent_id': section.parent_id,
                'children': section.children,
                'content_length': len(section.content),
                'word_count': len(section.content.split())
            } for section in sections
        }


class CachedSearchEngine(SearchEngine):
    """Search engine with result caching for improved performance"""

    def __init__(self, mcp_data_path: Path, config: SearchConfig = None, cache_size: int = 100):
        super().__init__(mcp_data_path, config)
        self.cache_size = cache_size
        self._query_cache = {}
        self._cache_hits = 0
        self._cache_misses = 0

    def query(self, query: str, max_tokens: int = None, limit: int = None) -> List[QueryResult]:
        """Execute query with caching"""
        # Create cache key
        cache_key = f"{query}:{max_tokens}:{limit}"

        # Check cache
        if cache_key in self._query_cache:
            self._cache_hits += 1
            return self._query_cache[cache_key]

        # Execute query
        results = super().query(query, max_tokens, limit)

        # Cache results
        self._cache_misses += 1
        if len(self._query_cache) >= self.cache_size:
            # Remove oldest entry (simple LRU)
            oldest_key = next(iter(self._query_cache))
            del self._query_cache[oldest_key]

        self._query_cache[cache_key] = results
        return results

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics"""
        total_requests = self._cache_hits + self._cache_misses
        hit_rate = self._cache_hits / total_requests if total_requests > 0 else 0

        return {
            'cache_hits': self._cache_hits,
            'cache_misses': self._cache_misses,
            'hit_rate': hit_rate,
            'cache_size': len(self._query_cache),
            'max_cache_size': self.cache_size
        }
