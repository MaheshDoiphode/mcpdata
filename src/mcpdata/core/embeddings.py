"""
Embeddings module for vector embedding management
"""

import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Tuple
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

from .models import DocumentSection, CodeSymbol

try:
    import numpy as np
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
    NUMPY_AVAILABLE = True
except ImportError:
    np = None
    SENTENCE_TRANSFORMERS_AVAILABLE = False
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
class EmbeddingConfig:
    """Configuration for embedding generation"""
    model_name: str = "all-MiniLM-L6-v2"
    batch_size: int = 32
    max_length: int = 512
    normalize_embeddings: bool = True
    cache_embeddings: bool = True
    parallel_workers: int = 4
    chunk_overlap: int = 50
    device: str = "cpu"  # or "cuda" if available


@dataclass
class EmbeddingMetadata:
    """Metadata for embeddings"""
    model_name: str
    embedding_dimension: int
    total_embeddings: int
    creation_time: float
    content_hashes: Dict[str, str]


class TextChunker:
    """Utility class for chunking text into embedding-sized pieces"""

    def __init__(self, max_length: int = 512, overlap: int = 50):
        self.max_length = max_length
        self.overlap = overlap

    def chunk_text(self, text: str, chunk_id_prefix: str = "") -> List[Tuple[str, str]]:
        """
        Split text into overlapping chunks
        Returns list of (chunk_id, chunk_text) tuples
        """
        if not text:
            return []

        # Split into sentences to avoid breaking mid-sentence
        sentences = self._split_into_sentences(text)

        chunks = []
        current_chunk = []
        current_length = 0
        chunk_index = 0

        for sentence in sentences:
            sentence_length = len(sentence)

            # If adding this sentence would exceed max length, create a chunk
            if current_length + sentence_length > self.max_length and current_chunk:
                chunk_text = " ".join(current_chunk)
                chunk_id = f"{chunk_id_prefix}_chunk_{chunk_index}"
                chunks.append((chunk_id, chunk_text))

                # Start new chunk with overlap
                overlap_sentences = self._get_overlap_sentences(current_chunk)
                current_chunk = overlap_sentences + [sentence]
                current_length = sum(len(s) for s in current_chunk)
                chunk_index += 1
            else:
                current_chunk.append(sentence)
                current_length += sentence_length

        # Add final chunk
        if current_chunk:
            chunk_text = " ".join(current_chunk)
            chunk_id = f"{chunk_id_prefix}_chunk_{chunk_index}"
            chunks.append((chunk_id, chunk_text))

        return chunks

    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        import re

        # Simple sentence splitting (could be improved with NLTK)
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]

    def _get_overlap_sentences(self, sentences: List[str]) -> List[str]:
        """Get sentences for overlap"""
        if not sentences:
            return []

        # Take last few sentences for overlap
        overlap_length = 0
        overlap_sentences = []

        for sentence in reversed(sentences):
            if overlap_length + len(sentence) <= self.overlap:
                overlap_sentences.insert(0, sentence)
                overlap_length += len(sentence)
            else:
                break

        return overlap_sentences


class EmbeddingModel:
    """Wrapper for embedding models"""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", device: str = "cpu"):
        self.model_name = model_name
        self.device = device
        self.model = None
        self.embedding_dimension = None
        self._load_model()

    def _load_model(self):
        """Load the embedding model"""
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError("sentence-transformers is required for embedding generation")

        try:
            self.model = SentenceTransformer(self.model_name, device=self.device)
            # Get embedding dimension
            test_embedding = self.model.encode(["test"], convert_to_tensor=False)
            self.embedding_dimension = len(test_embedding[0])
            logging.info(f"Loaded embedding model: {self.model_name} (dim: {self.embedding_dimension})")
        except Exception as e:
            raise RuntimeError(f"Failed to load embedding model {self.model_name}: {e}")

    def encode(self, texts: List[str], batch_size: int = 32, normalize: bool = True) -> 'numpy.ndarray':
        """Encode texts into embeddings"""
        if not self.model:
            raise RuntimeError("Model not loaded")

        if not texts:
            return np.array([]) if NUMPY_AVAILABLE else []

        try:
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                convert_to_tensor=False,
                normalize_embeddings=normalize,
                show_progress_bar=len(texts) > 100
            )
            return np.array(embeddings)
        except Exception as e:
            raise RuntimeError(f"Failed to generate embeddings: {e}")

    def encode_single(self, text: str, normalize: bool = True) -> 'numpy.ndarray':
        """Encode single text into embedding"""
        embeddings = self.encode([text], batch_size=1, normalize=normalize)
        return embeddings[0] if len(embeddings) > 0 else (np.array([]) if NUMPY_AVAILABLE else [])


class EmbeddingGenerator:
    """Main class for generating embeddings"""

    def __init__(self, config: EmbeddingConfig = None):
        self.config = config or EmbeddingConfig()
        self.model = None
        self.chunker = TextChunker(
            max_length=self.config.max_length,
            overlap=self.config.chunk_overlap
        )
        self.logger = logging.getLogger(__name__)

        # Initialize model
        self._initialize_model()

    def _initialize_model(self):
        """Initialize the embedding model"""
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            self.logger.warning("sentence-transformers not available, embeddings disabled")
            return

        try:
            self.model = EmbeddingModel(
                model_name=self.config.model_name,
                device=self.config.device
            )
        except Exception as e:
            self.logger.error(f"Failed to initialize embedding model: {e}")
            raise

    def generate_document_embeddings(self, sections: List[DocumentSection]) -> Tuple['numpy.ndarray', List[str], Dict[str, Any]]:
        """
        Generate embeddings for document sections
        Returns (embeddings_array, section_ids, metadata)
        """
        if not sections:
            return (np.array([]) if NUMPY_AVAILABLE else []), [], {}

        self.logger.info(f"Generating embeddings for {len(sections)} document sections")
        start_time = time.time()

        # Prepare texts and IDs
        texts = []
        section_ids = []
        content_hashes = {}

        for section in sections:
            # Combine title and content for better embeddings
            combined_text = f"{section.title}\n\n{section.content}"

            # Chunk long sections
            chunks = self.chunker.chunk_text(combined_text, section.id)

            for chunk_id, chunk_text in chunks:
                texts.append(chunk_text)
                section_ids.append(chunk_id)
                content_hashes[chunk_id] = self._calculate_hash(chunk_text)

        # Generate embeddings in batches
        embeddings = self._generate_embeddings_batch(texts)

        # Create metadata
        metadata = EmbeddingMetadata(
            model_name=self.config.model_name,
            embedding_dimension=self.model.embedding_dimension,
            total_embeddings=len(embeddings),
            creation_time=time.time(),
            content_hashes=content_hashes
        )

        generation_time = time.time() - start_time
        self.logger.info(f"Generated {len(embeddings)} embeddings in {generation_time:.2f}s")

        return embeddings, section_ids, metadata.__dict__

    def generate_code_embeddings(self, symbols: List[CodeSymbol]) -> Tuple['numpy.ndarray', List[str], Dict[str, Any]]:
        """
        Generate embeddings for code symbols
        Returns (embeddings_array, symbol_ids, metadata)
        """
        if not symbols:
            return (np.array([]) if NUMPY_AVAILABLE else []), [], {}

        self.logger.info(f"Generating embeddings for {len(symbols)} code symbols")
        start_time = time.time()

        # Prepare texts and IDs
        texts = []
        symbol_ids = []
        content_hashes = {}

        for symbol in symbols:
            # Create rich text representation of code symbol
            symbol_text = self._create_symbol_text(symbol)
            symbol_id = f"{symbol.file_path}:{symbol.name}"

            texts.append(symbol_text)
            symbol_ids.append(symbol_id)
            content_hashes[symbol_id] = self._calculate_hash(symbol_text)

        # Generate embeddings
        embeddings = self._generate_embeddings_batch(texts)

        # Create metadata
        metadata = EmbeddingMetadata(
            model_name=self.config.model_name,
            embedding_dimension=self.model.embedding_dimension,
            total_embeddings=len(embeddings),
            creation_time=time.time(),
            content_hashes=content_hashes
        )

        generation_time = time.time() - start_time
        self.logger.info(f"Generated {len(embeddings)} code embeddings in {generation_time:.2f}s")

        return embeddings, symbol_ids, metadata.__dict__

    def _generate_embeddings_batch(self, texts: List[str]) -> 'numpy.ndarray':
        """Generate embeddings in batches for efficiency"""
        if not texts:
            return np.array([]) if NUMPY_AVAILABLE else []

        if not NUMPY_AVAILABLE:
            raise RuntimeError("NumPy is required for embedding generation")

        # Split into batches
        batches = [texts[i:i + self.config.batch_size]
                  for i in range(0, len(texts), self.config.batch_size)]

        all_embeddings = []

        if len(batches) == 1:
            # Single batch
            embeddings = self.model.encode(
                batches[0],
                batch_size=self.config.batch_size,
                normalize=self.config.normalize_embeddings
            )
            all_embeddings.extend(embeddings)
        else:
            # Multiple batches with parallel processing
            with ThreadPoolExecutor(max_workers=self.config.parallel_workers) as executor:
                futures = []
                for batch in batches:
                    future = executor.submit(
                        self.model.encode,
                        batch,
                        self.config.batch_size,
                        self.config.normalize_embeddings
                    )
                    futures.append(future)

                for future in as_completed(futures):
                    embeddings = future.result()
                    all_embeddings.extend(embeddings)

        return np.array(all_embeddings)

    def _create_symbol_text(self, symbol: CodeSymbol) -> str:
        """Create rich text representation of code symbol"""
        parts = []

        # Add type and name
        parts.append(f"{symbol.type} {symbol.name}")

        # Add signature
        if symbol.signature:
            parts.append(symbol.signature)

        # Add docstring
        if symbol.docstring:
            parts.append(symbol.docstring)

        # Add context (file path and scope)
        parts.append(f"File: {symbol.file_path}")
        if symbol.scope != 'global':
            parts.append(f"Scope: {symbol.scope}")

        return "\n".join(parts)

    def _calculate_hash(self, text: str) -> str:
        """Calculate hash for content deduplication"""
        import hashlib
        return hashlib.md5(text.encode('utf-8')).hexdigest()


class EmbeddingStorage:
    """Storage management for embeddings"""

    def __init__(self, storage_path: Path):
        self.storage_path = storage_path
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)

    def save_embeddings(self, embeddings: 'numpy.ndarray', ids: List[str],
                       metadata: Dict[str, Any], filename: str):
        """Save embeddings to disk"""
        try:
            # Save embeddings
            embeddings_path = self.storage_path / f"{filename}.npy"
            if NUMPY_AVAILABLE:
                np.save(embeddings_path, embeddings)
            else:
                raise RuntimeError("NumPy is required for saving embeddings")

            # Save IDs
            ids_path = self.storage_path / f"{filename}_ids.json"
            with open(ids_path, 'w', encoding='utf-8') as f:
                json.dump(ids, f, indent=2)

            # Save metadata
            metadata_path = self.storage_path / f"{filename}_metadata.json"
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2)

            self.logger.info(f"Saved {len(embeddings)} embeddings to {embeddings_path}")

        except Exception as e:
            self.logger.error(f"Failed to save embeddings: {e}")
            raise

    def load_embeddings(self, filename: str) -> Tuple['numpy.ndarray', List[str], Dict[str, Any]]:
        """Load embeddings from disk"""
        try:
            # Load embeddings
            embeddings_path = self.storage_path / f"{filename}.npy"
            if NUMPY_AVAILABLE:
                embeddings = np.load(embeddings_path)
            else:
                raise RuntimeError("NumPy is required for loading embeddings")

            # Load IDs
            ids_path = self.storage_path / f"{filename}_ids.json"
            with open(ids_path, 'r', encoding='utf-8') as f:
                ids = json.load(f)

            # Load metadata
            metadata_path = self.storage_path / f"{filename}_metadata.json"
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)

            self.logger.info(f"Loaded {len(embeddings)} embeddings from {embeddings_path}")
            return embeddings, ids, metadata

        except Exception as e:
            self.logger.error(f"Failed to load embeddings: {e}")
            raise

    def embedding_exists(self, filename: str) -> bool:
        """Check if embedding file exists"""
        embeddings_path = self.storage_path / f"{filename}.npy"
        return embeddings_path.exists()

    def get_embedding_info(self, filename: str) -> Optional[Dict[str, Any]]:
        """Get embedding metadata without loading full embeddings"""
        metadata_path = self.storage_path / f"{filename}_metadata.json"
        if metadata_path.exists():
            try:
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception:
                return None
        return None

    def cleanup_old_embeddings(self, keep_latest: int = 5):
        """Clean up old embedding files"""
        try:
            # Get all embedding files
            embedding_files = list(self.storage_path.glob("*.npy"))

            if len(embedding_files) > keep_latest:
                # Sort by modification time
                embedding_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)

                # Remove old files
                for file_path in embedding_files[keep_latest:]:
                    file_path.unlink()

                    # Also remove associated files
                    base_name = file_path.stem
                    for suffix in ['_ids.json', '_metadata.json']:
                        associated_file = self.storage_path / f"{base_name}{suffix}"
                        if associated_file.exists():
                            associated_file.unlink()

                self.logger.info(f"Cleaned up {len(embedding_files) - keep_latest} old embedding files")

        except Exception as e:
            self.logger.error(f"Failed to cleanup old embeddings: {e}")


class EmbeddingManager:
    """High-level manager for embedding operations"""

    def __init__(self, storage_path: Path, config: EmbeddingConfig = None):
        self.config = config or EmbeddingConfig()
        self.generator = EmbeddingGenerator(self.config)
        self.storage = EmbeddingStorage(storage_path)
        self.logger = logging.getLogger(__name__)

    def generate_and_save_all(self, sections: List[DocumentSection],
                             symbols: List[CodeSymbol]) -> Dict[str, Any]:
        """Generate and save all embeddings"""
        results = {}

        try:
            # Generate document embeddings
            if sections:
                doc_embeddings, doc_ids, doc_metadata = self.generator.generate_document_embeddings(sections)
                self.storage.save_embeddings(doc_embeddings, doc_ids, doc_metadata, "doc_embeddings")
                results['documents'] = {
                    'count': len(doc_embeddings),
                    'dimension': self.generator.model.embedding_dimension
                }

            # Generate code embeddings
            if symbols:
                code_embeddings, code_ids, code_metadata = self.generator.generate_code_embeddings(symbols)
                self.storage.save_embeddings(code_embeddings, code_ids, code_metadata, "code_embeddings")
                results['code'] = {
                    'count': len(code_embeddings),
                    'dimension': self.generator.model.embedding_dimension
                }

            # Cleanup old embeddings
            self.storage.cleanup_old_embeddings()

            return results

        except Exception as e:
            self.logger.error(f"Failed to generate and save embeddings: {e}")
            raise

    def load_all_embeddings(self) -> Dict[str, Any]:
        """Load all available embeddings"""
        embeddings = {}

        # Load document embeddings
        if self.storage.embedding_exists("doc_embeddings"):
            try:
                doc_embeddings, doc_ids, doc_metadata = self.storage.load_embeddings("doc_embeddings")
                embeddings['documents'] = {
                    'embeddings': doc_embeddings,
                    'ids': doc_ids,
                    'metadata': doc_metadata
                }
            except Exception as e:
                self.logger.error(f"Failed to load document embeddings: {e}")

        # Load code embeddings
        if self.storage.embedding_exists("code_embeddings"):
            try:
                code_embeddings, code_ids, code_metadata = self.storage.load_embeddings("code_embeddings")
                embeddings['code'] = {
                    'embeddings': code_embeddings,
                    'ids': code_ids,
                    'metadata': code_metadata
                }
            except Exception as e:
                self.logger.error(f"Failed to load code embeddings: {e}")

        return embeddings

    def get_embedding_stats(self) -> Dict[str, Any]:
        """Get statistics about stored embeddings"""
        stats = {
            'model_name': self.config.model_name,
            'embedding_dimension': self.generator.model.embedding_dimension if self.generator.model else 0,
            'documents': {},
            'code': {}
        }

        # Document embeddings stats
        doc_info = self.storage.get_embedding_info("doc_embeddings")
        if doc_info:
            stats['documents'] = {
                'total_embeddings': doc_info.get('total_embeddings', 0),
                'creation_time': doc_info.get('creation_time', 0),
                'model_used': doc_info.get('model_name', 'unknown')
            }

        # Code embeddings stats
        code_info = self.storage.get_embedding_info("code_embeddings")
        if code_info:
            stats['code'] = {
                'total_embeddings': code_info.get('total_embeddings', 0),
                'creation_time': code_info.get('creation_time', 0),
                'model_used': code_info.get('model_name', 'unknown')
            }

        return stats
