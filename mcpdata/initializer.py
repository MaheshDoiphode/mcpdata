"""
MCP Initializer - Main initialization logic for MCP data directory
"""

import os
import hashlib
import time
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict

from .core.models import FileMetadata, DocumentSection, CodeSymbol
from .core.config import ConfigManager, MCPConfig
from .core.parser import ParserFactory, ContentPreprocessor
from .core.search import SearchIndexBuilder
from .core.embeddings import EmbeddingManager, EmbeddingConfig
from .core.registry import CentralRegistry, GlobalSearchMetadata, get_global_registry


class MCPInitializer:
    """Main initializer for MCP data directory"""

    def __init__(self, root_path: Path, config_path: Optional[str] = None,
                 verbose: bool = False, enable_embeddings: bool = True,
                 parallel_workers: int = 4, register_with_central: bool = True,
                 workspace_name: Optional[str] = None, workspace_description: Optional[str] = None):
        """
        Initialize MCP system

        Args:
            root_path: Root directory to index
            config_path: Optional path to config file
            verbose: Enable verbose logging
            enable_embeddings: Enable embedding generation
            parallel_workers: Number of parallel workers
            register_with_central: Whether to register with central registry
            workspace_name: Name for the workspace (required if register_with_central=True)
            workspace_description: Description for the workspace (required if register_with_central=True)
        """
        self.root_path = Path(root_path).resolve()
        self.mcp_data_path = self.root_path / '.mcpdata'
        self.verbose = verbose
        self.parallel_workers = parallel_workers
        self.register_with_central = register_with_central
        self.workspace_name = workspace_name
        self.workspace_description = workspace_description

        # Initialize configuration
        config_file_path = Path(config_path) if config_path else self.mcp_data_path / 'config.json'
        self.config_manager = ConfigManager(config_file_path)
        self.config = self.config_manager.load_config()

        # Override embedding setting if specified
        if not enable_embeddings:
            self.config.indexing.enable_embeddings = False

        # Initialize central registry if enabled
        self.central_registry = None
        self.workspace_id = None
        if self.register_with_central:
            # Create central registry (will create Documents/mcpdata if needed)
            self.central_registry = get_global_registry()

            # Prompt for workspace details if not provided
            if not self.workspace_name:
                self.workspace_name = self._prompt_workspace_name()
            if not self.workspace_description:
                self.workspace_description = self._prompt_workspace_description()

        # Initialize components
        self.parser_factory = ParserFactory()
        self.preprocessor = ContentPreprocessor()
        self.search_builder = SearchIndexBuilder()

        # Initialize embedding manager if enabled
        self.embedding_manager = None
        if self.config.indexing.enable_embeddings:
            embedding_config = EmbeddingConfig(
                model_name=self.config.indexing.embedding_model,
                batch_size=self.config.indexing.embedding_batch_size,
                parallel_workers=self.config.indexing.parallel_workers
            )
            self.embedding_manager = EmbeddingManager(
                self.mcp_data_path / 'vectors',
                embedding_config
            )

        # Initialize logging
        self.logger = self._setup_logging()

        # Statistics tracking
        self.stats = {
            'files_processed': 0,
            'documents_parsed': 0,
            'code_files_analyzed': 0,
            'symbols_extracted': 0,
            'sections_created': 0,
            'embeddings_generated': 0,
            'files_ignored': 0,
            'errors_encountered': 0
        }

    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration"""
        logger = logging.getLogger('mcp_initializer')
        logger.setLevel(logging.DEBUG if self.verbose else logging.INFO)

        # Clear existing handlers
        logger.handlers.clear()

        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        )

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG if self.verbose else logging.INFO)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # File handler (will be created when .mcpdata directory exists)
        if self.mcp_data_path.exists():
            log_dir = self.mcp_data_path / 'logs'
            log_dir.mkdir(exist_ok=True)

            file_handler = logging.FileHandler(log_dir / 'initialization.log')
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

        return logger

    def _should_ignore_file(self, file_path: Path) -> bool:
        """Check if file should be ignored based on configuration"""
        try:
            # Get relative path for pattern matching
            relative_path = file_path.relative_to(self.root_path)
            path_str = str(relative_path)
            path_parts = relative_path.parts

            # Check against ignored patterns
            for pattern in self.config.ignored_patterns:
                if pattern.startswith('*'):
                    # Wildcard pattern
                    if path_str.endswith(pattern[1:]):
                        return True
                elif pattern.endswith('/'):
                    # Directory pattern
                    dir_name = pattern[:-1]
                    if dir_name in path_parts:
                        return True
                else:
                    # Exact match pattern
                    if pattern in path_str or pattern in path_parts:
                        return True

            # Check file size
            if file_path.stat().st_size > self.config.indexing.max_file_size:
                self.logger.warning(f"Ignoring large file: {file_path}")
                return True

            # Additional checks for hidden files
            if file_path.name.startswith('.'):
                # Allow common config files
                allowed_hidden = {'.gitignore', '.env', '.env.example', '.dockerignore'}
                if file_path.name not in allowed_hidden and file_path.suffix not in ['.md', '.txt', '.json', '.yaml', '.yml']:
                    return True

            return False

        except Exception as e:
            self.logger.error(f"Error checking ignore status for {file_path}: {e}")
            return True

    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA256 hash of file content"""
        try:
            hash_sha256 = hashlib.sha256()
            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    hash_sha256.update(chunk)
            return hash_sha256.hexdigest()
        except Exception as e:
            self.logger.error(f"Error calculating hash for {file_path}: {e}")
            return ""

    def _detect_file_type(self, file_path: Path) -> str:
        """Detect file type based on extension"""
        suffix = file_path.suffix.lower()

        for file_type, extensions in self.config.file_types.items():
            if suffix in extensions:
                return file_type

        return 'unknown'

    def _detect_encoding(self, file_path: Path) -> str:
        """Detect file encoding"""
        try:
            import chardet
            with open(file_path, 'rb') as f:
                raw_data = f.read(10000)  # Read first 10KB
                result = chardet.detect(raw_data)
                return result.get('encoding', 'utf-8') or 'utf-8'
        except ImportError:
            # Fallback without chardet
            encodings = ['utf-8', 'latin-1', 'cp1252']
            for encoding in encodings:
                try:
                    with open(file_path, 'r', encoding=encoding) as f:
                        f.read(1000)  # Try to read a portion
                    return encoding
                except UnicodeDecodeError:
                    continue
            return 'utf-8'
        except Exception:
            return 'utf-8'

    def scan_files(self) -> List[FileMetadata]:
        """Scan all files in the directory and create metadata"""
        self.logger.info(f"Starting file scan in {self.root_path}")

        files_metadata = []

        for file_path in self.root_path.rglob('*'):
            if not file_path.is_file():
                continue

            if self._should_ignore_file(file_path):
                self.stats['files_ignored'] += 1
                continue

            try:
                stat = file_path.stat()
                file_type = self._detect_file_type(file_path)
                encoding = self._detect_encoding(file_path)

                file_metadata = FileMetadata(
                    path=str(file_path.relative_to(self.root_path)),
                    size=stat.st_size,
                    modified=stat.st_mtime,
                    file_type=file_type,
                    hash=self._calculate_file_hash(file_path),
                    language=None,  # Could be enhanced with language detection
                    encoding=encoding
                )

                files_metadata.append(file_metadata)
                self.stats['files_processed'] += 1

                # Log progress
                if self.verbose:
                    self.logger.debug(f"Processed {file_type} file: {file_path}")
                elif file_type in ['documents', 'code']:
                    self.logger.info(f"Found {file_type} file: {file_path}")

            except Exception as e:
                self.logger.error(f"Error processing {file_path}: {e}")
                self.stats['errors_encountered'] += 1

        # Log summary
        type_counts = {}
        for fm in files_metadata:
            type_counts[fm.file_type] = type_counts.get(fm.file_type, 0) + 1

        self.logger.info(f"Scanned {len(files_metadata)} files, ignored {self.stats['files_ignored']} files")
        self.logger.info(f"File types found: {type_counts}")

        return files_metadata

    def parse_file_content(self, file_path: Path, file_metadata: FileMetadata) -> List[Any]:
        """Parse file content using appropriate parser"""
        try:
            # Read file content
            with open(file_path, 'r', encoding=file_metadata.encoding) as f:
                content = f.read()

            # Use parser factory to get appropriate parser
            result = self.parser_factory.parse_file(file_path, content)

            if result is None:
                return []

            # Update statistics
            if isinstance(result, list) and len(result) > 0:
                if isinstance(result[0], DocumentSection):
                    self.stats['documents_parsed'] += 1
                    self.stats['sections_created'] += len(result)
                elif isinstance(result[0], CodeSymbol):
                    self.stats['code_files_analyzed'] += 1
                    self.stats['symbols_extracted'] += len(result)

            return result

        except Exception as e:
            self.logger.error(f"Error parsing {file_path}: {e}")
            self.stats['errors_encountered'] += 1
            return []

    def create_directory_structure(self):
        """Create .mcpdata directory structure"""
        self.logger.info("Creating directory structure")

        # Create main directory
        self.mcp_data_path.mkdir(exist_ok=True)

        # Create subdirectories
        subdirs = ['index', 'metadata', 'cache', 'vectors', 'logs']
        for subdir in subdirs:
            (self.mcp_data_path / subdir).mkdir(exist_ok=True)

        # Save configuration
        self.config_manager.save_config(self.config)

        # Update logger to include file handler now that directory exists
        if not any(isinstance(h, logging.FileHandler) for h in self.logger.handlers):
            log_dir = self.mcp_data_path / 'logs'
            file_handler = logging.FileHandler(log_dir / 'initialization.log')
            file_handler.setLevel(logging.DEBUG)
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)

    def create_search_indices(self, files_metadata: List[FileMetadata],
                              sections: List[DocumentSection], symbols: List[CodeSymbol]):
        """Create search indices for fast lookup"""
        self.logger.info("Creating search indices")

        index_dir = self.mcp_data_path / 'index'

        # Build inverted index for sections
        search_index = self.search_builder.build_inverted_index(sections)

        # Create content index
        content_index = {
            'files': {f.path: f.hash for f in files_metadata},
            'sections': {
                s.id: {
                    'title': s.title,
                    'content': s.content,
                    'file_path': s.file_path or 'unknown'
                } for s in sections
            },
            'symbols': {
                f"{s.file_path}:{s.name}": s.signature or '' for s in symbols
            }
        }

        # Create advanced search index
        advanced_index = {
            'inverted_index': search_index.inverted_index,
            'term_frequencies': {k: dict(v) for k, v in search_index.term_frequencies.items()},
            'document_frequencies': search_index.document_frequencies,
            'section_lengths': search_index.section_lengths,
            'total_sections': search_index.total_sections,
            'section_metadata': self.search_builder.create_search_metadata(sections)
        }

        # Save indices
        import json

        with open(index_dir / 'content.idx', 'w', encoding='utf-8') as f:
            json.dump(content_index, f, indent=2, ensure_ascii=False)

        with open(index_dir / 'search.idx', 'w', encoding='utf-8') as f:
            json.dump(advanced_index, f, indent=2, ensure_ascii=False)

        self.logger.info("Search indices created successfully")

    def generate_embeddings(self, sections: List[DocumentSection], symbols: List[CodeSymbol]):
        """Generate vector embeddings for semantic search"""
        if not self.embedding_manager:
            self.logger.info("Embedding generation disabled")
            return

        self.logger.info("Generating embeddings")

        try:
            # Generate and save embeddings
            results = self.embedding_manager.generate_and_save_all(sections, symbols)

            # Update statistics
            total_embeddings = 0
            for content_type, result in results.items():
                total_embeddings += result.get('count', 0)

            self.stats['embeddings_generated'] = total_embeddings
            self.logger.info(f"Generated {total_embeddings} embeddings")

        except Exception as e:
            self.logger.error(f"Error generating embeddings: {e}")
            self.stats['errors_encountered'] += 1

    def save_metadata(self, files_metadata: List[FileMetadata],
                      sections: List[DocumentSection], symbols: List[CodeSymbol]):
        """Save all metadata to JSON files"""
        self.logger.info("Saving metadata")

        metadata_dir = self.mcp_data_path / 'metadata'

        try:
            import json

            # Save file metadata
            with open(metadata_dir / 'files.json', 'w', encoding='utf-8') as f:
                json.dump([asdict(fm) for fm in files_metadata], f, indent=2, ensure_ascii=False)

            # Save document sections
            with open(metadata_dir / 'sections.json', 'w', encoding='utf-8') as f:
                json.dump([asdict(s) for s in sections], f, indent=2, ensure_ascii=False)

            # Save code symbols
            with open(metadata_dir / 'symbols.json', 'w', encoding='utf-8') as f:
                json.dump([asdict(s) for s in symbols], f, indent=2, ensure_ascii=False)

            self.logger.info("Metadata saved successfully")

        except Exception as e:
            self.logger.error(f"Error saving metadata: {e}")
            self.stats['errors_encountered'] += 1

    def create_cache(self, sections: List[DocumentSection]):
        """Create cached content for fast retrieval"""
        self.logger.info("Creating content cache")

        cache_dir = self.mcp_data_path / 'cache'

        try:
            import json

            # Create chunks for different token limits
            token_limits = [1000, 2000, 4000, 8000]

            for limit in token_limits:
                chunks = self._create_content_chunks(sections, limit)

                with open(cache_dir / f'chunks_{limit}.json', 'w', encoding='utf-8') as f:
                    json.dump(chunks, f, indent=2, ensure_ascii=False)

            # Create section summaries
            summaries = {}
            for section in sections:
                summaries[section.id] = {
                    'title': section.title,
                    'file_path': section.file_path,
                    'line_range': [section.start_line, section.end_line],
                    'content_preview': section.content[:200] + ('...' if len(section.content) > 200 else ''),
                    'word_count': len(section.content.split())
                }

            with open(cache_dir / 'summaries.json', 'w', encoding='utf-8') as f:
                json.dump(summaries, f, indent=2, ensure_ascii=False)

            self.logger.info(f"Created cache for {len(token_limits)} token limits")

        except Exception as e:
            self.logger.error(f"Error creating cache: {e}")
            self.stats['errors_encountered'] += 1

    def _create_content_chunks(self, sections: List[DocumentSection], token_limit: int) -> Dict[str, Any]:
        """Create content chunks optimized for specific token limits"""
        chunks = {}
        current_chunk = []
        current_tokens = 0

        for section in sections:
            # Rough token estimation (4 characters per token)
            section_tokens = len(section.content) // 4

            if current_tokens + section_tokens <= token_limit:
                current_chunk.append(section)
                current_tokens += section_tokens
            else:
                if current_chunk:
                    chunks[f"chunk_{len(chunks)}"] = {
                        'sections': [s.id for s in current_chunk],
                        'content': '\n\n'.join([f"# {s.title}\n{s.content}" for s in current_chunk]),
                        'token_count': current_tokens,
                        'section_count': len(current_chunk)
                    }

                current_chunk = [section]
                current_tokens = section_tokens

        # Add final chunk
        if current_chunk:
            chunks[f"chunk_{len(chunks)}"] = {
                'sections': [s.id for s in current_chunk],
                'content': '\n\n'.join([f"# {s.title}\n{s.content}" for s in current_chunk]),
                'token_count': current_tokens,
                'section_count': len(current_chunk)
            }

        return chunks

    def initialize(self) -> Dict[str, Any]:
        """
        Main initialization process

        Returns:
            Dictionary with initialization statistics
        """
        start_time = time.time()
        self.logger.info("Starting MCP initialization")

        try:
            # Phase 1: Setup
            self.logger.info("Phase 1: Setting up directory structure")
            self.create_directory_structure()

            # Register workspace with central registry if enabled
            if self.register_with_central and self.central_registry:
                self.logger.info("Registering workspace with central registry")
                self.workspace_id = self._register_workspace()
                self.logger.info(f"Workspace registered with ID: {self.workspace_id}")

            # Phase 2: File scanning
            self.logger.info("Phase 2: Scanning files")
            files_metadata = self.scan_files()

            if not files_metadata:
                self.logger.warning("No files found to process")
                return self.stats

            # Phase 3: Content parsing
            self.logger.info("Phase 3: Parsing content")
            all_sections = []
            all_symbols = []

            # Process files in parallel
            with ThreadPoolExecutor(max_workers=self.parallel_workers) as executor:
                futures = []

                for file_metadata in files_metadata:
                    if file_metadata.file_type in ['documents', 'code']:
                        file_path = self.root_path / file_metadata.path
                        future = executor.submit(self.parse_file_content, file_path, file_metadata)
                        futures.append(future)

                # Collect results
                for future in as_completed(futures):
                    try:
                        result = future.result()
                        if result:
                            if isinstance(result[0], DocumentSection):
                                all_sections.extend(result)
                            elif isinstance(result[0], CodeSymbol):
                                all_symbols.extend(result)
                    except Exception as e:
                        self.logger.error(f"Error processing future result: {e}")
                        self.stats['errors_encountered'] += 1

            # Phase 4: Create search indices
            self.logger.info("Phase 4: Creating search indices")
            self.create_search_indices(files_metadata, all_sections, all_symbols)

            # Phase 5: Generate embeddings
            self.logger.info("Phase 5: Generating embeddings")
            self.generate_embeddings(all_sections, all_symbols)

            # Phase 6: Save metadata
            self.logger.info("Phase 6: Saving metadata")
            self.save_metadata(files_metadata, all_sections, all_symbols)

            # Phase 7: Create cache
            self.logger.info("Phase 7: Creating cache")
            self.create_cache(all_sections)

            # Phase 8: Update central registry
            if self.register_with_central and self.central_registry and self.workspace_id:
                self.logger.info("Updating central registry")
                parsed_data = {
                    'sections': all_sections,
                    'symbols': all_symbols,
                    'files': files_metadata
                }
                self._update_central_registry(parsed_data)
                self.logger.info("Central registry updated")

            # Final statistics
            duration = time.time() - start_time
            self.logger.info(f"Initialization completed in {duration:.2f} seconds")
            self.logger.info(f"Final statistics: {self.stats}")

            # Add timing to stats
            self.stats['initialization_time'] = duration

            return self.stats

        except Exception as e:
            self.logger.error(f"Initialization failed: {e}")
            self.stats['errors_encountered'] += 1
            raise

    def _prompt_workspace_name(self) -> str:
        """Prompt user for workspace name"""
        default_name = self.root_path.name
        while True:
            name = input(f"\nEnter workspace name (default: {default_name}): ").strip()
            if not name:
                name = default_name
            if name:
                return name
            print("Workspace name cannot be empty. Please try again.")

    def _prompt_workspace_description(self) -> str:
        """Prompt user for workspace description"""
        while True:
            description = input("Enter workspace description: ").strip()
            if description:
                return description
            print("Workspace description is required. Please try again.")

    def _register_workspace(self) -> str:
        """Register workspace with central registry"""
        if not self.central_registry:
            raise ValueError("Central registry not initialized")

        # Detect project type
        project_type = self.config_manager.detect_project_type(self.root_path)

        # Generate tags based on project type and content
        tags = [project_type]

        # Add additional tags based on directory structure
        if (self.root_path / "docs").exists():
            tags.append("documentation")
        if (self.root_path / "test").exists() or (self.root_path / "tests").exists():
            tags.append("testing")
        if (self.root_path / "api").exists():
            tags.append("api")

        # Register with central registry
        workspace_id = self.central_registry.register_workspace(
            workspace_path=str(self.root_path),
            name=self.workspace_name,
            description=self.workspace_description,
            project_type=project_type,
            tags=tags,
            force=True  # Allow re-registration
        )

        return workspace_id

    def _update_central_registry(self, parsed_data: dict) -> None:
        """Update central registry with workspace stats and global search data"""
        if not self.central_registry or not self.workspace_id:
            return

        # Calculate index size
        index_size_mb = 0.0
        if self.mcp_data_path.exists():
            index_size_mb = sum(f.stat().st_size for f in self.mcp_data_path.rglob('*') if f.is_file()) / (1024 * 1024)

        # Update workspace statistics
        workspace_stats = {
            'file_count': self.stats['files_processed'],
            'document_count': self.stats['documents_parsed'],
            'code_symbol_count': self.stats['symbols_extracted'],
            'embedding_count': self.stats['embeddings_generated'],
            'index_size_mb': index_size_mb,
            'health_score': 1.0  # TODO: Calculate based on processing success rate
        }

        self.central_registry.update_workspace_stats(self.workspace_id, workspace_stats)

        # Create global search metadata
        global_search_data = []

        # Add document sections
        for section in parsed_data.get('sections', []):
            if isinstance(section, DocumentSection):
                # Extract keywords from content
                keywords = self._extract_keywords(section.content)

                # Create preview
                content_preview = section.content[:200] + "..." if len(section.content) > 200 else section.content

                search_meta = GlobalSearchMetadata(
                    workspace_id=self.workspace_id,
                    workspace_name=self.workspace_name,
                    section_id=section.id,
                    title=section.title,
                    content_preview=content_preview,
                    file_path=section.file_path or "",
                    section_type="document",
                    keywords=keywords
                )
                global_search_data.append(search_meta)

        # Add code symbols
        for symbol in parsed_data.get('symbols', []):
            if isinstance(symbol, CodeSymbol):
                keywords = [symbol.name, symbol.type, symbol.scope]
                if symbol.docstring:
                    keywords.extend(self._extract_keywords(symbol.docstring))

                content_preview = f"{symbol.type} {symbol.name}({symbol.signature})"
                if symbol.docstring:
                    content_preview += f" - {symbol.docstring[:100]}"

                search_meta = GlobalSearchMetadata(
                    workspace_id=self.workspace_id,
                    workspace_name=self.workspace_name,
                    section_id=f"{symbol.file_path}:{symbol.line_number}",
                    title=symbol.name,
                    content_preview=content_preview,
                    file_path=symbol.file_path,
                    section_type="code",
                    keywords=keywords
                )
                global_search_data.append(search_meta)

        # Update global search data
        self.central_registry.update_global_search_data(self.workspace_id, global_search_data)

    def _extract_keywords(self, content: str) -> List[str]:
        """Extract keywords from content"""
        import re

        # Simple keyword extraction - can be improved
        words = re.findall(r'\b[a-zA-Z]{3,}\b', content.lower())

        # Remove common stop words
        stop_words = {
            'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had', 'her', 'was', 'one',
            'our', 'out', 'day', 'get', 'has', 'him', 'his', 'how', 'its', 'new', 'now', 'old', 'see',
            'two', 'who', 'boy', 'did', 'men', 'oil', 'sit', 'way', 'who', 'ask', 'big', 'cut', 'few',
            'got', 'hot', 'let', 'man', 'may', 'not', 'put', 'run', 'say', 'she', 'too', 'use', 'was',
            'win', 'yes', 'yet', 'you', 'add', 'ago', 'air', 'arm', 'art', 'ask', 'bad', 'bag', 'bed',
            'box', 'boy', 'bus', 'buy', 'car', 'cat', 'cup', 'cut', 'day', 'dog', 'ear', 'eat', 'end',
            'eye', 'far', 'fly', 'for', 'fun', 'get', 'god', 'got', 'gun', 'hat', 'him', 'hit', 'hot',
            'how', 'job', 'key', 'law', 'let', 'lot', 'low', 'man', 'map', 'may', 'mom', 'new', 'not',
            'now', 'off', 'old', 'one', 'our', 'out', 'own', 'pay', 'put', 'red', 'run', 'say', 'see',
            'she', 'sit', 'six', 'son', 'sun', 'ten', 'the', 'too', 'top', 'try', 'two', 'use', 'war',
            'way', 'who', 'why', 'win', 'yes', 'yet', 'you'
        }

        # Filter and count
        from collections import Counter
        word_counts = Counter(word for word in words if word not in stop_words and len(word) > 2)

        # Return top keywords
        return [word for word, count in word_counts.most_common(10)]

    def update_incremental(self) -> Dict[str, Any]:
        """
        Perform incremental update (only process changed files)

        Returns:
            Dictionary with update statistics
        """
        self.logger.info("Starting incremental update")

        # Load existing metadata
        metadata_dir = self.mcp_data_path / 'metadata'
        existing_files = {}

        if (metadata_dir / 'files.json').exists():
            import json
            with open(metadata_dir / 'files.json', 'r', encoding='utf-8') as f:
                existing_data = json.load(f)
                existing_files = {item['path']: item['hash'] for item in existing_data}

        # Scan for changes
        current_files = self.scan_files()
        changed_files = []

        for file_meta in current_files:
            if (file_meta.path not in existing_files or
                    existing_files[file_meta.path] != file_meta.hash):
                changed_files.append(file_meta)

        if not changed_files:
            self.logger.info("No changes detected")
            return {'changed_files': 0, 'update_time': 0}

        self.logger.info(f"Processing {len(changed_files)} changed files")

        # Process only changed files
        # This is a simplified version - in practice, you'd need to
        # handle deletions and update indices incrementally

        # For now, just re-run full initialization
        return self.initialize()
