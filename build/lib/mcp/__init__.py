#!/usr/bin/env python3
"""
MCP Data Initialization Script
Creates .mcpdata directory with pre-computed indices for fast MCP responses
"""

import os
import json
import hashlib
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

# External dependencies (would need to be installed)
# pip install tree-sitter markdown whoosh sentence-transformers

@dataclass
class FileMetadata:
    path: str
    size: int
    modified: float
    file_type: str
    hash: str
    language: Optional[str] = None
    encoding: str = 'utf-8'

@dataclass
class DocumentSection:
    id: str
    title: str
    level: int
    start_line: int
    end_line: int
    content: str
    parent_id: Optional[str] = None
    children: List[str] = None

@dataclass
class CodeSymbol:
    name: str
    type: str  # function, class, variable, etc.
    file_path: str
    line_number: int
    column: int
    scope: str
    signature: Optional[str] = None
    docstring: Optional[str] = None

class MCPInitializer:
    def __init__(self, root_path: str):
        self.root_path = Path(root_path)
        self.mcp_data_path = self.root_path / '.mcpdata'
        self.config = self._load_or_create_config()
        self.logger = self._setup_logging()

        # Statistics
        self.stats = {
            'files_processed': 0,
            'documents_parsed': 0,
            'code_files_analyzed': 0,
            'symbols_extracted': 0,
            'sections_created': 0,
            'embeddings_generated': 0
        }

    def _setup_logging(self) -> logging.Logger:
        """Setup logging to .mcpdata/logs/"""
        log_dir = self.mcp_data_path / 'logs'
        log_dir.mkdir(parents=True, exist_ok=True)

        logger = logging.getLogger('mcp_init')
        logger.setLevel(logging.INFO)

        handler = logging.FileHandler(log_dir / 'last_index.log')
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        # Also log to console
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        return logger

    def _load_or_create_config(self) -> Dict[str, Any]:
        """Load existing config or create default one"""
        config_path = self.mcp_data_path / 'config.json'

        default_config = {
            'version': '1.0.0',
            'project_type': 'auto',  # auto, docs, code, mixed
            'ignored_patterns': [
                # Build artifacts
                'build/', 'dist/', 'target/', 'out/', 'bin/', 'obj/',
                # Dependencies
                'node_modules/', '.gradle/', '.m2/', 'vendor/',
                # Version control
                '.git/', '.svn/', '.hg/',
                # IDE files
                '.idea/', '.vscode/', '*.iml', '*.swp', '*.swo',
                # OS files
                '.DS_Store', 'Thumbs.db', '*.tmp', '*.log',
                # Compiled files
                '*.pyc', '__pycache__/', '*.class', '*.jar', '*.war',
                '*.exe', '*.dll', '*.so', '*.dylib',
                # Archive files
                '*.zip', '*.tar', '*.gz', '*.7z', '*.rar',
                # Generated files
                'gradlew', 'gradlew.bat', 'mvnw', 'mvnw.cmd',
                # Our own data
                '.mcpdata'
            ],
            'file_types': {
                'documents': ['.md', '.rst', '.txt', '.adoc', '.asciidoc'],
                'code': [
                    # JVM languages
                    '.java', '.kt', '.kts', '.scala', '.groovy',
                    # Web languages
                    '.js', '.ts', '.jsx', '.tsx', '.html', '.css', '.scss',
                    # Systems languages
                    '.c', '.cpp', '.cc', '.cxx', '.h', '.hpp', '.hxx',
                    '.rust', '.rs', '.go', '.zig',
                    # Scripting languages
                    '.py', '.rb', '.pl', '.sh', '.bash', '.zsh', '.fish',
                    # Functional languages
                    '.hs', '.elm', '.clj', '.cljs', '.ml', '.fs',
                    # Other
                    '.php', '.swift', '.dart', '.r'
                ],
                'config': ['.json', '.yaml', '.yml', '.toml', '.ini', '.properties', '.gradle', '.xml', '.plist']
            },
            'max_file_size': 5 * 1024 * 1024,  # 5MB (increased from 1MB)
            'enable_embeddings': True,
            'embedding_model': 'all-MiniLM-L6-v2',
            'chunk_size': 512,
            'chunk_overlap': 50
        }

        if config_path.exists():
            with open(config_path, 'r') as f:
                config = json.load(f)
                # Merge with defaults for any missing keys
                for key, value in default_config.items():
                    if key not in config:
                        config[key] = value
        else:
            config = default_config
            # Don't create directory here - wait for actual initialization

        return config

    def _should_ignore_file(self, file_path: Path) -> bool:
        """Check if file should be ignored based on patterns"""
        file_str = str(file_path.relative_to(self.root_path))
        file_parts = file_path.parts

        for pattern in self.config['ignored_patterns']:
            # Handle wildcard patterns
            if pattern.startswith('*'):
                if file_str.endswith(pattern[1:]):
                    return True
            # Handle directory patterns (ending with /)
            elif pattern.endswith('/'):
                dir_pattern = pattern[:-1]
                if dir_pattern in file_parts:
                    return True
            # Handle exact matches
            else:
                if pattern in file_str or pattern in file_parts:
                    return True

        # Additional logic: ignore hidden files/directories unless they're common config files
        if file_path.name.startswith('.') and file_path.suffix not in ['.md', '.txt', '.json', '.yaml', '.yml']:
            return True

        return False

    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA256 hash of file content"""
        hash_sha256 = hashlib.sha256()
        try:
            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_sha256.update(chunk)
            return hash_sha256.hexdigest()
        except Exception:
            return ""

    def _detect_file_type(self, file_path: Path) -> str:
        """Detect file type based on extension"""
        suffix = file_path.suffix.lower()

        for file_type, extensions in self.config['file_types'].items():
            if suffix in extensions:
                return file_type

        return 'unknown'

    def scan_files(self) -> List[FileMetadata]:
        """Scan all files in the directory and create metadata"""
        self.logger.info(f"Starting file scan in {self.root_path}")

        files_metadata = []
        ignored_count = 0

        for file_path in self.root_path.rglob('*'):
            if not file_path.is_file():
                continue

            if self._should_ignore_file(file_path):
                ignored_count += 1
                continue

            try:
                stat = file_path.stat()
                if stat.st_size > self.config['max_file_size']:
                    self.logger.warning(f"Skipping large file: {file_path} ({stat.st_size} bytes)")
                    continue

                file_type = self._detect_file_type(file_path)

                file_metadata = FileMetadata(
                    path=str(file_path.relative_to(self.root_path)),
                    size=stat.st_size,
                    modified=stat.st_mtime,
                    file_type=file_type,
                    hash=self._calculate_file_hash(file_path)
                )

                files_metadata.append(file_metadata)
                self.stats['files_processed'] += 1

                # Log progress for different file types
                if file_type == 'code':
                    self.logger.info(f"Found code file: {file_path}")
                elif file_type == 'documents':
                    self.logger.info(f"Found document: {file_path}")

            except Exception as e:
                self.logger.error(f"Error processing {file_path}: {e}")

        self.logger.info(f"Scanned {len(files_metadata)} files, ignored {ignored_count} files")

        # Log file type breakdown
        type_counts = {}
        for fm in files_metadata:
            type_counts[fm.file_type] = type_counts.get(fm.file_type, 0) + 1

        self.logger.info(f"File types found: {type_counts}")

        return files_metadata

    def parse_markdown_document(self, file_path: Path) -> List[DocumentSection]:
        """Parse markdown document into sections"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            lines = content.split('\n')
            sections = []
            current_section = None
            section_stack = []

            for i, line in enumerate(lines):
                if line.strip().startswith('#'):
                    # Save previous section
                    if current_section:
                        current_section.end_line = i - 1
                        current_section.content = '\n'.join(lines[current_section.start_line:i])
                        sections.append(current_section)

                    # Parse header
                    level = len(line) - len(line.lstrip('#'))
                    title = line.strip('#').strip()

                    # Handle section hierarchy
                    while section_stack and section_stack[-1][1] >= level:
                        section_stack.pop()

                    parent_id = section_stack[-1][0] if section_stack else None
                    section_id = f"{file_path.stem}_{i}_{level}"

                    current_section = DocumentSection(
                        id=section_id,
                        title=title,
                        level=level,
                        start_line=i,
                        end_line=len(lines),
                        content="",
                        parent_id=parent_id,
                        children=[]
                    )

                    section_stack.append((section_id, level))

                    # Update parent's children
                    if parent_id:
                        for section in sections:
                            if section.id == parent_id:
                                if section.children is None:
                                    section.children = []
                                section.children.append(section_id)
                                break

            # Handle last section
            if current_section:
                current_section.end_line = len(lines)
                current_section.content = '\n'.join(lines[current_section.start_line:])
                sections.append(current_section)

            self.stats['documents_parsed'] += 1
            self.stats['sections_created'] += len(sections)

            return sections

        except Exception as e:
            self.logger.error(f"Error parsing markdown {file_path}: {e}")
            return []

    def extract_code_symbols(self, file_path: Path) -> List[CodeSymbol]:
        """Extract symbols from code files (supports multiple languages)"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            self.logger.error(f"Error reading {file_path}: {e}")
            return []

        symbols = []
        lines = content.split('\n')
        file_path_str = str(file_path.relative_to(self.root_path))

        if file_path.suffix == '.py':
            symbols = self._extract_python_symbols(lines, file_path_str)
        elif file_path.suffix in ['.java', '.kt', '.kts']:
            symbols = self._extract_jvm_symbols(lines, file_path_str)
        elif file_path.suffix in ['.js', '.ts', '.jsx', '.tsx']:
            symbols = self._extract_javascript_symbols(lines, file_path_str)
        elif file_path.suffix in ['.c', '.cpp', '.cc', '.cxx', '.h', '.hpp']:
            symbols = self._extract_c_symbols(lines, file_path_str)

        self.stats['code_files_analyzed'] += 1
        self.stats['symbols_extracted'] += len(symbols)

        return symbols

    def _extract_python_symbols(self, lines: List[str], file_path: str) -> List[CodeSymbol]:
        """Extract Python symbols"""
        symbols = []

        for i, line in enumerate(lines):
            stripped = line.strip()

            # Functions
            if stripped.startswith('def '):
                func_name = stripped.split('(')[0].replace('def ', '')
                symbols.append(CodeSymbol(
                    name=func_name,
                    type='function',
                    file_path=file_path,
                    line_number=i + 1,
                    column=line.find('def'),
                    scope=self._get_scope(lines, i),
                    signature=stripped
                ))

            # Classes
            elif stripped.startswith('class '):
                class_name = stripped.split('(')[0].replace('class ', '').rstrip(':')
                symbols.append(CodeSymbol(
                    name=class_name,
                    type='class',
                    file_path=file_path,
                    line_number=i + 1,
                    column=line.find('class'),
                    scope='global',
                    signature=stripped
                ))

        return symbols

    def _extract_jvm_symbols(self, lines: List[str], file_path: str) -> List[CodeSymbol]:
        """Extract Java/Kotlin symbols"""
        symbols = []
        current_class = None

        for i, line in enumerate(lines):
            stripped = line.strip()

            # Skip comments and empty lines
            if not stripped or stripped.startswith('//') or stripped.startswith('/*'):
                continue

            # Package declaration
            if stripped.startswith('package '):
                package_name = stripped.replace('package ', '').rstrip(';')
                symbols.append(CodeSymbol(
                    name=package_name,
                    type='package',
                    file_path=file_path,
                    line_number=i + 1,
                    column=0,
                    scope='global',
                    signature=stripped
                ))

            # Class declarations
            elif any(keyword in stripped for keyword in ['class ', 'interface ', 'enum ', 'object ']):
                # Extract class name (handle generics)
                for keyword in ['class ', 'interface ', 'enum ', 'object ']:
                    if keyword in stripped:
                        parts = stripped.split(keyword, 1)
                        if len(parts) > 1:
                            class_part = parts[1].split()[0]
                            class_name = class_part.split('<')[0].split('(')[0]
                            current_class = class_name
                            symbols.append(CodeSymbol(
                                name=class_name,
                                type=keyword.strip(),
                                file_path=file_path,
                                line_number=i + 1,
                                column=line.find(keyword),
                                scope='global',
                                signature=stripped
                            ))
                        break

            # Function declarations
            elif 'fun ' in stripped or ('(' in stripped and ')' in stripped and
                  any(modifier in stripped for modifier in ['public ', 'private ', 'protected ', 'internal '])):

                # Kotlin functions
                if 'fun ' in stripped:
                    fun_part = stripped.split('fun ', 1)
                    if len(fun_part) > 1:
                        func_name = fun_part[1].split('(')[0].strip()
                        symbols.append(CodeSymbol(
                            name=func_name,
                            type='function',
                            file_path=file_path,
                            line_number=i + 1,
                            column=line.find('fun'),
                            scope=current_class or 'global',
                            signature=stripped
                        ))

                # Java methods (basic detection)
                elif ('(' in stripped and ')' in stripped and
                      not stripped.startswith('if') and not stripped.startswith('while')):
                    # Try to extract method name
                    try:
                        # Look for pattern: [modifiers] [return_type] method_name(
                        parts = stripped.split('(')[0].split()
                        if len(parts) >= 2:
                            method_name = parts[-1]
                            if method_name.isidentifier():
                                symbols.append(CodeSymbol(
                                    name=method_name,
                                    type='method',
                                    file_path=file_path,
                                    line_number=i + 1,
                                    column=line.find(method_name),
                                    scope=current_class or 'global',
                                    signature=stripped
                                ))
                    except:
                        pass

        return symbols

    def _extract_javascript_symbols(self, lines: List[str], file_path: str) -> List[CodeSymbol]:
        """Extract JavaScript/TypeScript symbols"""
        symbols = []

        for i, line in enumerate(lines):
            stripped = line.strip()

            # Function declarations
            if stripped.startswith('function '):
                func_name = stripped.split('(')[0].replace('function ', '')
                symbols.append(CodeSymbol(
                    name=func_name,
                    type='function',
                    file_path=file_path,
                    line_number=i + 1,
                    column=line.find('function'),
                    scope='global',
                    signature=stripped
                ))

            # Arrow functions and const/let/var functions
            elif (' => ' in stripped or 'function(' in stripped) and ('const ' in stripped or 'let ' in stripped or 'var ' in stripped):
                for keyword in ['const ', 'let ', 'var ']:
                    if keyword in stripped:
                        var_part = stripped.split(keyword, 1)[1].split('=')[0].strip()
                        symbols.append(CodeSymbol(
                            name=var_part,
                            type='function',
                            file_path=file_path,
                            line_number=i + 1,
                            column=line.find(keyword),
                            scope='global',
                            signature=stripped
                        ))
                        break

            # Classes
            elif stripped.startswith('class '):
                class_name = stripped.split()[1].split('(')[0].split('{')[0]
                symbols.append(CodeSymbol(
                    name=class_name,
                    type='class',
                    file_path=file_path,
                    line_number=i + 1,
                    column=line.find('class'),
                    scope='global',
                    signature=stripped
                ))

        return symbols

    def _extract_c_symbols(self, lines: List[str], file_path: str) -> List[CodeSymbol]:
        """Extract C/C++ symbols"""
        symbols = []

        for i, line in enumerate(lines):
            stripped = line.strip()

            # Skip preprocessor directives and comments
            if stripped.startswith('#') or stripped.startswith('//') or stripped.startswith('/*'):
                continue

            # Function definitions (basic detection)
            if ('(' in stripped and ')' in stripped and '{' in stripped and
                not stripped.startswith('if') and not stripped.startswith('while') and
                not stripped.startswith('for')):

                # Try to extract function name
                try:
                    paren_pos = stripped.find('(')
                    before_paren = stripped[:paren_pos].strip()
                    parts = before_paren.split()
                    if parts:
                        func_name = parts[-1]
                        if func_name.replace('_', '').replace('*', '').isalnum():
                            symbols.append(CodeSymbol(
                                name=func_name,
                                type='function',
                                file_path=file_path,
                                line_number=i + 1,
                                column=line.find(func_name),
                                scope='global',
                                signature=stripped
                            ))
                except:
                    pass

        return symbols

    def _get_scope(self, lines: List[str], line_index: int) -> str:
        """Determine the scope of a symbol based on indentation"""
        current_line = lines[line_index]
        indent_level = len(current_line) - len(current_line.lstrip())

        # Look backwards for class definitions
        for i in range(line_index - 1, -1, -1):
            line = lines[i]
            line_indent = len(line) - len(line.lstrip())

            if line_indent < indent_level and line.strip().startswith('class '):
                class_name = line.strip().split()[1].split('(')[0].rstrip(':')
                return class_name

        return 'global'

    def create_search_indices(self, files_metadata: List[FileMetadata],
                            sections: List[DocumentSection],
                            symbols: List[CodeSymbol]):
        """Create search indices for fast lookup"""
        # This would use Whoosh, Tantivy, or similar in real implementation
        self.logger.info("Creating search indices...")

        # Create index directories
        index_dir = self.mcp_data_path / 'index'
        index_dir.mkdir(exist_ok=True)

        # Simplified index creation (would be more sophisticated)
        content_index = {
            'files': {f.path: f.hash for f in files_metadata},
            'sections': {s.id: {'title': s.title, 'content': s.content} for s in sections},
            'symbols': {f"{s.file_path}:{s.name}": s.signature for s in symbols}
        }

        with open(index_dir / 'content.idx', 'w') as f:
            json.dump(content_index, f, indent=2)

        self.logger.info("Search indices created")

    def generate_embeddings(self, sections: List[DocumentSection], symbols: List[CodeSymbol]):
        """Generate vector embeddings for semantic search"""
        if not self.config['enable_embeddings']:
            return

        self.logger.info("Generating embeddings...")

        # This would use sentence-transformers or similar
        # For now, just placeholder

        vectors_dir = self.mcp_data_path / 'vectors'
        vectors_dir.mkdir(exist_ok=True)

        # Placeholder embeddings
        embeddings = {
            'documents': {s.id: [0.1] * 384 for s in sections},  # 384-dim embeddings
            'code': {f"{s.file_path}:{s.name}": [0.2] * 384 for s in symbols}
        }

        with open(vectors_dir / 'doc_embeddings.bin', 'w') as f:
            json.dump(embeddings, f)

        self.stats['embeddings_generated'] = len(embeddings['documents']) + len(embeddings['code'])
        self.logger.info(f"Generated {self.stats['embeddings_generated']} embeddings")

    def save_metadata(self, files_metadata: List[FileMetadata],
                     sections: List[DocumentSection],
                     symbols: List[CodeSymbol]):
        """Save all metadata to JSON files"""
        metadata_dir = self.mcp_data_path / 'metadata'
        metadata_dir.mkdir(exist_ok=True)

        # Save files metadata
        with open(metadata_dir / 'files.json', 'w') as f:
            json.dump([asdict(fm) for fm in files_metadata], f, indent=2)

        # Save sections
        with open(metadata_dir / 'sections.json', 'w') as f:
            json.dump([asdict(s) for s in sections], f, indent=2)

        # Save symbols
        with open(metadata_dir / 'symbols.json', 'w') as f:
            json.dump([asdict(s) for s in symbols], f, indent=2)

        self.logger.info("Metadata saved")

    def _save_config(self):
        """Save config to file"""
        config_path = self.mcp_data_path / 'config.json'
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=2)

    def initialize(self):
        """Main initialization process"""
        start_time = time.time()
        self.logger.info("Starting MCP initialization...")

        # Create directory structure and save config
        for subdir in ['index', 'metadata', 'cache', 'vectors', 'logs']:
            (self.mcp_data_path / subdir).mkdir(parents=True, exist_ok=True)

        # Now save the config file
        self._save_config()

        # Phase 1: File scanning
        self.logger.info("Phase 1: Scanning files...")
        files_metadata = self.scan_files()

        # Phase 2: Content parsing
        self.logger.info("Phase 2: Parsing content...")
        all_sections = []
        all_symbols = []

        # Process in parallel for speed
        with ThreadPoolExecutor(max_workers=4) as executor:
            doc_futures = []
            code_futures = []

            for file_meta in files_metadata:
                file_path = self.root_path / file_meta.path

                if file_meta.file_type == 'documents':
                    future = executor.submit(self.parse_markdown_document, file_path)
                    doc_futures.append(future)
                elif file_meta.file_type == 'code':
                    future = executor.submit(self.extract_code_symbols, file_path)
                    code_futures.append(future)

            # Collect results
            for future in as_completed(doc_futures):
                sections = future.result()
                all_sections.extend(sections)

            for future in as_completed(code_futures):
                symbols = future.result()
                all_symbols.extend(symbols)

        # Phase 3: Index creation
        self.logger.info("Phase 3: Creating search indices...")
        self.create_search_indices(files_metadata, all_sections, all_symbols)

        # Phase 4: Vector embeddings
        self.logger.info("Phase 4: Generating embeddings...")
        self.generate_embeddings(all_sections, all_symbols)

        # Phase 5: Save metadata
        self.logger.info("Phase 5: Saving metadata...")
        self.save_metadata(files_metadata, all_sections, all_symbols)

        # Final statistics
        duration = time.time() - start_time
        self.logger.info(f"Initialization completed in {duration:.2f} seconds")
        self.logger.info(f"Statistics: {self.stats}")

        print(f"\n✅ MCP initialization completed!")
        print(f"📊 Processed {self.stats['files_processed']} files")
        print(f"📄 Parsed {self.stats['documents_parsed']} documents with {self.stats['sections_created']} sections")
        print(f"🔍 Extracted {self.stats['symbols_extracted']} code symbols")
        print(f"🚀 Ready for fast MCP queries!")

def main():
    import argparse

    parser = argparse.ArgumentParser(description='Initialize MCP data for fast context retrieval')
    parser.add_argument('path', nargs='?', default='.', help='Path to initialize (default: current directory)')
    parser.add_argument('--force', action='store_true', help='Force re-initialization')

    args = parser.parse_args()

    # Check if .mcpdata exists BEFORE creating MCPInitializer
    mcp_data_path = Path(args.path) / '.mcpdata'

    if mcp_data_path.exists() and not args.force:
        print(f"⚠️  .mcpdata already exists in {args.path}")
        print("Use --force to re-initialize")
        return

    # Now create initializer and run
    initializer = MCPInitializer(args.path)
    initializer.initialize()

if __name__ == "__main__":
    main()
