"""
Configuration management for MCP indexing system
"""

import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict


@dataclass
class IndexingConfig:
    """Configuration for indexing behavior"""
    max_file_size: int = 5 * 1024 * 1024  # 5MB
    chunk_size: int = 512
    chunk_overlap: int = 50
    enable_embeddings: bool = True
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_batch_size: int = 32
    parallel_workers: int = 4


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


@dataclass
class LoggingConfig:
    """Configuration for logging"""
    level: str = "INFO"
    format: str = "%(asctime)s - %(levelname)s - %(message)s"
    file_path: Optional[str] = None
    console_output: bool = True


@dataclass
class MCPConfig:
    """Main configuration class"""
    version: str = "1.0.0"
    project_type: str = "auto"
    indexing: IndexingConfig = None
    search: SearchConfig = None
    logging: LoggingConfig = None
    ignored_patterns: List[str] = None
    file_types: Dict[str, List[str]] = None

    def __post_init__(self):
        if self.indexing is None:
            self.indexing = IndexingConfig()
        if self.search is None:
            self.search = SearchConfig()
        if self.logging is None:
            self.logging = LoggingConfig()
        if self.ignored_patterns is None:
            self.ignored_patterns = self._get_default_ignored_patterns()
        if self.file_types is None:
            self.file_types = self._get_default_file_types()

    @staticmethod
    def _get_default_ignored_patterns() -> List[str]:
        """Get default patterns to ignore during indexing"""
        return [
            # Build directories
            "build/", "dist/", "target/", "out/", "bin/", "obj/",
            # Dependencies
            "node_modules/", ".gradle/", ".m2/", "vendor/", "venv/", "env/",
            # Version control
            ".git/", ".svn/", ".hg/",
            # IDE files
            ".idea/", ".vscode/", "*.iml", "*.swp", "*.swo",
            # OS files
            ".DS_Store", "Thumbs.db",
            # Temp files
            "*.tmp", "*.log", "*.pyc", "__pycache__/",
            # Compiled files
            "*.class", "*.jar", "*.war", "*.exe", "*.dll", "*.so", "*.dylib",
            # Archives
            "*.zip", "*.tar", "*.gz", "*.7z", "*.rar",
            # Build scripts
            "gradlew", "gradlew.bat", "mvnw", "mvnw.cmd",
            # MCP data
            ".mcpdata"
        ]

    @staticmethod
    def _get_default_file_types() -> Dict[str, List[str]]:
        """Get default file type mappings"""
        return {
            "documents": [
                ".md", ".rst", ".txt", ".adoc", ".asciidoc", ".org",
                ".tex", ".wiki", ".confluence"
            ],
            "code": [
                # JVM languages
                ".java", ".kt", ".kts", ".scala", ".groovy", ".clj", ".cljs",
                # JavaScript/TypeScript
                ".js", ".ts", ".jsx", ".tsx", ".vue", ".svelte",
                # Web
                ".html", ".htm", ".css", ".scss", ".sass", ".less",
                # Systems programming
                ".c", ".cpp", ".cc", ".cxx", ".h", ".hpp", ".hxx",
                ".rust", ".rs", ".go", ".zig",
                # Scripting
                ".py", ".rb", ".pl", ".sh", ".bash", ".zsh", ".fish",
                # Functional
                ".hs", ".elm", ".ml", ".fs", ".ex", ".exs",
                # Other
                ".php", ".swift", ".dart", ".r", ".jl", ".nim",
                ".lua", ".ps1", ".psm1"
            ],
            "config": [
                ".json", ".yaml", ".yml", ".toml", ".ini", ".properties",
                ".conf", ".cfg", ".config", ".env", ".dotenv",
                ".gradle", ".xml", ".plist", ".dockerfile", "Dockerfile",
                ".makefile", "Makefile", ".cmake", "CMakeLists.txt"
            ],
            "data": [
                ".csv", ".tsv", ".json", ".jsonl", ".xml", ".parquet",
                ".avro", ".orc", ".arrow", ".feather", ".hdf5", ".h5"
            ],
            "notebooks": [
                ".ipynb", ".rmd", ".qmd", ".jmd"
            ]
        }


class ConfigManager:
    """Configuration manager for loading and saving configurations"""

    def __init__(self, config_path: Path):
        self.config_path = config_path
        self._config: Optional[MCPConfig] = None

    def load_config(self) -> MCPConfig:
        """Load configuration from file or create default"""
        if self.config_path.exists():
            try:
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    config_data = json.load(f)

                # Convert nested dicts back to dataclasses
                if 'indexing' in config_data:
                    config_data['indexing'] = IndexingConfig(**config_data['indexing'])
                if 'search' in config_data:
                    config_data['search'] = SearchConfig(**config_data['search'])
                if 'logging' in config_data:
                    config_data['logging'] = LoggingConfig(**config_data['logging'])

                self._config = MCPConfig(**config_data)
                return self._config
            except Exception as e:
                print(f"Warning: Could not load config from {self.config_path}: {e}")
                print("Using default configuration")

        # Create default configuration
        self._config = MCPConfig()
        return self._config

    def save_config(self, config: MCPConfig) -> None:
        """Save configuration to file"""
        try:
            # Convert dataclasses to dict for JSON serialization
            config_dict = asdict(config)

            # Ensure parent directory exists
            self.config_path.parent.mkdir(parents=True, exist_ok=True)

            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump(config_dict, f, indent=2, ensure_ascii=False)

            self._config = config
        except Exception as e:
            raise RuntimeError(f"Could not save config to {self.config_path}: {e}")

    def get_config(self) -> MCPConfig:
        """Get current configuration"""
        if self._config is None:
            return self.load_config()
        return self._config

    def update_config(self, **kwargs) -> None:
        """Update configuration with new values"""
        if self._config is None:
            self.load_config()

        # Update configuration
        for key, value in kwargs.items():
            if hasattr(self._config, key):
                setattr(self._config, key, value)

        # Save updated configuration
        self.save_config(self._config)

    def detect_project_type(self, root_path: Path) -> str:
        """Detect project type based on files present"""
        # Check for common project indicators
        indicators = {
            "python": ["setup.py", "pyproject.toml", "requirements.txt", "Pipfile"],
            "javascript": ["package.json", "yarn.lock", "package-lock.json"],
            "java": ["pom.xml", "build.gradle", "build.gradle.kts"],
            "rust": ["Cargo.toml", "Cargo.lock"],
            "go": ["go.mod", "go.sum"],
            "dotnet": ["*.csproj", "*.sln", "*.fsproj", "*.vbproj"],
            "ruby": ["Gemfile", "Gemfile.lock", "*.gemspec"],
            "php": ["composer.json", "composer.lock"],
            "documentation": ["README.md", "docs/", "doc/", "*.md"]
        }

        for project_type, files in indicators.items():
            for file_pattern in files:
                if file_pattern.startswith("*."):
                    # Handle glob patterns
                    extension = file_pattern[1:]
                    if any(f.suffix == extension for f in root_path.rglob("*")):
                        return project_type
                else:
                    # Handle exact file names
                    if (root_path / file_pattern).exists():
                        return project_type

        return "mixed"

    def get_language_specific_config(self, project_type: str) -> Dict[str, Any]:
        """Get language-specific configuration overrides"""
        language_configs = {
            "python": {
                "ignored_patterns": [
                    "*.pyc", "__pycache__/", ".pytest_cache/", ".coverage",
                    ".tox/", ".venv/", "venv/", "env/", ".mypy_cache/"
                ]
            },
            "javascript": {
                "ignored_patterns": [
                    "node_modules/", ".npm/", ".yarn/", "dist/", "build/",
                    "coverage/", ".nyc_output/", ".next/", ".nuxt/"
                ]
            },
            "java": {
                "ignored_patterns": [
                    "target/", "build/", ".gradle/", ".m2/", "*.class",
                    "*.jar", "*.war", "*.ear"
                ]
            },
            "rust": {
                "ignored_patterns": [
                    "target/", "Cargo.lock", "*.rlib", "*.rmeta"
                ]
            },
            "go": {
                "ignored_patterns": [
                    "vendor/", "go.sum", "*.exe", "*.test"
                ]
            },
            "documentation": {
                "indexing": {
                    "enable_embeddings": True,
                    "chunk_size": 256,  # Smaller chunks for docs
                    "chunk_overlap": 25
                }
            }
        }

        return language_configs.get(project_type, {})
