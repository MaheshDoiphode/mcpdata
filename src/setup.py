from setuptools import setup, find_packages

setup(
    name="mcpdata",
    version="0.1",
    description="Fast, efficient indexing and searching for documentation and code repositories",
    author="MCP Team",
    packages=find_packages(where="."),
    package_dir={"": "."},
    entry_points={
        "console_scripts": [
            "mcpdata = mcpdata:main"
        ]
    },
    python_requires=">=3.8",
    install_requires=[
        # Core functionality uses only standard library
        # No external dependencies required for basic operation
    ],
    extras_require={
        "embeddings": [
            "numpy>=1.21.0",
            "sentence-transformers>=2.2.0",
        ],
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
        ],
    },
)
