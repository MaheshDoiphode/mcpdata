from setuptools import setup, find_packages

setup(
    name="mcp-global-server",
    version="1.0.0",
    description="Global MCP Server for Documentation Search across workspaces",
    author="MCP Team",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "mcp>=1.0.0",
        "fastmcp>=0.1.0",
        "httpx>=0.25.0",
        "python-dotenv>=1.0.0",
        "pathlib",
    ],
    entry_points={
        "console_scripts": [
            "mcp-global-server=server:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
)
