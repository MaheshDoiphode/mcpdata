from setuptools import setup, find_packages

setup(
    name="mcpdata",
    version="0.1",
    packages=find_packages(),
    entry_points={
        "console_scripts": [
            "mcpdata = mcpdata.__init__:main"
        ]
    },
)
