from setuptools import setup

setup(
  name="mcp",
  version="0.1",
  packages=["mcp"],
  entry_points={
    "console_scripts": [
      "mcp = mcp.__init__:main"
    ]
  },
)
