#!/usr/bin/env python3
"""
Setup script for Global MCP Server
Handles installation, configuration, and management of the global documentation server
"""

import os
import sys
import json
import shutil
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional

# Server configuration
SERVER_NAME = "mcp-global-docs"
SERVER_VERSION = "1.0.0"
DEFAULT_REGISTRY_PATH = Path.home() / "Documents" / "mcpdata"

# Default configuration
DEFAULT_CONFIG = {
    "server": {
        "name": SERVER_NAME,
        "version": SERVER_VERSION,
        "log_level": "INFO"
    },
    "registry": {
        "path": str(DEFAULT_REGISTRY_PATH),
        "max_workspaces": 100,
        "auto_cleanup_days": 30
    },
    "search": {
        "max_results": 50,
        "token_limit": 4000
    }
}


class GlobalServerSetup:
    """Setup and management for Global MCP Server"""

    def __init__(self, registry_path: Optional[Path] = None):
        self.registry_path = registry_path or DEFAULT_REGISTRY_PATH
        self.config_path = self.registry_path / "server_config.json"
        self.logs_path = self.registry_path / "logs"
        self.pid_file = self.registry_path / "server.pid"

    def create_directories(self) -> None:
        """Create necessary directories"""
        directories = [
            self.registry_path,
            self.registry_path / "workspaces",
            self.registry_path / "global",
            self.registry_path / "backups",
            self.logs_path
        ]

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            print(f"✓ Created directory: {directory}")

    def create_config(self, force: bool = False) -> None:
        """Create server configuration"""
        if self.config_path.exists() and not force:
            print(f"⚠️  Configuration already exists at {self.config_path}")
            print("Use --force to overwrite")
            return

        # Update paths in config
        config = DEFAULT_CONFIG.copy()
        config["registry"]["path"] = str(self.registry_path)

        # Write configuration
        with open(self.config_path, 'w') as f:
            json.dump(config, f, indent=2)

        print(f"✓ Created configuration: {self.config_path}")

    def start_server(self, background: bool = False) -> None:
        """Start the global MCP server"""
        if self.is_running():
            print("⚠️  Server is already running")
            return

        print("Starting Global MCP Server...")

        # Get the server module path
        server_module = Path(__file__).parent / "__init__.py"

        # Command to start server
        cmd = [
            sys.executable, str(server_module),
            "--registry-path", str(self.registry_path)
        ]

        if background:
            # Start in background
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                stdin=subprocess.PIPE
            )

            # Save PID
            with open(self.pid_file, 'w') as f:
                f.write(str(process.pid))

            print(f"✓ Server started in background (PID: {process.pid})")
        else:
            # Start in foreground
            try:
                subprocess.run(cmd, check=True)
            except KeyboardInterrupt:
                print("\n✓ Server stopped by user")
            except subprocess.CalledProcessError as e:
                print(f"❌ Server failed: {e}")

    def stop_server(self) -> None:
        """Stop the global MCP server"""
        if not self.is_running():
            print("⚠️  Server is not running")
            return

        try:
            with open(self.pid_file, 'r') as f:
                pid = int(f.read().strip())

            import signal
            os.kill(pid, signal.SIGTERM)

            # Remove PID file
            self.pid_file.unlink(missing_ok=True)

            print("✓ Server stopped")
        except Exception as e:
            print(f"⚠️  Could not stop server: {e}")

    def is_running(self) -> bool:
        """Check if server is running"""
        if not self.pid_file.exists():
            return False

        try:
            with open(self.pid_file, 'r') as f:
                pid = int(f.read().strip())

            # Check if process exists
            import signal
            os.kill(pid, 0)
            return True
        except (OSError, ValueError):
            # Process doesn't exist, clean up PID file
            self.pid_file.unlink(missing_ok=True)
            return False

    def status(self) -> None:
        """Show server status"""
        print(f"Registry Path: {self.registry_path}")
        print(f"Config Path: {self.config_path}")
        print(f"Logs Path: {self.logs_path}")

        if self.is_running():
            with open(self.pid_file, 'r') as f:
                pid = f.read().strip()
            print(f"Status: Running (PID: {pid})")
        else:
            print("Status: Stopped")

        # Show registry stats if available
        try:
            sys.path.insert(0, str(Path(__file__).parent.parent / "mcp"))
            from mcp.core.registry import CentralRegistry
            registry = CentralRegistry(self.registry_path)
            stats = registry.get_stats()
            print(f"Workspaces: {stats.active_workspaces}/{stats.total_workspaces}")
            print(f"Documents: {stats.total_documents}")
            print(f"Total Size: {stats.total_size_mb:.2f} MB")
        except Exception as e:
            print(f"⚠️  Could not load registry stats: {e}")

    def cleanup(self) -> None:
        """Clean up server data"""
        print("Cleaning up server data...")

        # Stop server if running
        if self.is_running():
            self.stop_server()

        # Clean up logs
        if self.logs_path.exists():
            for log_file in self.logs_path.glob("*.log*"):
                log_file.unlink()
            print("✓ Cleaned up log files")

        # Clean up old backups
        backup_path = self.registry_path / "backups"
        if backup_path.exists():
            for backup_file in backup_path.glob("*.json"):
                backup_file.unlink()
            print("✓ Cleaned up backups")

    def setup(self, force: bool = False) -> None:
        """Perform setup"""
        print(f"Setting up Global MCP Server...")
        print(f"Registry Path: {self.registry_path}")
        print()

        # Create directories
        self.create_directories()

        # Create configuration
        self.create_config(force=force)

        print()
        print("✅ Setup completed successfully!")
        print()
        print("Next steps:")
        print("1. Initialize workspaces with: mcp /path/to/workspace --workspace-name 'Name' --workspace-description 'Description'")
        print("2. Start the server with: python setup_global_server.py --start")


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description="Global MCP Server Setup")
    parser.add_argument(
        "--registry-path",
        type=str,
        help=f"Registry path (default: {DEFAULT_REGISTRY_PATH})"
    )
    parser.add_argument(
        "--setup",
        action="store_true",
        help="Perform setup"
    )
    parser.add_argument(
        "--start",
        action="store_true",
        help="Start the server"
    )
    parser.add_argument(
        "--stop",
        action="store_true",
        help="Stop the server"
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Show server status"
    )
    parser.add_argument(
        "--cleanup",
        action="store_true",
        help="Clean up server data"
    )
    parser.add_argument(
        "--background",
        action="store_true",
        help="Run server in background"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force overwrite existing files"
    )

    args = parser.parse_args()

    # Create setup instance
    registry_path = Path(args.registry_path) if args.registry_path else None
    setup = GlobalServerSetup(registry_path)

    # Handle commands
    if args.setup:
        setup.setup(force=args.force)
    elif args.start:
        setup.start_server(background=args.background)
    elif args.stop:
        setup.stop_server()
    elif args.status:
        setup.status()
    elif args.cleanup:
        setup.cleanup()
    else:
        # Show help by default
        parser.print_help()


if __name__ == "__main__":
    main()
