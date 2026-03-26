"""
Environment detection and configuration module.
Detects runtime environment (Termux, Linux, Windows) and available resources.
Compatible with Python 3.8+
"""

import os
import sys
import platform
from pathlib import Path
from typing import Dict, Any, Optional, Tuple


class EnvironmentInfo:
    """Container for environment information."""

    def __init__(self) -> None:
        self.is_termux: bool = False
        self.is_arm: bool = False
        self.platform: str = ""
        self.arch: str = ""
        self.ram_mb: int = 0
        self.cpu_cores: int = 1
        self.tiktoken_available: bool = False
        self.uvicorn_available: bool = False
        self.g4f_installed: bool = False
        self.g4f_version: str = ""
        self.lightweight_mode: bool = False
        self.python_version: str = ""
        self.storage_path: Path = Path.home() / "g4f-bridge"
        self.data_dir: Path = Path.home() / "g4f-bridge" / "data"
        self.config_dir: Path = Path.home() / "g4f-bridge" / "config"

        # Derived settings
        self.server_backend: str = "uvicorn"
        self.token_counter: str = "tiktoken"
        self.max_cache_entries: int = 1000
        self.scan_interval_minutes: int = 30
        self.max_concurrent: int = 10

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "is_termux": self.is_termux,
            "is_arm": self.is_arm,
            "platform": self.platform,
            "arch": self.arch,
            "ram_mb": self.ram_mb,
            "cpu_cores": self.cpu_cores,
            "tiktoken_available": self.tiktoken_available,
            "uvicorn_available": self.uvicorn_available,
            "g4f_installed": self.g4f_installed,
            "g4f_version": self.g4f_version,
            "lightweight_mode": self.lightweight_mode,
            "python_version": self.python_version,
            "storage_path": str(self.storage_path),
            "data_dir": str(self.data_dir),
            "config_dir": str(self.config_dir),
            "server_backend": self.server_backend,
            "token_counter": self.token_counter,
            "max_concurrent": self.max_concurrent,
        }


class EnvironmentDetector:
    """Detects and analyzes the runtime environment."""

    @staticmethod
    def detect() -> EnvironmentInfo:
        """Detect all environment information."""
        info = EnvironmentInfo()

        # Platform & architecture
        info.is_termux = EnvironmentDetector._is_termux()
        info.platform = platform.system()
        info.arch = platform.machine()
        info.is_arm = info.arch.lower() in (
            "aarch64", "armv7l", "armv8", "arm64",
        )

        # Resources
        info.ram_mb = EnvironmentDetector._get_ram_mb()
        info.cpu_cores = EnvironmentDetector._get_cpu_cores()

        # Dependencies
        info.tiktoken_available = EnvironmentDetector._check_import("tiktoken")
        info.uvicorn_available = EnvironmentDetector._check_import("uvicorn")
        info.g4f_installed, info.g4f_version = EnvironmentDetector._check_g4f()

        # Python version
        info.python_version = platform.python_version()

        # Lightweight mode
        info.lightweight_mode = info.ram_mb < 1024 or info.is_arm

        # Derived settings
        info.server_backend = "uvicorn" if info.uvicorn_available else "flask"
        info.token_counter = "tiktoken" if info.tiktoken_available else "fallback"

        if info.lightweight_mode:
            info.max_cache_entries = 100
            info.scan_interval_minutes = 60
            info.max_concurrent = 2
        else:
            info.max_cache_entries = 1000
            info.scan_interval_minutes = 30
            info.max_concurrent = 10

        # Storage paths
        if info.is_termux:
            home = Path(os.environ.get(
                "HOME", "/data/data/com.termux/files/home"
            ))
            info.storage_path = home / "g4f-bridge"
        else:
            info.storage_path = Path.home() / "g4f-bridge"

        info.data_dir = info.storage_path / "data"
        info.config_dir = info.storage_path / "config"

        # Create directories
        info.data_dir.mkdir(parents=True, exist_ok=True)
        info.config_dir.mkdir(parents=True, exist_ok=True)

        return info

    @staticmethod
    def _is_termux() -> bool:
        """Check if running in Termux."""
        prefix = os.environ.get("PREFIX", "")
        if prefix.startswith("/data/data/com.termux"):
            return True
        if os.path.exists("/data/data/com.termux"):
            return True
        return False

    @staticmethod
    def _get_ram_mb() -> int:
        """Get total RAM in MB."""
        try:
            import psutil  # type: ignore
            return int(psutil.virtual_memory().total / (1024 * 1024))
        except ImportError:
            pass

        try:
            if platform.system() == "Linux":
                with open("/proc/meminfo", "r") as f:
                    for line in f:
                        if line.startswith("MemTotal:"):
                            return int(line.split()[1]) // 1024
        except Exception:
            pass

        return 2048  # Default

    @staticmethod
    def _get_cpu_cores() -> int:
        """Get number of CPU cores."""
        return os.cpu_count() or 1

    @staticmethod
    def _check_import(module_name: str) -> bool:
        """Check if a module can be imported."""
        try:
            __import__(module_name)
            return True
        except ImportError:
            return False

    @staticmethod
    def _check_g4f() -> Tuple[bool, str]:
        """Check if g4f is installed and get version."""
        try:
            import g4f  # type: ignore
            version = getattr(g4f, "__version__", "unknown")
            return True, str(version)
        except ImportError:
            return False, ""

    @staticmethod
    def print_info(info: EnvironmentInfo) -> None:
        """Print environment information."""
        print("=" * 60)
        print("ENVIRONMENT DETECTION REPORT")
        print("=" * 60)
        print(f"Platform:           {info.platform} ({info.arch})")
        print(f"Termux:             {'Yes' if info.is_termux else 'No'}")
        print(f"ARM Architecture:   {'Yes' if info.is_arm else 'No'}")
        print(f"RAM:                {info.ram_mb} MB")
        print(f"CPU Cores:          {info.cpu_cores}")
        print(f"Python Version:     {info.python_version}")
        print(f"Lightweight Mode:   {'Enabled' if info.lightweight_mode else 'Disabled'}")
        print("-" * 60)
        print("DEPENDENCIES:")
        g4f_str = ("✓ " + info.g4f_version) if info.g4f_installed else "✗ Not installed"
        print(f"  g4f:              {g4f_str}")
        print(f"  tiktoken:         {'✓' if info.tiktoken_available else '✗ (fallback mode)'}")
        print(f"  uvicorn:          {'✓' if info.uvicorn_available else '✗ (will use flask)'}")
        print("-" * 60)
        print("DERIVED SETTINGS:")
        print(f"  Server backend:   {info.server_backend}")
        print(f"  Token counter:    {info.token_counter}")
        print(f"  Max concurrent:   {info.max_concurrent}")
        print("=" * 60)


# ---------------------------------------------------------------------------
# Global singleton with thread-safety
# ---------------------------------------------------------------------------
import threading

_env_lock = threading.Lock()
_env_info: Optional[EnvironmentInfo] = None


def get_environment() -> EnvironmentInfo:
    """Get or create global environment info (thread-safe)."""
    global _env_info
    if _env_info is None:
        with _env_lock:
            if _env_info is None:
                _env_info = EnvironmentDetector.detect()
    return _env_info


def reload_environment() -> EnvironmentInfo:
    """Force reload environment detection."""
    global _env_info
    with _env_lock:
        _env_info = EnvironmentDetector.detect()
    return _env_info


if __name__ == "__main__":
    env = get_environment()
    EnvironmentDetector.print_info(env)