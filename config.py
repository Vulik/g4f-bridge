"""
Configuration management module.
Updated with Function Calling and Provider Locking support.
"""

import json
import secrets
import threading
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict, field


@dataclass
class ServerConfig:
    host: str = "0.0.0.0"
    port: int = 8080
    api_key: str = ""
    workers: int = 1
    log_level: str = "INFO"

    def __post_init__(self) -> None:
        if not self.api_key:
            self.api_key = "sk-" + secrets.token_urlsafe(32)


@dataclass
class G4FConfig:
    default_model: str = "auto"
    preferred_models: List[str] = field(default_factory=lambda: [
        "gpt-4o", "gpt-4", "claude-3.5-sonnet", "gpt-3.5-turbo",
    ])
    timeout_seconds: int = 60
    max_retries_per_provider: int = 2
    max_provider_fallbacks: int = 5


@dataclass
class FunctionCallingConfig:
    """Function Calling Emulator configuration."""
    enabled: bool = True
    max_parse_retries: int = 2        # Retry with stronger prompt if JSON parse fails
    fallback_to_text: bool = True     # If all parsing fails, return text response
    log_injected_prompt: bool = False  # Debug: log the injected system prompt


@dataclass
class ProviderLockConfig:
    """Provider Locking configuration."""
    strict_provider_mode: bool = False  # Lock to single provider
    locked_provider: str = ""           # Provider name (e.g., "Blackbox")
    locked_model: str = ""              # Model name (e.g., "deepseek-r1")
    fail_on_lock_error: bool = True     # Error instead of fallback


@dataclass
class MCPConfig:
    """MCP (Model Context Protocol) offloading configuration."""
    enabled: bool = False
    g4f_mcp_url: str = "http://localhost:8765"  # g4f MCP server URL
    # NOTE: MCP offloading means PicoClaw connects to g4f MCP server
    # DIRECTLY for tool execution, while using this bridge only for
    # LLM reasoning. See README for setup instructions.


@dataclass
class TokenManagerConfig:
    max_tokens_per_session: int = 45000
    enable_auto_continuation: bool = True
    max_continuations: int = 5
    enable_context_summarization: bool = True
    sliding_window_messages: int = 20


@dataclass
class ScannerConfig:
    scan_interval_minutes: int = 30
    lazy_health_check: bool = True
    initial_scan_on_startup: bool = True


@dataclass
class UpdaterConfig:
    auto_update_enabled: bool = False
    check_interval_hours: int = 6
    auto_rescan_after_update: bool = True


@dataclass
class CircuitBreakerConfig:
    failure_threshold: int = 3
    cooldown_seconds: int = 600
    rate_limit_per_minute: int = 5
    half_open_max_calls: int = 1


@dataclass
class StorageConfig:
    database_file: str = "bridge_data.db"
    max_database_size_mb: int = 50
    conversation_retention_hours: int = 168


@dataclass
class Config:
    server: ServerConfig = field(default_factory=ServerConfig)
    g4f: G4FConfig = field(default_factory=G4FConfig)
    function_calling: FunctionCallingConfig = field(default_factory=FunctionCallingConfig)
    provider_lock: ProviderLockConfig = field(default_factory=ProviderLockConfig)
    mcp: MCPConfig = field(default_factory=MCPConfig)
    token_manager: TokenManagerConfig = field(default_factory=TokenManagerConfig)
    scanner: ScannerConfig = field(default_factory=ScannerConfig)
    updater: UpdaterConfig = field(default_factory=UpdaterConfig)
    circuit_breaker: CircuitBreakerConfig = field(default_factory=CircuitBreakerConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "server": asdict(self.server),
            "g4f": asdict(self.g4f),
            "function_calling": asdict(self.function_calling),
            "provider_lock": asdict(self.provider_lock),
            "mcp": asdict(self.mcp),
            "token_manager": asdict(self.token_manager),
            "scanner": asdict(self.scanner),
            "updater": asdict(self.updater),
            "circuit_breaker": asdict(self.circuit_breaker),
            "storage": asdict(self.storage),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Config":
        """Create from dictionary with safe merging."""
        cfg = cls()

        section_map = [
            ("server", ServerConfig),
            ("g4f", G4FConfig),
            ("function_calling", FunctionCallingConfig),
            ("provider_lock", ProviderLockConfig),
            ("mcp", MCPConfig),
            ("token_manager", TokenManagerConfig),
            ("scanner", ScannerConfig),
            ("updater", UpdaterConfig),
            ("circuit_breaker", CircuitBreakerConfig),
            ("storage", StorageConfig),
        ]

        for section_name, section_cls in section_map:
            section_data = data.get(section_name, {})
            if isinstance(section_data, dict):
                valid_fields = {
                    f.name for f in section_cls.__dataclass_fields__.values()
                }
                filtered = {
                    k: v for k, v in section_data.items() if k in valid_fields
                }
                try:
                    setattr(cfg, section_name, section_cls(**filtered))
                except TypeError:
                    pass

        return cfg


class ConfigManager:
    def __init__(self, config_path: Optional[Path] = None) -> None:
        from environment import get_environment

        env = get_environment()
        if config_path is None:
            config_path = env.config_dir / "config.json"

        self.config_path = Path(config_path)
        self.config: Config = Config()
        self._lock = threading.Lock()

    def load(self) -> Config:
        with self._lock:
            if self.config_path.exists():
                try:
                    with open(self.config_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    self.config = Config.from_dict(data)
                    print(f"✓ Configuration loaded from {self.config_path}")
                except Exception as e:
                    print(f"⚠ Failed to load config: {e}, using defaults")
                    self.config = Config()
            else:
                print(f"ℹ Creating default config at {self.config_path}")
                self.config = Config()
                self._save_internal()

        return self.config

    def save(self) -> None:
        with self._lock:
            self._save_internal()

    def _save_internal(self) -> None:
        try:
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.config_path, "w", encoding="utf-8") as f:
                json.dump(self.config.to_dict(), f, indent=2)
        except Exception as e:
            print(f"✗ Failed to save config: {e}")

    def get(self) -> Config:
        return self.config


_config_lock = threading.Lock()
_config_manager: Optional[ConfigManager] = None


def get_config() -> Config:
    global _config_manager
    if _config_manager is None:
        with _config_lock:
            if _config_manager is None:
                _config_manager = ConfigManager()
                _config_manager.load()
    return _config_manager.config


def save_config() -> None:
    global _config_manager
    if _config_manager is not None:
        _config_manager.save()


def reload_config() -> Config:
    global _config_manager
    with _config_lock:
        _config_manager = ConfigManager()
        return _config_manager.load()


if __name__ == "__main__":
    config = get_config()
    print(json.dumps(config.to_dict(), indent=2))