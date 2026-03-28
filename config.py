"""
Configuration management module.
v5: Premium API loaded from separate file (won't be overwritten).
"""

import json
import threading
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict, field


@dataclass
class ServerConfig:
    host: str = "0.0.0.0"
    port: int = 8080
    api_key: str = "sk-free"
    workers: int = 1
    log_level: str = "INFO"


@dataclass
class PremiumAPIConfig:
    """Premium API configuration (loaded from separate file)."""
    enabled: bool = False
    providers: List[Dict[str, Any]] = field(default_factory=list)


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
    enabled: bool = True
    max_parse_retries: int = 2
    fallback_to_text: bool = True


@dataclass
class TestingConfig:
    enabled: bool = True
    test_on_startup: bool = True
    retest_interval_hours: int = 24
    test_timeout_seconds: int = 30
    sequential_delay_seconds: int = 2
    min_fc_score: int = 40


@dataclass
class RoutingConfig:
    g4f_max_fallbacks: int = 5
    only_use_tested: bool = True
    prefer_fc_score_above: int = 50


@dataclass
class TokenManagementConfig:
    g4f_sliding_window: int = 30


@dataclass
class CircuitBreakerConfig:
    failure_threshold: int = 3
    cooldown_seconds: int = 600
    rate_limit_per_minute: int = 10
    half_open_max_calls: int = 1


@dataclass
class StorageConfig:
    database_file: str = "bridge_data.db"
    max_database_size_mb: int = 50


@dataclass
class UpdaterConfig:
    auto_update_enabled: bool = False
    check_interval_hours: int = 24
    auto_rescan_after_update: bool = True


@dataclass
class Config:
    server: ServerConfig = field(default_factory=ServerConfig)
    premium_api: PremiumAPIConfig = field(default_factory=PremiumAPIConfig)
    g4f: G4FConfig = field(default_factory=G4FConfig)
    function_calling: FunctionCallingConfig = field(default_factory=FunctionCallingConfig)
    testing: TestingConfig = field(default_factory=TestingConfig)
    routing: RoutingConfig = field(default_factory=RoutingConfig)
    token_management: TokenManagementConfig = field(default_factory=TokenManagementConfig)
    circuit_breaker: CircuitBreakerConfig = field(default_factory=CircuitBreakerConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    updater: UpdaterConfig = field(default_factory=UpdaterConfig)

    def to_dict(self) -> Dict[str, Any]:
        """to_dict WITHOUT premium_api (saved separately)."""
        d = {}
        for section_name in (
            "server", "g4f", "function_calling",
            "testing", "routing", "token_management",
            "circuit_breaker", "storage", "updater",
        ):
            d[section_name] = asdict(getattr(self, section_name))
        return d

    def to_safe_dict(self) -> Dict[str, Any]:
        """Include premium (masked) for display."""
        d = self.to_dict()

        k = d.get("server", {}).get("api_key", "")
        if len(k) > 8:
            d["server"]["api_key"] = k[:4] + "..." + k[-4:]

        d["premium_api"] = {
            "enabled": self.premium_api.enabled,
            "provider_count": len(self.premium_api.providers),
            "providers": [
                {
                    "name": p.get("name", ""),
                    "enabled": p.get("enabled", False),
                    "models": p.get("models", []),
                }
                for p in self.premium_api.providers
            ],
        }

        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Config":
        cfg = cls()

        simple_sections = [
            ("server", ServerConfig),
            ("g4f", G4FConfig),
            ("function_calling", FunctionCallingConfig),
            ("testing", TestingConfig),
            ("routing", RoutingConfig),
            ("token_management", TokenManagementConfig),
            ("circuit_breaker", CircuitBreakerConfig),
            ("storage", StorageConfig),
            ("updater", UpdaterConfig),
        ]

        for section_name, section_cls in simple_sections:
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
        self.premium_path = Path(config_path).parent / "premium_config.json"
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

            self._load_premium()

        return self.config

    def _load_premium(self) -> None:
        """Load premium config from separate file (never overwritten)."""
        if self.premium_path.exists():
            try:
                with open(self.premium_path, "r", encoding="utf-8") as f:
                    premium_data = json.load(f)

                self.config.premium_api = PremiumAPIConfig(
                    enabled=premium_data.get("enabled", False),
                    providers=premium_data.get("providers", []),
                )

                if self.config.premium_api.enabled:
                    enabled_providers = [
                        p.get("name", "?")
                        for p in self.config.premium_api.providers
                        if p.get("enabled", False)
                    ]
                    if enabled_providers:
                        print(f"✓ Premium API: {', '.join(enabled_providers)}")

            except Exception as e:
                print(f"⚠ Failed to load premium config: {e}")
                self.config.premium_api = PremiumAPIConfig()
        else:
            self.config.premium_api = PremiumAPIConfig()

    def save(self) -> None:
        with self._lock:
            self._save_internal()

    def _save_internal(self) -> None:
        """Save ONLY main config (NOT premium)."""
        try:
            self.config_path.parent.mkdir(parents=True, exist_ok=True)

            data = {}
            for section_name in (
                "server", "g4f", "function_calling",
                "testing", "routing", "token_management",
                "circuit_breaker", "storage", "updater",
            ):
                data[section_name] = asdict(getattr(self.config, section_name))

            with open(self.config_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)

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
    print(json.dumps(config.to_safe_dict(), indent=2))