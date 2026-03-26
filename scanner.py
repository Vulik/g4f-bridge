"""
Provider and model scanner for g4f.
"""

import time
import inspect
import threading
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from logger_setup import get_logger


@dataclass
class ProviderInfo:
    """Information about a g4f provider."""
    name: str
    class_ref: Any
    models: List[str] = field(default_factory=list)
    supports_stream: bool = False
    needs_auth: bool = False
    working: bool = False
    priority: int = 99
    status: str = "untested"
    success_count: int = 0
    fail_count: int = 0
    avg_response_time: float = 0.0
    last_used: Optional[datetime] = None
    last_error: Optional[str] = None
    cooldown_until: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "models": self.models,
            "supports_stream": self.supports_stream,
            "needs_auth": self.needs_auth,
            "working": self.working,
            "priority": self.priority,
            "status": self.status,
            "success_count": self.success_count,
            "fail_count": self.fail_count,
            "avg_response_time": round(self.avg_response_time, 3),
            "last_used": self.last_used.isoformat() if self.last_used else None,
            "last_error": self.last_error,
            "cooldown_until": self.cooldown_until.isoformat() if self.cooldown_until else None,
        }


class ProviderScanner:
    """Scans and manages g4f providers."""

    MODEL_ALIASES: Dict[str, List[str]] = {
        "gpt-4": ["gpt-4", "gpt-4-turbo", "gpt-4-0613", "gpt-4-32k"],
        "gpt-4o": ["gpt-4o", "gpt-4o-2024-05-13", "gpt-4o-mini"],
        "gpt-3.5-turbo": ["gpt-3.5-turbo", "gpt-3.5-turbo-0125", "gpt-3.5-turbo-16k"],
        "claude": ["claude-3-sonnet", "claude-3.5-sonnet", "claude-3-opus", "claude-3-haiku"],
        "gemini": ["gemini-pro", "gemini-1.5-pro", "gemini-1.5-flash"],
        "llama": ["llama-2-70b", "llama-3-70b", "llama-3.1-70b"],
        "auto": [],
    }

    # Skip these base/internal classes during scanning
    _SKIP_CLASSES = frozenset({
        "BaseProvider", "AsyncProvider", "RetryProvider",
        "BaseRetryProvider", "AsyncGeneratorProvider", "ProviderModelMixin",
        "AbstractProvider", "CreateImagesProvider",
    })

    def __init__(self) -> None:
        self.logger = get_logger("scanner")
        self.providers: Dict[str, ProviderInfo] = {}
        self.model_to_providers: Dict[str, List[str]] = {}
        self.last_scan: Optional[datetime] = None
        self.g4f_version: str = ""
        self._lock = threading.Lock()

    def scan(self) -> None:
        """Scan all available g4f providers."""
        self.logger.info("Starting provider scan...")
        start_time = time.time()

        try:
            import g4f  # type: ignore
            self.g4f_version = getattr(g4f, "__version__", "unknown")
            self.logger.info(f"g4f version: {self.g4f_version}")
        except ImportError:
            self.logger.error("g4f is not installed!")
            return

        discovered = self._discover_providers()

        if not discovered:
            self.logger.error("No providers discovered!")
            return

        self.logger.info(f"Discovered {len(discovered)} provider classes")

        new_providers: Dict[str, ProviderInfo] = {}
        for provider_class in discovered:
            try:
                info = self._analyze_provider(provider_class)
                if info is not None:
                    new_providers[info.name] = info
            except Exception as e:
                self.logger.debug(f"Failed to analyze {getattr(provider_class, '__name__', '?')}: {e}")

        with self._lock:
            # Merge with existing stats
            for name, info in new_providers.items():
                if name in self.providers:
                    old = self.providers[name]
                    info.success_count = old.success_count
                    info.fail_count = old.fail_count
                    info.avg_response_time = old.avg_response_time
                    info.status = old.status
                    info.last_used = old.last_used

            self.providers = new_providers
            self._build_model_map()
            self._load_stats_from_db()
            self._update_priorities()
            self.last_scan = datetime.now()

        elapsed = time.time() - start_time
        self.logger.info(
            f"Scan complete: {len(self.providers)} providers, "
            f"{len(self.model_to_providers)} models ({elapsed:.2f}s)"
        )

    def _discover_providers(self) -> List[Any]:
        """Discover providers using multiple strategies."""
        providers = []

        # Strategy 1: g4f.Provider
        try:
            from g4f import Provider  # type: ignore
            for name in dir(Provider):
                if name.startswith("_") or name in self._SKIP_CLASSES:
                    continue
                attr = getattr(Provider, name, None)
                if attr is not None and inspect.isclass(attr):
                    providers.append(attr)
            if providers:
                self.logger.debug(f"Found {len(providers)} via g4f.Provider")
                return providers
        except Exception as e:
            self.logger.debug(f"Strategy 1: {e}")

        # Strategy 2: g4f.providers
        try:
            import g4f.providers as pm  # type: ignore
            for name in dir(pm):
                if name.startswith("_") or name in self._SKIP_CLASSES:
                    continue
                attr = getattr(pm, name, None)
                if attr is not None and inspect.isclass(attr):
                    providers.append(attr)
            if providers:
                self.logger.debug(f"Found {len(providers)} via g4f.providers")
                return providers
        except Exception as e:
            self.logger.debug(f"Strategy 2: {e}")

        return providers

    def _analyze_provider(self, provider_class: Any) -> Optional[ProviderInfo]:
        """Analyze a single provider. Returns None to skip."""
        name = getattr(provider_class, "__name__", "")
        if not name or name in self._SKIP_CLASSES:
            return None

        working = getattr(provider_class, "working", False)
        needs_auth = getattr(provider_class, "needs_auth", False)

        # Skip non-working and auth-required providers
        if not working or needs_auth:
            return None

        # Collect models
        models = self._extract_models(provider_class)
        if not models:
            return None

        info = ProviderInfo(
            name=name,
            class_ref=provider_class,
            models=models,
            supports_stream=bool(getattr(provider_class, "supports_stream", False)),
            needs_auth=needs_auth,
            working=working,
        )

        self.logger.debug(f"  ✓ {name} → {len(models)} models (stream={info.supports_stream})")
        return info

    @staticmethod
    def _extract_models(provider_class: Any) -> List[str]:
        """Extract supported models from provider class."""
        # Try multiple attribute patterns
        for attr_name in ("models", "model", "default_model", "supported_models"):
            val = getattr(provider_class, attr_name, None)
            if val is None:
                continue
            if isinstance(val, (list, tuple)):
                return [str(m) for m in val if m]
            if isinstance(val, set):
                return sorted(str(m) for m in val if m)
            if isinstance(val, str) and val:
                return [val]
        return []

    def _build_model_map(self) -> None:
        self.model_to_providers.clear()
        for pname, pinfo in self.providers.items():
            for model in pinfo.models:
                self.model_to_providers.setdefault(model, []).append(pname)

    def _load_stats_from_db(self) -> None:
        try:
            from storage import get_storage
            for stat in get_storage().get_all_provider_stats():
                name = stat["provider"]
                if name in self.providers:
                    p = self.providers[name]
                    p.success_count = max(p.success_count, stat["success_count"])
                    p.fail_count = max(p.fail_count, stat["fail_count"])
                    if stat["avg_response_time"] > 0:
                        p.avg_response_time = stat["avg_response_time"]
                    if stat["status"] != "untested":
                        p.status = stat["status"]
        except Exception as e:
            self.logger.debug(f"Could not load stats: {e}")

    def _update_priorities(self) -> None:
        from config import get_config
        preferred = get_config().g4f.preferred_models

        for info in self.providers.values():
            priority = 50

            # Bonus for preferred models
            for i, pref in enumerate(preferred):
                if any(self.model_matches(m, pref) for m in info.models):
                    priority -= (10 - min(i, 9))
                    break

            # Performance adjustments
            total = info.success_count + info.fail_count
            if total > 5:
                rate = info.success_count / total
                priority -= int(rate * 20)

            if info.avg_response_time > 10:
                priority += 15
            elif info.avg_response_time > 5:
                priority += 5

            if info.status == "degraded":
                priority += 20
            elif info.status == "dead":
                priority += 100

            info.priority = max(1, priority)

    # ------------------------------------------------------------------
    # Public query methods
    # ------------------------------------------------------------------

    def model_matches(self, model: str, pattern: str) -> bool:
        """Check if model matches pattern (with aliases). PUBLIC."""
        ml, pl = model.lower(), pattern.lower()
        if ml == pl:
            return True
        for alias_list in self.MODEL_ALIASES.values():
            lower_aliases = [a.lower() for a in alias_list]
            if pl in lower_aliases and ml in lower_aliases:
                return True
        return False

    def resolve_model_aliases(self, model: str) -> List[str]:
        """Resolve model to possible names."""
        ml = model.lower()
        if ml == "auto":
            from config import get_config
            return get_config().g4f.preferred_models

        for alias_list in self.MODEL_ALIASES.values():
            if ml in [a.lower() for a in alias_list]:
                return alias_list
        return [model]

    def get_providers_for_model(
        self, model: str, streaming: bool = False
    ) -> List[ProviderInfo]:
        """Get providers sorted by priority."""
        resolved = self.resolve_model_aliases(model)
        now = datetime.now()
        result = []

        with self._lock:
            for pinfo in self.providers.values():
                if pinfo.needs_auth:
                    continue
                if pinfo.cooldown_until and now < pinfo.cooldown_until:
                    continue
                if streaming and not pinfo.supports_stream:
                    continue

                for rmodel in resolved:
                    if any(self.model_matches(m, rmodel) for m in pinfo.models):
                        result.append(pinfo)
                        break

        result.sort(key=lambda p: p.priority)
        return result

    def get_all_models(self) -> List[str]:
        with self._lock:
            models = set()
            for pinfo in self.providers.values():
                models.update(pinfo.models)
        return sorted(models)

    def update_provider_status(
        self,
        provider_name: str,
        success: bool,
        response_time: float = 0.0,
        error: Optional[str] = None,
    ) -> None:
        with self._lock:
            if provider_name not in self.providers:
                return
            info = self.providers[provider_name]

        info.last_used = datetime.now()

        if success:
            info.success_count += 1
            info.last_error = None
            if info.avg_response_time == 0:
                info.avg_response_time = response_time
            else:
                info.avg_response_time = info.avg_response_time * 0.8 + response_time * 0.2
            if info.status in ("degraded", "untested"):
                info.status = "active"
        else:
            info.fail_count += 1
            info.last_error = error
            total = info.success_count + info.fail_count
            if total > 0:
                fail_rate = info.fail_count / total
                if fail_rate > 0.8:
                    info.status = "dead"
                    info.cooldown_until = datetime.now() + timedelta(hours=1)
                elif fail_rate > 0.5:
                    info.status = "degraded"
                    info.cooldown_until = datetime.now() + timedelta(minutes=10)

        # Persist
        try:
            from storage import get_storage
            for model in info.models:
                get_storage().update_provider_stats(
                    provider_name, model, success, response_time, error
                )
        except Exception:
            pass

    def get_status(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "last_scan": self.last_scan.isoformat() if self.last_scan else None,
                "g4f_version": self.g4f_version,
                "total_providers": len(self.providers),
                "total_models": len(self.model_to_providers),
                "providers": {
                    n: p.to_dict() for n, p in list(self.providers.items())[:15]
                },
            }


# Global
_scanner_lock = threading.Lock()
_scanner: Optional[ProviderScanner] = None


def get_scanner() -> ProviderScanner:
    global _scanner
    if _scanner is None:
        with _scanner_lock:
            if _scanner is None:
                _scanner = ProviderScanner()
    return _scanner


def rescan() -> None:
    get_scanner().scan()