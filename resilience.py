"""
Resilience layer: circuit breaker, rate limiter, connectivity check.
Thread-safe implementation.
"""

import time
import socket
import asyncio
import threading
from enum import Enum
from typing import Dict, Optional, Any
from datetime import datetime, timedelta
from collections import deque
from logger_setup import get_logger


class CircuitState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker:
    """Thread-safe circuit breaker."""

    def __init__(
        self,
        failure_threshold: int = 3,
        cooldown_seconds: int = 600,
        half_open_max_calls: int = 1,
    ) -> None:
        self.failure_threshold = failure_threshold
        self.cooldown_seconds = cooldown_seconds
        self.half_open_max_calls = half_open_max_calls

        self.state = CircuitState.CLOSED
        self.consecutive_failures = 0
        self.total_failures = 0
        self.total_successes = 0
        self.opened_at: Optional[datetime] = None
        self.half_open_calls = 0
        self._lock = threading.Lock()

    def is_available(self) -> bool:
        """Check if circuit allows requests (without consuming)."""
        with self._lock:
            if self.state == CircuitState.CLOSED:
                return True

            if self.state == CircuitState.OPEN:
                if self._cooldown_elapsed():
                    self.state = CircuitState.HALF_OPEN
                    self.half_open_calls = 0
                    return True
                return False

            # HALF_OPEN
            return self.half_open_calls < self.half_open_max_calls

    def record_success(self) -> None:
        """Record successful call."""
        with self._lock:
            self.consecutive_failures = 0
            self.total_successes += 1
            if self.state == CircuitState.HALF_OPEN:
                self.state = CircuitState.CLOSED
                self.opened_at = None

    def record_failure(self) -> None:
        """Record failed call."""
        with self._lock:
            self.consecutive_failures += 1
            self.total_failures += 1

            if self.state == CircuitState.HALF_OPEN:
                self.state = CircuitState.OPEN
                self.opened_at = datetime.now()
            elif self.consecutive_failures >= self.failure_threshold:
                self.state = CircuitState.OPEN
                self.opened_at = datetime.now()

    def _cooldown_elapsed(self) -> bool:
        if self.opened_at is None:
            return True
        return (datetime.now() - self.opened_at).total_seconds() >= self.cooldown_seconds

    def reset(self) -> None:
        with self._lock:
            self.state = CircuitState.CLOSED
            self.consecutive_failures = 0
            self.opened_at = None
            self.half_open_calls = 0

    def get_status(self) -> Dict[str, Any]:
        return {
            "state": self.state.value,
            "consecutive_failures": self.consecutive_failures,
            "total_failures": self.total_failures,
            "total_successes": self.total_successes,
            "opened_at": self.opened_at.isoformat() if self.opened_at else None,
        }


class RateLimiter:
    """Thread-safe sliding window rate limiter."""

    def __init__(self, max_calls: int, window_seconds: int = 60) -> None:
        self.max_calls = max_calls
        self.window_seconds = window_seconds
        self.calls: deque = deque()
        self._lock = threading.Lock()

    def check(self) -> bool:
        """Check if request would be allowed (without consuming slot)."""
        with self._lock:
            self._prune()
            return len(self.calls) < self.max_calls

    def acquire(self) -> bool:
        """Consume a slot. Returns True if allowed."""
        with self._lock:
            self._prune()
            if len(self.calls) < self.max_calls:
                self.calls.append(time.time())
                return True
            return False

    def wait_time(self) -> float:
        """Seconds until next slot opens."""
        with self._lock:
            self._prune()
            if len(self.calls) < self.max_calls:
                return 0.0
            return max(0.0, self.calls[0] + self.window_seconds - time.time())

    def _prune(self) -> None:
        cutoff = time.time() - self.window_seconds
        while self.calls and self.calls[0] < cutoff:
            self.calls.popleft()

    def reset(self) -> None:
        with self._lock:
            self.calls.clear()


class ConnectivityChecker:
    """Internet connectivity checker."""

    HOSTS = [
        ("8.8.8.8", 53),
        ("1.1.1.1", 53),
        ("208.67.222.222", 53),
    ]

    def __init__(self) -> None:
        self.logger = get_logger("connectivity")
        self.is_online: bool = True
        self._last_check: Optional[datetime] = None
        self._check_interval = 30
        self._lock = threading.Lock()

    def check(self, force: bool = False) -> bool:
        with self._lock:
            now = datetime.now()
            if (
                not force
                and self._last_check
                and (now - self._last_check).total_seconds() < self._check_interval
            ):
                return self.is_online

            for host, port in self.HOSTS:
                try:
                    s = socket.create_connection((host, port), timeout=3)
                    s.close()
                    was_offline = not self.is_online
                    self.is_online = True
                    self._last_check = now
                    if was_offline:
                        self.logger.info("Internet connectivity restored")
                    return True
                except (socket.timeout, socket.error, OSError):
                    continue

            if self.is_online:
                self.logger.warning("Internet connectivity lost")
            self.is_online = False
            self._last_check = now
            return False

    async def check_async(self, force: bool = False) -> bool:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.check, force)


class ResilienceManager:
    """Manages all resilience components."""

    def __init__(self) -> None:
        from config import get_config

        config = get_config()
        cb_cfg = config.circuit_breaker

        self._circuit_breakers: Dict[str, CircuitBreaker] = {}
        self._rate_limiters: Dict[str, RateLimiter] = {}
        self.connectivity = ConnectivityChecker()
        self._lock = threading.Lock()

        self._failure_threshold = cb_cfg.failure_threshold
        self._cooldown_seconds = cb_cfg.cooldown_seconds
        self._rate_limit_per_minute = cb_cfg.rate_limit_per_minute

    def get_circuit_breaker(self, provider: str) -> CircuitBreaker:
        with self._lock:
            if provider not in self._circuit_breakers:
                self._circuit_breakers[provider] = CircuitBreaker(
                    failure_threshold=self._failure_threshold,
                    cooldown_seconds=self._cooldown_seconds,
                )
            return self._circuit_breakers[provider]

    def get_rate_limiter(self, provider: str) -> RateLimiter:
        with self._lock:
            if provider not in self._rate_limiters:
                self._rate_limiters[provider] = RateLimiter(
                    max_calls=self._rate_limit_per_minute,
                    window_seconds=60,
                )
            return self._rate_limiters[provider]

    def is_provider_available(self, provider: str) -> tuple:
        """Returns (is_available: bool, reason: Optional[str])."""
        if not self.connectivity.check():
            return False, "No internet connection"

        cb = self.get_circuit_breaker(provider)
        if not cb.is_available():
            return False, f"Circuit breaker {cb.state.value}"

        rl = self.get_rate_limiter(provider)
        if not rl.check():
            return False, f"Rate limited (wait {rl.wait_time():.1f}s)"

        return True, None

    def get_status(self) -> Dict[str, Any]:
        return {
            "online": self.connectivity.is_online,
            "circuit_breakers": {
                n: cb.get_status()
                for n, cb in self._circuit_breakers.items()
            },
        }


# Global
_resilience_lock = threading.Lock()
_resilience: Optional[ResilienceManager] = None


def get_resilience() -> ResilienceManager:
    global _resilience
    if _resilience is None:
        with _resilience_lock:
            if _resilience is None:
                _resilience = ResilienceManager()
    return _resilience