"""
config.py
~~~~~~~~~
Konfigurasi terpusat untuk seluruh komponen bridge.
Semua konstanta bisa di-override via environment variable.
"""

from __future__ import annotations

import os
from pathlib import Path


def _env_int(key: str, default: int) -> int:
    return int(os.getenv(key, default))


def _env_float(key: str, default: float) -> float:
    return float(os.getenv(key, default))


# ═══════════════════════════════════════════════════
#  SCANNER — Inisialisasi & Pengujian Provider
# ═══════════════════════════════════════════════════

SCAN_TIMEOUT: int = _env_int("SCAN_TIMEOUT", 20)
"""Detik maksimum per pengujian satu provider."""

SCAN_CONCURRENCY: int = _env_int("SCAN_CONCURRENCY", 8)
"""Jumlah pengujian provider yang berjalan bersamaan."""

SCAN_TEST_MESSAGES: list[dict] = [
    {"role": "user", "content": "Reply with exactly one word: ok"}
]
"""Pesan tes ringan yang dikirim ke setiap provider."""

# ═══════════════════════════════════════════════════
#  CACHE — Penyimpanan Hasil Scan
# ═══════════════════════════════════════════════════

CACHE_FILE: Path = Path(
    os.getenv("CACHE_FILE", ".g4f_bridge_cache.json")
)
CACHE_TTL: int = _env_int("CACHE_TTL", 3600)
"""Masa berlaku cache dalam detik (default: 1 jam)."""

# ═══════════════════════════════════════════════════
#  RUNTIME — Eksekusi Request
# ═══════════════════════════════════════════════════

REQUEST_TIMEOUT: int = _env_int("REQUEST_TIMEOUT", 60)
"""Detik maksimum per percobaan satu provider saat runtime."""

MAX_RETRIES: int = _env_int("MAX_RETRIES", 2)
"""Jumlah retry per provider sebelum pindah ke provider berikutnya."""

RETRY_BACKOFF: float = _env_float("RETRY_BACKOFF", 1.0)
"""Basis backoff eksponensial antar retry (detik)."""

COOLDOWN_BASE: int = _env_int("COOLDOWN_BASE", 30)
"""Basis cooldown untuk provider yang gagal berturut-turut (detik)."""

COOLDOWN_MAX: int = _env_int("COOLDOWN_MAX", 300)
"""Cooldown maksimum (detik)."""

# ═══════════════════════════════════════════════════
#  AUTO-SELECT — Preferensi Model
# ═══════════════════════════════════════════════════

AUTO_MODEL_PRIORITY: list[str] = [
    "gpt-4o-mini",
    "gpt-4o",
    "gpt-4",
    "gpt-3.5-turbo",
    "claude-3.5-sonnet",
    "claude-3-haiku",
    "llama-3.1-70b",
    "llama-3.1-8b",
    "mixtral-8x7b",
]
"""Urutan preferensi model untuk mode auto (index 0 = paling disukai)."""

# ═══════════════════════════════════════════════════
#  SERVER
# ═══════════════════════════════════════════════════

HOST: str = os.getenv("HOST", "0.0.0.0")
PORT: int = _env_int("PORT", 8820)