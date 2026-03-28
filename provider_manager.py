"""
provider_manager.py
~~~~~~~~~~~~~~~~~~~
Mengelola lifecycle provider g4f:
  1. Auto-discovery  — membaca g4f.models.ModelUtils untuk mapping model→provider
  2. Health tracking — mencatat sukses/gagal setiap provider
  3. Ranked fallback — menyortir provider berdasarkan skor kesehatan
  4. Cooldown        — provider yang gagal berturut-turut diistirahatkan sementara
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("g4f-bridge.providers")


# ───────────────────── HEALTH TRACKER ─────────────

@dataclass
class ProviderHealth:
    """Metrik kesehatan satu provider."""

    name: str
    provider_class: Any
    success_count: int = 0
    failure_count: int = 0
    last_success: float = 0.0
    last_failure: float = 0.0
    consecutive_failures: int = 0

    # -- properties --------------------------------------------------

    @property
    def is_healthy(self) -> bool:
        """Provider dianggap sehat jika belum pernah gagal, atau
        sudah melewati masa cooldown."""
        if self.consecutive_failures == 0:
            return True
        if self.last_success > self.last_failure:
            return True
        # Cooldown eksponensial: 30s, 60s, 90s ... max 5 menit
        cooldown = min(300, 30 * self.consecutive_failures)
        return (time.time() - self.last_failure) > cooldown

    @property
    def score(self) -> float:
        """Skor [0..1+] — semakin tinggi semakin diprioritaskan."""
        total = self.success_count + self.failure_count
        if total == 0:
            return 0.5                       # belum pernah dicoba
        rate = self.success_count / total
        bonus = 0.1 if self.last_success > self.last_failure else 0.0
        return rate + bonus

    # -- mutators ----------------------------------------------------

    def record_success(self):
        self.last_success = time.time()
        self.success_count += 1
        self.consecutive_failures = 0

    def record_failure(self):
        self.last_failure = time.time()
        self.failure_count += 1
        self.consecutive_failures += 1


# ───────────────────── PROVIDER MANAGER ───────────

class ProviderManager:
    """Registry sentral yang memetakan model → [provider] dan
    melacak kondisi masing-masing."""

    def __init__(self):
        self._health: Dict[str, ProviderHealth] = {}
        self._model_providers: Dict[str, List[Any]] = {}
        self._discover()

    # -- discovery ---------------------------------------------------

    def _discover(self):
        """Baca g4f internal registry untuk mendapat mapping
        model_name → [ProviderClass, ...]."""
        try:
            from g4f.models import ModelUtils
        except ImportError:
            logger.error("g4f.models.ModelUtils tidak ditemukan — "
                         "pastikan g4f terinstal dengan benar")
            return

        for model_name, model_obj in ModelUtils.convert.items():
            providers = self._extract(model_obj)
            if not providers:
                continue
            self._model_providers[model_name] = providers
            for p in providers:
                pn = self._pname(p)
                if pn not in self._health:
                    self._health[pn] = ProviderHealth(
                        name=pn, provider_class=p
                    )

        logger.info(
            f"Discovery selesai → {len(self._model_providers)} model, "
            f"{len(self._health)} provider unik"
        )

    @staticmethod
    def _extract(model_obj) -> List[Any]:
        """Mengekstrak provider konkret dari model_obj.best_provider
        (bisa berupa IterListProvider/RetryProvider atau provider tunggal)."""
        bp = getattr(model_obj, "best_provider", None)
        if bp is None:
            return []

        # IterListProvider / RetryProvider membungkus list
        if hasattr(bp, "providers"):
            return [p for p in bp.providers
                    if getattr(p, "working", True)]

        if getattr(bp, "working", True):
            return [bp]
        return []

    @staticmethod
    def _pname(provider) -> str:
        return getattr(provider, "__name__", type(provider).__name__)

    # -- query -------------------------------------------------------

    def get_ranked_providers(
        self, model: str
    ) -> List[Tuple[str, Any]]:
        """Mengembalikan [(name, cls), ...] diurutkan skor tertinggi,
        provider yang sedang cooldown dilewati."""
        raw = self._model_providers.get(model)

        # Coba fuzzy match jika tidak ada exact match
        if not raw:
            for key, val in self._model_providers.items():
                if model in key or key in model:
                    raw = val
                    logger.info(f"Fuzzy match: '{model}' → '{key}'")
                    break

        if not raw:
            return []

        scored: list[tuple[float, str, Any]] = []
        for p in raw:
            pn = self._pname(p)
            h = self._health.get(pn)
            if h and not h.is_healthy:
                continue                     # sedang cooldown
            scored.append((h.score if h else 0.5, pn, p))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [(n, c) for _, n, c in scored]

    # -- reporting ---------------------------------------------------

    def report_success(self, name: str):
        if name in self._health:
            self._health[name].record_success()

    def report_failure(self, name: str, reason: str = ""):
        if name in self._health:
            self._health[name].record_failure()

    # -- info --------------------------------------------------------

    def list_models(self) -> List[str]:
        return sorted(self._model_providers.keys())

    def status_report(self) -> Dict[str, Any]:
        return {
            n: {
                "healthy": h.is_healthy,
                "score": round(h.score, 3),
                "successes": h.success_count,
                "failures": h.failure_count,
                "consecutive_failures": h.consecutive_failures,
            }
            for n, h in sorted(self._health.items())
        }