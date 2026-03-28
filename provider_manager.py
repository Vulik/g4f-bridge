"""
provider_manager.py
~~~~~~~~~~~~~~~~~~~
Manajemen provider saat runtime:
  • Health tracking per provider (success/failure/latency)
  • Cooldown otomatis untuk provider bermasalah
  • Seleksi otomatis model terbaik (mode "auto")
  • Fallback chain dengan retry berjenjang
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple

import g4f

from config import (
    AUTO_MODEL_PRIORITY,
    COOLDOWN_BASE,
    COOLDOWN_MAX,
    MAX_RETRIES,
    REQUEST_TIMEOUT,
    RETRY_BACKOFF,
)

log = logging.getLogger("g4f-bridge.manager")


# ══════════════════════════════════════════════════
#  HEALTH TRACKER
# ══════════════════════════════════════════════════

@dataclass
class ProviderHealth:
    """Melacak kondisi runtime satu provider."""

    name: str
    provider_class: Any
    models: List[str] = field(default_factory=list)
    scan_latency_ms: float = 5000.0

    # — runtime stats —
    success_count: int = 0
    failure_count: int = 0
    total_latency_ms: float = 0.0
    last_success: float = 0.0
    last_failure: float = 0.0
    consecutive_failures: int = 0

    # — properties ────────────────────────────────

    @property
    def is_healthy(self) -> bool:
        if self.consecutive_failures == 0:
            return True
        if self.last_success > self.last_failure:
            return True
        cd = min(COOLDOWN_MAX, COOLDOWN_BASE * self.consecutive_failures)
        return (time.time() - self.last_failure) > cd

    @property
    def avg_latency_ms(self) -> float:
        if self.success_count == 0:
            return self.scan_latency_ms
        return self.total_latency_ms / self.success_count

    @property
    def success_rate(self) -> float:
        total = self.success_count + self.failure_count
        return self.success_count / total if total > 0 else 0.5

    @property
    def score(self) -> float:
        """Skor komposit: semakin tinggi semakin baik."""
        sr = self.success_rate
        lat = 1.0 / (1.0 + self.avg_latency_ms / 1000.0)
        recency = 0.1 if (
            self.last_success > 0
            and (time.time() - self.last_success) < 300
        ) else 0.0
        penalty = min(0.5, 0.1 * self.consecutive_failures)
        return sr * 0.5 + lat * 0.3 + recency - penalty

    # — mutators ──────────────────────────────────

    def record_success(self, latency_ms: float = 0.0):
        self.success_count += 1
        self.total_latency_ms += latency_ms
        self.last_success = time.time()
        self.consecutive_failures = 0

    def record_failure(self):
        self.failure_count += 1
        self.last_failure = time.time()
        self.consecutive_failures += 1


# ══════════════════════════════════════════════════
#  PROVIDER MANAGER
# ══════════════════════════════════════════════════

class ProviderManager:
    """Sentral runtime manager untuk provider g4f."""

    def __init__(self):
        self._providers: Dict[str, ProviderHealth] = {}
        self._model_providers: Dict[str, List[str]] = {}
        self._active_models: List[str] = []

    # ── INISIALISASI ────────────────────────────────

    def load_scan_results(self, results) -> None:
        """Muat hasil dari ProviderScanner (hanya yang aktif)."""
        from provider_scanner import ScanResult

        self._providers.clear()
        self._model_providers.clear()

        for r in results:
            if not isinstance(r, ScanResult):
                continue
            if not r.is_active or r.provider_class is None:
                continue

            pname = r.provider_name

            # Buat/update health
            if pname not in self._providers:
                self._providers[pname] = ProviderHealth(
                    name=pname,
                    provider_class=r.provider_class,
                    scan_latency_ms=r.latency_ms,
                )
            h = self._providers[pname]
            if r.model not in h.models:
                h.models.append(r.model)

            # Map model → provider names
            self._model_providers.setdefault(r.model, [])
            if pname not in self._model_providers[r.model]:
                self._model_providers[r.model].append(pname)

        self._active_models = sorted(self._model_providers.keys())

        log.info(
            f"📋 Dimuat: {len(self._providers)} provider aktif, "
            f"{len(self._active_models)} model"
        )

    # ── SELEKSI PROVIDER ────────────────────────────

    def get_ranked_providers(
        self, model: str
    ) -> List[Tuple[str, Any]]:
        """
        Dapatkan provider untuk model tertentu,
        diurutkan berdasarkan skor (tertinggi dulu),
        melewati yang sedang cooldown.
        """
        pnames = self._model_providers.get(model)

        # Fuzzy match
        if not pnames:
            for key in self._model_providers:
                if model in key or key in model:
                    pnames = self._model_providers[key]
                    log.info(f"Fuzzy match: '{model}' → '{key}'")
                    break

        if not pnames:
            return []

        scored: list[tuple[float, str, Any]] = []
        for pn in pnames:
            h = self._providers.get(pn)
            if not h or not h.is_healthy:
                continue
            scored.append((h.score, pn, h.provider_class))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [(n, c) for _, n, c in scored]

    def select_auto(
        self,
    ) -> List[Tuple[str, str, Any]]:
        """
        Mode auto: kembalikan chain [(model, provider_name, cls), ...]
        diurutkan berdasarkan skor + preferensi model.
        """
        candidates: list[tuple[float, str, str, Any]] = []

        for model, pnames in self._model_providers.items():
            # Bonus preferensi model
            try:
                idx = AUTO_MODEL_PRIORITY.index(model)
                bonus = 0.3 * (
                    1.0 - idx / max(len(AUTO_MODEL_PRIORITY), 1)
                )
            except ValueError:
                bonus = 0.0

            for pn in pnames:
                h = self._providers.get(pn)
                if not h or not h.is_healthy:
                    continue
                candidates.append(
                    (h.score + bonus, model, pn, h.provider_class)
                )

        candidates.sort(key=lambda x: x[0], reverse=True)

        if not candidates:
            log.warning("⚠️  Tidak ada provider aktif untuk auto-select!")
            return []

        top = candidates[0]
        log.info(
            f"🎯 Auto-select terbaik: {top[2]} ({top[1]}) "
            f"skor={top[0]:.3f}"
        )
        return [(m, n, c) for _, m, n, c in candidates]

    # ── G4F CALL WRAPPER ────────────────────────────

    async def _call_g4f(
        self,
        model: str,
        messages: list[dict],
        provider_cls: Any | None,
        temperature: float | None,
        max_tokens: int | None,
    ) -> str:
        """Panggil g4f (non-streaming)."""
        kw: dict[str, Any] = {"model": model, "messages": messages}
        if provider_cls is not None:
            kw["provider"] = provider_cls
        if temperature is not None:
            kw["temperature"] = temperature
        if max_tokens is not None:
            kw["max_tokens"] = max_tokens

        try:
            return await g4f.ChatCompletion.create_async(**kw)
        except (AttributeError, NotImplementedError, TypeError):
            pass

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None, lambda: g4f.ChatCompletion.create(**kw)
        )

    async def _stream_g4f(
        self,
        model: str,
        messages: list[dict],
        provider_cls: Any | None,
        temperature: float | None,
        max_tokens: int | None,
    ) -> AsyncGenerator[str, None]:
        """Streaming dari g4f via thread → asyncio.Queue bridge."""
        kw: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": True,
        }
        if provider_cls is not None:
            kw["provider"] = provider_cls
        if temperature is not None:
            kw["temperature"] = temperature
        if max_tokens is not None:
            kw["max_tokens"] = max_tokens

        queue: asyncio.Queue[tuple[str, Any]] = asyncio.Queue()
        loop = asyncio.get_running_loop()

        def _producer():
            try:
                resp = g4f.ChatCompletion.create(**kw)
                if isinstance(resp, str):
                    loop.call_soon_threadsafe(
                        queue.put_nowait, ("chunk", resp)
                    )
                else:
                    for token in resp:
                        if token:
                            loop.call_soon_threadsafe(
                                queue.put_nowait, ("chunk", str(token))
                            )
                loop.call_soon_threadsafe(
                    queue.put_nowait, ("done", None)
                )
            except Exception as exc:
                loop.call_soon_threadsafe(
                    queue.put_nowait, ("error", exc)
                )

        loop.run_in_executor(None, _producer)

        while True:
            kind, payload = await queue.get()
            if kind == "done":
                return
            if kind == "error":
                raise payload
            yield payload

    # ── RETRY WRAPPER ───────────────────────────────

    async def call_with_retry(
        self,
        model: str,
        messages: list[dict],
        provider_cls: Any,
        provider_name: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """
        Panggil satu provider dengan mekanisme retry.
        
        Retry ke-N menunggu RETRY_BACKOFF × 2^(N-1) detik.
        Jika semua retry gagal, raise exception terakhir.
        """
        last_error: Exception | None = None

        for attempt in range(1 + MAX_RETRIES):
            # Backoff sebelum retry (bukan attempt pertama)
            if attempt > 0:
                wait = RETRY_BACKOFF * (2 ** (attempt - 1))
                log.info(
                    f"    🔄 Retry {attempt}/{MAX_RETRIES} "
                    f"{provider_name} (tunggu {wait:.1f}s)"
                )
                await asyncio.sleep(wait)

            t0 = time.time()
            try:
                text = await asyncio.wait_for(
                    self._call_g4f(
                        model, messages, provider_cls,
                        temperature, max_tokens,
                    ),
                    timeout=REQUEST_TIMEOUT,
                )
                elapsed = (time.time() - t0) * 1000

                if not text or not str(text).strip():
                    raise ValueError("Respons kosong")

                # Sukses!
                self._providers[provider_name].record_success(elapsed)
                log.info(
                    f"    ✅ {provider_name} berhasil "
                    f"(attempt {attempt + 1}, {elapsed:.0f}ms)"
                )
                return str(text).strip()

            except asyncio.TimeoutError:
                elapsed = (time.time() - t0) * 1000
                last_error = TimeoutError(
                    f"{provider_name}: timeout {elapsed:.0f}ms"
                )
                log.warning(
                    f"    ⏱️ {provider_name} "
                    f"attempt {attempt + 1}: timeout"
                )

            except Exception as exc:
                last_error = exc
                log.warning(
                    f"    ❌ {provider_name} "
                    f"attempt {attempt + 1}: {exc}"
                )

        # Semua retry habis → record failure
        if provider_name in self._providers:
            self._providers[provider_name].record_failure()
        raise last_error  # type: ignore[misc]

    # ── FALLBACK CHAIN (NON-STREAMING) ──────────────

    async def execute_with_fallback(
        self,
        model: str,
        messages: list[dict],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> Tuple[str, str, str]:
        """
        Coba setiap provider secara berurutan (dengan retry per provider).
        Mengembalikan (response_text, actual_model, provider_name).
        """
        is_auto = model.lower() == "auto"

        if is_auto:
            chain = self.select_auto()
            if not chain:
                raise RuntimeError("Tidak ada provider aktif")
            attempts = [
                (m, n, c) for m, n, c in chain
            ]
        else:
            ranked = self.get_ranked_providers(model)
            attempts = [(model, n, c) for n, c in ranked]

        # Jaring pengaman terakhir: g4f tanpa provider spesifik
        fallback_model = model if not is_auto else "gpt-4o-mini"
        attempts.append((fallback_model, "g4f-auto-select", None))

        last_err = ""

        for m, pname, pcls in attempts:
            try:
                log.info(f"  ⏩ [{m}] mencoba: {pname}")
                text = await self.call_with_retry(
                    model=m,
                    messages=messages,
                    provider_cls=pcls,
                    provider_name=pname,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                return text, m, pname

            except Exception as exc:
                last_err = f"{pname} ({m}): {exc}"
                log.warning(f"  ⛔ {pname} ({m}) gagal total: {exc}")
                continue

        raise RuntimeError(
            f"Semua provider gagal. Terakhir: {last_err}"
        )

    # ── FALLBACK CHAIN (STREAMING) ──────────────────

    async def stream_with_fallback(
        self,
        model: str,
        messages: list[dict],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> Tuple[AsyncGenerator[str, None], str, str]:
        """
        Sama seperti execute_with_fallback tapi streaming.
        Menarik chunk pertama sebagai validasi sebelum commit.
        Mengembalikan (async_generator, actual_model, provider_name).
        """
        is_auto = model.lower() == "auto"

        if is_auto:
            chain = self.select_auto()
            attempts = [(m, n, c) for m, n, c in chain] if chain else []
        else:
            ranked = self.get_ranked_providers(model)
            attempts = [(model, n, c) for n, c in ranked]

        fallback_model = model if not is_auto else "gpt-4o-mini"
        attempts.append((fallback_model, "g4f-auto-select", None))

        last_err = ""

        for m, pname, pcls in attempts:
            try:
                log.info(f"  ⏩ [{m}] stream: mencoba {pname}")
                gen = self._stream_g4f(
                    m, messages, pcls, temperature, max_tokens
                )

                # Validasi: tarik chunk pertama
                first = await asyncio.wait_for(
                    gen.__anext__(), timeout=REQUEST_TIMEOUT
                )

                if pname in self._providers:
                    self._providers[pname].record_success(0)

                log.info(f"  ✅ [{m}] stream aktif via {pname}")

                # Re-chain chunk pertama ke generator
                async def _rechain(
                    first_chunk: str, rest: AsyncGenerator
                ) -> AsyncGenerator[str, None]:
                    yield first_chunk
                    async for c in rest:
                        yield c

                return _rechain(first, gen), m, pname

            except StopAsyncIteration:
                last_err = f"{pname}: stream kosong"
                log.warning(f"  ❌ [{m}] {pname}: stream kosong")
                if pname in self._providers:
                    self._providers[pname].record_failure()

            except asyncio.TimeoutError:
                last_err = f"{pname}: timeout"
                log.warning(f"  ⏱️ [{m}] {pname}: stream timeout")
                if pname in self._providers:
                    self._providers[pname].record_failure()

            except Exception as exc:
                last_err = f"{pname}: {exc}"
                log.warning(f"  ❌ [{m}] {pname}: {exc}")
                if pname in self._providers:
                    self._providers[pname].record_failure()

        raise RuntimeError(
            f"Semua provider stream gagal. Terakhir: {last_err}"
        )

    # ── INFO & STATUS ───────────────────────────────

    def list_models(self) -> List[str]:
        return list(self._active_models)

    def status_report(self) -> Dict[str, Any]:
        return {
            n: {
                "healthy": h.is_healthy,
                "score": round(h.score, 4),
                "success_rate": round(h.success_rate, 4),
                "avg_latency_ms": round(h.avg_latency_ms, 1),
                "successes": h.success_count,
                "failures": h.failure_count,
                "consecutive_failures": h.consecutive_failures,
                "models": h.models,
            }
            for n, h in sorted(self._providers.items())
        }

    def summary(self) -> Dict[str, Any]:
        total = len(self._providers)
        healthy = sum(
            1 for h in self._providers.values() if h.is_healthy
        )
        return {
            "total_providers": total,
            "healthy_providers": healthy,
            "total_models": len(self._active_models),
            "models": self._active_models,
        }