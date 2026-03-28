"""
provider_scanner.py
~~~~~~~~~~~~~~~~~~~
Inisialisasi scanner yang:
  1. Menemukan semua pasangan (model, provider) dari g4f
  2. Mengirim request tes ke setiap pasangan secara konkuren
  3. Menyimpan hanya provider yang aktif/berfungsi
  4. Meng-cache hasil ke disk dengan TTL
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import g4f

from config import (
    CACHE_FILE,
    CACHE_TTL,
    SCAN_CONCURRENCY,
    SCAN_TEST_MESSAGES,
    SCAN_TIMEOUT,
)

log = logging.getLogger("g4f-bridge.scanner")


# ══════════════════════════════════════════════════
#  DATA MODEL
# ══════════════════════════════════════════════════

@dataclass
class ScanResult:
    """Hasil pengujian satu pasangan (model, provider)."""

    model: str
    provider_name: str
    is_active: bool
    latency_ms: float
    error: Optional[str] = None
    tested_at: float = field(default_factory=time.time)
    provider_class: Any = field(default=None, repr=False)

    def to_dict(self) -> dict:
        """Serialisasi untuk cache JSON (tanpa provider_class)."""
        return {
            "model": self.model,
            "provider_name": self.provider_name,
            "is_active": self.is_active,
            "latency_ms": round(self.latency_ms, 2),
            "error": self.error,
            "tested_at": self.tested_at,
        }


# ══════════════════════════════════════════════════
#  SCANNER
# ══════════════════════════════════════════════════

class ProviderScanner:
    """Memindai dan menguji semua provider g4f."""

    def __init__(self):
        self._registry: Dict[str, Any] = {}     # name → class
        self._results: List[ScanResult] = []
        self._scan_duration: float = 0.0

    # ── PUBLIC ──────────────────────────────────────

    async def initialize(
        self, force_rescan: bool = False
    ) -> List[ScanResult]:
        """
        Entry point utama.
        1. Bangun registry provider
        2. Cek cache (jika tidak force_rescan)
        3. Jika cache expired/tidak ada → full scan
        4. Kembalikan hasil
        """
        self._build_registry()

        if not force_rescan:
            cached = self._load_cache()
            if cached is not None:
                self._results = cached
                return cached

        log.info("=" * 65)
        log.info("  MEMULAI PEMINDAIAN PROVIDER (full scan)")
        log.info("=" * 65)

        t0 = time.time()
        results = await self._full_scan()
        self._scan_duration = time.time() - t0
        self._results = results

        self._save_cache(results)
        self._print_summary(results)
        return results

    @property
    def results(self) -> List[ScanResult]:
        return self._results

    @property
    def scan_duration(self) -> float:
        return self._scan_duration

    # ── DISCOVERY ───────────────────────────────────

    def _build_registry(self):
        """Bangun lookup name→class dari g4f.Provider."""
        try:
            import g4f.Provider as pmod

            for name in dir(pmod):
                obj = getattr(pmod, name, None)
                if obj is None:
                    continue
                # Provider punya minimal create/create_async
                has_create = (
                    hasattr(obj, "create")
                    or hasattr(obj, "create_async")
                )
                if has_create and hasattr(obj, "__name__"):
                    self._registry[name] = obj

            log.info(
                f"📚 Registry: {len(self._registry)} provider classes "
                f"ditemukan di g4f.Provider"
            )
        except Exception as exc:
            log.warning(f"Gagal membangun registry: {exc}")

    def _discover_pairs(self) -> List[Tuple[str, str, Any]]:
        """
        Temukan semua pasangan unik (model_name, provider_name, provider_class)
        dari registry internal g4f.
        """
        pairs: List[Tuple[str, str, Any]] = []
        seen: set[Tuple[str, str]] = set()

        try:
            from g4f.models import ModelUtils
        except ImportError:
            log.error("Tidak bisa import g4f.models.ModelUtils")
            return pairs

        for model_name, model_obj in ModelUtils.convert.items():
            providers = self._extract_providers(model_obj)
            for pcls in providers:
                pname = self._pname(pcls)
                key = (model_name, pname)
                if key in seen:
                    continue
                seen.add(key)
                pairs.append((model_name, pname, pcls))
                # Pastikan ada di registry
                if pname not in self._registry:
                    self._registry[pname] = pcls

        log.info(
            f"🔎 Ditemukan {len(pairs)} pasangan unik "
            f"(model × provider)"
        )
        return pairs

    @staticmethod
    def _extract_providers(model_obj) -> List[Any]:
        """Ekstrak provider konkret dari model.best_provider."""
        bp = getattr(model_obj, "best_provider", None)
        if bp is None:
            return []
        # IterListProvider / RetryProvider membungkus list
        if hasattr(bp, "providers"):
            return [
                p for p in bp.providers
                if getattr(p, "working", True)
            ]
        if getattr(bp, "working", True):
            return [bp]
        return []

    @staticmethod
    def _pname(provider) -> str:
        return getattr(provider, "__name__", type(provider).__name__)

    # ── TESTING ─────────────────────────────────────

    async def _test_one(
        self, model: str, pname: str, pcls: Any
    ) -> ScanResult:
        """Uji satu pasangan (model, provider) dengan request ringan."""
        t0 = time.time()
        try:
            text = await asyncio.wait_for(
                self._probe(model, pcls),
                timeout=SCAN_TIMEOUT,
            )
            elapsed = (time.time() - t0) * 1000

            if not text or not str(text).strip():
                raise ValueError("Respons kosong")

            log.info(
                f"  ✅  {pname:32s} │ {model:28s} │ "
                f"{elapsed:7.0f} ms"
            )
            return ScanResult(
                model=model,
                provider_name=pname,
                provider_class=pcls,
                is_active=True,
                latency_ms=elapsed,
            )

        except asyncio.TimeoutError:
            elapsed = (time.time() - t0) * 1000
            log.debug(
                f"  ⏱️  {pname:32s} │ {model:28s} │ TIMEOUT"
            )
            return ScanResult(
                model=model,
                provider_name=pname,
                provider_class=pcls,
                is_active=False,
                latency_ms=elapsed,
                error="timeout",
            )

        except Exception as exc:
            elapsed = (time.time() - t0) * 1000
            err = str(exc)[:200]
            log.debug(
                f"  ❌  {pname:32s} │ {model:28s} │ "
                f"{err[:50]}"
            )
            return ScanResult(
                model=model,
                provider_name=pname,
                provider_class=pcls,
                is_active=False,
                latency_ms=elapsed,
                error=err,
            )

    async def _probe(self, model: str, pcls: Any) -> str:
        """
        Kirim pesan tes ke g4f.
        Coba async dulu; fallback ke sync di thread pool.
        """
        kw: dict[str, Any] = {
            "model": model,
            "provider": pcls,
            "messages": SCAN_TEST_MESSAGES,
        }
        # --- async path ---
        try:
            return await g4f.ChatCompletion.create_async(**kw)
        except (AttributeError, NotImplementedError, TypeError):
            pass

        # --- sync fallback ---
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None, lambda: g4f.ChatCompletion.create(**kw)
        )

    async def _full_scan(self) -> List[ScanResult]:
        """Jalankan tes pada semua pasangan secara konkuren."""
        pairs = self._discover_pairs()
        if not pairs:
            log.warning("⚠️  Tidak ada pasangan yang ditemukan!")
            return []

        log.info(
            f"🧪 Menguji {len(pairs)} pasangan "
            f"(maks {SCAN_CONCURRENCY} bersamaan, "
            f"timeout {SCAN_TIMEOUT}s/tes)..."
        )
        log.info(f"  {'Provider':32s} │ {'Model':28s} │ Hasil")
        log.info(f"  {'─' * 32} │ {'─' * 28} │ {'─' * 15}")

        sem = asyncio.Semaphore(SCAN_CONCURRENCY)

        async def _guarded(m: str, pn: str, pc: Any) -> ScanResult:
            async with sem:
                return await self._test_one(m, pn, pc)

        tasks = [_guarded(m, pn, pc) for m, pn, pc in pairs]
        raw = await asyncio.gather(*tasks, return_exceptions=True)

        results: List[ScanResult] = []
        for item in raw:
            if isinstance(item, ScanResult):
                results.append(item)
            elif isinstance(item, BaseException):
                log.error(f"  Unexpected scan error: {item}")

        return results

    # ── SUMMARY ─────────────────────────────────────

    def _print_summary(self, results: List[ScanResult]):
        active = [r for r in results if r.is_active]
        failed = [r for r in results if not r.is_active]
        models = sorted({r.model for r in active})
        providers = sorted({r.provider_name for r in active})

        log.info("=" * 65)
        log.info("  HASIL PEMINDAIAN")
        log.info("=" * 65)
        log.info(f"  Total diuji      : {len(results)}")
        log.info(f"  Aktif             : {len(active)}")
        log.info(f"  Gagal             : {len(failed)}")
        log.info(f"  Model aktif       : {len(models)}")
        log.info(f"  Provider aktif    : {len(providers)}")
        log.info(f"  Durasi scan       : {self._scan_duration:.1f}s")

        if active:
            fastest = min(active, key=lambda r: r.latency_ms)
            log.info(
                f"  Tercepat          : {fastest.provider_name} "
                f"({fastest.model}) — {fastest.latency_ms:.0f}ms"
            )

        # Tampilkan error breakdown
        if failed:
            errors: dict[str, int] = {}
            for r in failed:
                cat = (r.error or "unknown").split(":")[0][:30]
                errors[cat] = errors.get(cat, 0) + 1
            log.info("  Error breakdown   :")
            for cat, cnt in sorted(
                errors.items(), key=lambda x: -x[1]
            )[:5]:
                log.info(f"    {cat:30s} → {cnt}")

        log.info("=" * 65)

    # ── CACHE ───────────────────────────────────────

    def _save_cache(self, results: List[ScanResult]):
        """Simpan hasil scan ke file JSON."""
        try:
            payload = {
                "version": 3,
                "scanned_at": time.time(),
                "ttl_seconds": CACHE_TTL,
                "scan_duration": round(self._scan_duration, 2),
                "total": len(results),
                "active": sum(1 for r in results if r.is_active),
                "results": [r.to_dict() for r in results],
            }
            CACHE_FILE.write_text(
                json.dumps(payload, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            log.info(
                f"💾 Cache disimpan → {CACHE_FILE} "
                f"({len(results)} entri, TTL {CACHE_TTL}s)"
            )
        except Exception as exc:
            log.warning(f"Gagal menyimpan cache: {exc}")

    def _load_cache(self) -> Optional[List[ScanResult]]:
        """
        Muat hasil scan dari cache jika masih berlaku.
        Mengembalikan None jika cache tidak ada, rusak, atau expired.
        """
        if not CACHE_FILE.exists():
            log.info("📭 Tidak ada file cache")
            return None

        try:
            raw = json.loads(CACHE_FILE.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            log.warning(f"Cache rusak: {exc}")
            return None

        # Cek versi
        if raw.get("version", 0) < 3:
            log.info("📭 Versi cache lama, akan rescan")
            return None

        # Cek TTL
        scanned_at = raw.get("scanned_at", 0)
        ttl = raw.get("ttl_seconds", CACHE_TTL)
        age = time.time() - scanned_at
        if age > ttl:
            log.info(f"📭 Cache expired ({age:.0f}s > {ttl}s)")
            return None

        # Rekonstruksi objek ScanResult
        results: List[ScanResult] = []
        unresolved = 0
        for entry in raw.get("results", []):
            pname = entry["provider_name"]
            pcls = self._registry.get(pname)
            if pcls is None:
                unresolved += 1
            results.append(
                ScanResult(
                    model=entry["model"],
                    provider_name=pname,
                    provider_class=pcls,
                    is_active=entry["is_active"] and pcls is not None,
                    latency_ms=entry["latency_ms"],
                    error=entry.get("error"),
                    tested_at=entry.get("tested_at", scanned_at),
                )
            )

        remaining = ttl - age
        active = sum(1 for r in results if r.is_active)
        log.info(
            f"📦 Cache valid ({remaining:.0f}s tersisa) — "
            f"{active}/{len(results)} aktif"
            + (f", {unresolved} unresolved" if unresolved else "")
        )
        return results