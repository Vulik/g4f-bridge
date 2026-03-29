"""
server.py  v7
═════════════
G4F Bridge — Rate-Limited (5 req/min) + Exact OpenAI Format

Changelog v6 → v7:
  ✦ Rate limiter: sliding window 5 req/60s, request di-queue
  ✦ JSON output dicocokkan persis dengan real OpenAI API
  ✦ Scan rate-limited (tidak membanjiri g4f)
  ✦ Scan hanya model preferred (hemat kuota)
  ✦ Progress bar & ETA saat scan
  ✦ Queue position logging saat runtime
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import sys
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

try:
    import g4f
except ImportError:
    sys.exit("\n  ❌ pip install g4f\n")

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field


# ══════════════════════════════════════════════════════════════
#  KONFIGURASI
# ══════════════════════════════════════════════════════════════

HOST                  = os.getenv("HOST", "0.0.0.0")
PORT                  = int(os.getenv("PORT", "8820"))
G4F_TIMEOUT           = int(os.getenv("G4F_TIMEOUT", "120"))
SCAN_TIMEOUT          = int(os.getenv("SCAN_TIMEOUT", "25"))
MAX_RETRIES           = int(os.getenv("MAX_RETRIES", "1"))
RETRY_BACKOFF         = float(os.getenv("RETRY_BACKOFF", "2.0"))
COOLDOWN_BASE         = int(os.getenv("COOLDOWN_BASE", "60"))
COOLDOWN_MAX          = int(os.getenv("COOLDOWN_MAX", "600"))
MAX_TOOL_TRIES        = int(os.getenv("MAX_TOOL_TRIES", "3"))
MAX_CONSECUTIVE_TOOLS = int(os.getenv("MAX_CONSECUTIVE_TOOLS", "2"))
LOG_LEVEL             = os.getenv("LOG_LEVEL", "DEBUG")

# Rate limit g4f
G4F_RPM      = int(os.getenv("G4F_RPM", "5"))       # request per menit
G4F_PERIOD   = float(os.getenv("G4F_PERIOD", "60"))  # window (detik)
QUEUE_TIMEOUT = int(os.getenv("QUEUE_TIMEOUT", "180")) # max tunggu di queue

# Scan
SCAN_ONLY_PREFERRED = os.getenv("SCAN_ALL", "0") != "1"

PREFERRED_MODELS: list[str] = [
    "gpt-4o-mini", "gpt-4o", "gpt-4", "gpt-3.5-turbo",
    "claude-3.5-sonnet", "claude-3-haiku",
    "llama-3.1-70b", "llama-3.1-8b", "mixtral-8x7b",
]


# ══════════════════════════════════════════════════════════════
#  LOGGING
# ══════════════════════════════════════════════════════════════

log = logging.getLogger("bridge")
log.handlers.clear()
log.setLevel(getattr(logging, LOG_LEVEL, logging.DEBUG))
_h = logging.StreamHandler(sys.stdout)
_h.setFormatter(logging.Formatter(
    "%(asctime)s │ %(levelname)-7s │ %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"))
log.addHandler(_h)
log.propagate = False


def _sep(title="", w=72):
    if title:
        pad = max(1, (w - len(title) - 2) // 2)
        log.info(f"{'─' * pad} {title} {'─' * pad}")
    else:
        log.info("─" * w)


def _log_json(label, data, limit=2500):
    try:
        txt = json.dumps(data, ensure_ascii=False, indent=2)
    except (TypeError, ValueError):
        txt = str(data)
    if len(txt) > limit:
        txt = txt[:limit] + f"\n  … [{len(txt)} ch]"
    for line in txt.split("\n"):
        log.debug(f"  {label:>14s} │ {line}")


# ══════════════════════════════════════════════════════════════
#  RATE LIMITER — Sliding Window Queue
# ══════════════════════════════════════════════════════════════

class RateLimiter:
    """
    Sliding window rate limiter.

    Mekanisme:
      - Menyimpan timestamp setiap panggilan g4f
      - Sebelum panggilan baru, cek apakah sudah ada
        G4F_RPM panggilan dalam G4F_PERIOD detik terakhir
      - Jika ya: tunggu sampai slot tersedia
      - Jika tidak: langsung eksekusi

    ┌──────── 60 detik window ─────────┐
    │  [call1] [call2] [call3] [c4] [c5] │  ← 5 calls = FULL
    │       ↑                             │
    │   call1 expire → slot terbuka       │
    └─────────────────────────────────────┘
    """

    def __init__(self, max_calls: int, period: float):
        self.max_calls = max_calls
        self.period    = period
        self._times: deque[float] = deque()
        self._lock = asyncio.Lock()
        self._waiters = 0

    async def acquire(self, label: str = ""):
        """Tunggu sampai ada slot, lalu ambil."""
        async with self._lock:
            now = time.time()

            # Buang timestamp yang sudah expired
            while self._times and self._times[0] < now - self.period:
                self._times.popleft()

            if len(self._times) >= self.max_calls:
                # Hitung waktu tunggu
                wait = self._times[0] + self.period - now + 0.1
                remaining = len(self._times) - self.max_calls + 1
                self._waiters += 1
                pos = self._waiters

                log.info(
                    f"  ⏳ Rate limit: slot penuh "
                    f"({len(self._times)}/{self.max_calls}), "
                    f"antrian #{pos}, tunggu {wait:.1f}s"
                    f"{f' [{label}]' if label else ''}")

                # Release lock saat menunggu
                self._lock.release()
                try:
                    await asyncio.sleep(wait)
                finally:
                    await self._lock.acquire()
                    self._waiters -= 1

                # Bersihkan lagi setelah menunggu
                now = time.time()
                while (self._times and
                       self._times[0] < now - self.period):
                    self._times.popleft()

            self._times.append(time.time())
            slots_left = self.max_calls - len(self._times)
            log.debug(f"  🎫 Rate: slot terpakai "
                      f"{len(self._times)}/{self.max_calls}, "
                      f"sisa {slots_left}"
                      f"{f' [{label}]' if label else ''}")

    @property
    def available_slots(self) -> int:
        now = time.time()
        active = sum(1 for t in self._times
                     if t >= now - self.period)
        return max(0, self.max_calls - active)

    @property
    def next_slot_in(self) -> float:
        """Detik sampai slot berikutnya tersedia."""
        if self.available_slots > 0:
            return 0.0
        now = time.time()
        if self._times:
            return max(0, self._times[0] + self.period - now)
        return 0.0

    @property
    def queue_size(self) -> int:
        return self._waiters


# Global rate limiter
rate_limiter = RateLimiter(G4F_RPM, G4F_PERIOD)


# ══════════════════════════════════════════════════════════════
#  SCAN PROMPTS
# ══════════════════════════════════════════════════════════════

_ALIVE_MSGS = [{"role": "user", "content": "Reply: ok"}]
_TOOL_MSGS  = [
    {"role": "system", "content": (
        "You have ONE tool: test_func(q: string).\n"
        "You MUST call it. Reply ONLY with JSON:\n"
        '{"nama_alat":"test_func","argumen":{"q":"<question>"}}\n'
        "No markdown. Only JSON.")},
    {"role": "user", "content": "What is 1+1?"},
]


def _is_tool_resp(text: str) -> bool:
    if not text:
        return False
    cl = re.sub(r"```(?:json)?\s*\n?(.*?)\n?\s*```",
                r"\1", text, flags=re.DOTALL).strip()
    try:
        obj = json.loads(cl)
        if isinstance(obj, dict):
            for k in ("nama_alat", "name"):
                if obj.get(k) == "test_func":
                    return True
    except (json.JSONDecodeError, TypeError):
        pass
    return False


# ══════════════════════════════════════════════════════════════
#  PROVIDER REGISTRY
# ══════════════════════════════════════════════════════════════

@dataclass
class Health:
    successes: int = 0
    failures: int = 0
    consecutive_fails: int = 0
    total_lat_ms: float = 0.0
    last_ok: float = 0.0
    last_fail: float = 0.0
    tool_capable: bool = False
    tool_tested: bool = False

    @property
    def healthy(self) -> bool:
        if self.consecutive_fails == 0:
            return True
        cd = min(COOLDOWN_MAX,
                 COOLDOWN_BASE * self.consecutive_fails)
        return (time.time() - self.last_fail) > cd

    @property
    def avg_lat(self) -> float:
        return ((self.total_lat_ms / self.successes)
                if self.successes else 9999.0)

    @property
    def score(self) -> float:
        total = self.successes + self.failures
        if total == 0:
            return 0.5
        rate  = self.successes / total
        lat   = 1.0 / (1.0 + self.avg_lat / 5000.0)
        fresh = (0.1 if self.last_ok
                 and (time.time() - self.last_ok) < 300 else 0.0)
        pen   = min(0.5, 0.15 * self.consecutive_fails)
        return rate * 0.5 + lat * 0.3 + fresh - pen

    def ok(self, ms=0.0):
        self.successes += 1
        self.total_lat_ms += ms
        self.last_ok = time.time()
        self.consecutive_fails = 0

    def fail(self):
        self.failures += 1
        self.last_fail = time.time()
        self.consecutive_fails += 1


class Registry:

    def __init__(self):
        self._map: Dict[str, List[Tuple[str, Any]]] = {}
        self._hp: Dict[str, Health] = {}
        self._discover()

    def _discover(self):
        try:
            from g4f.models import ModelUtils
        except ImportError:
            log.warning("⚠️  g4f.models.ModelUtils unavailable")
            return
        for mn, mo in ModelUtils.convert.items():
            provs = self._ext(mo)
            if not provs:
                continue
            pairs = []
            for p in provs:
                pn = self._pn(p)
                pairs.append((pn, p))
                if pn not in self._hp:
                    self._hp[pn] = Health()
            self._map[mn] = pairs

    @staticmethod
    def _ext(mo) -> list:
        bp = getattr(mo, "best_provider", None)
        if bp is None:
            return []
        if hasattr(bp, "providers"):
            return [p for p in bp.providers
                    if getattr(p, "working", True)]
        return [bp] if getattr(bp, "working", True) else []

    @staticmethod
    def _pn(p) -> str:
        return getattr(p, "__name__", type(p).__name__)

    # ── RATE-LIMITED SCAN ──────────────────────────

    async def scan_all(self):
        """Model-first scan, rate-limited, hanya preferred."""

        # Tentukan model mana yang di-scan
        if SCAN_ONLY_PREFERRED:
            models = [m for m in PREFERRED_MODELS
                      if m in self._map]
            log.info(f"  📋 Scanning {len(models)} preferred "
                     f"model (set SCAN_ALL=1 for all)")
        else:
            models = sorted(self._map.keys())
            log.info(f"  📋 Scanning ALL {len(models)} model")

        all_pairs = [(m, pn, pc)
                     for m in models
                     for pn, pc in self._map.get(m, [])]

        if not all_pairs:
            log.warning("  ⚠️  Tidak ada pasangan")
            return

        # ── Show models ──
        _sep("DISCOVERY")
        for mdl in models:
            ps = self._map.get(mdl, [])
            pn = ", ".join(n for n, _ in ps)
            if len(pn) > 50:
                pn = pn[:47] + "..."
            log.info(f"  {mdl:30s} │ {len(ps):2d}p │ {pn}")

        # ── Phase 1: Alive (rate-limited) ──
        eta_s = len(all_pairs) * (G4F_PERIOD / G4F_RPM)
        _sep("PHASE 1: ALIVE")
        log.info(f"  🧪 {len(all_pairs)} pairs, "
                 f"~{G4F_RPM} req/{G4F_PERIOD:.0f}s, "
                 f"ETA ~{eta_s:.0f}s")

        alive = await self._test_all(all_pairs, _ALIVE_MSGS)
        alive_pairs = [(r[0], r[1], r[2])
                       for r in alive if r[3]]

        # Show per model
        by_m: dict[str, list] = {}
        for m, pn, pc, ok, ms, err in alive:
            by_m.setdefault(m, []).append((pn, ok, ms, err))

        log.info("")
        for mdl in models:
            items = by_m.get(mdl, [])
            n_ok  = sum(1 for _, ok, _, _ in items if ok)
            if n_ok == 0:
                continue
            log.info(f"  📋 {mdl} ({n_ok}/{len(items)})")
            for pn, ok, ms, err in sorted(
                    items, key=lambda x: (-x[1], x[2])):
                if ok:
                    log.info(f"     ✅ {pn:28s} │ {ms:6.0f}ms")
                else:
                    log.info(f"     ❌ {pn:28s} │ {err}")

        n_alive = len(alive_pairs)
        n_models = len({m for m, _, _ in alive_pairs})
        n_provs  = len({pn for _, pn, _ in alive_pairs})
        log.info(f"\n  📊 Alive: {n_alive} pairs, "
                 f"{n_models} model, {n_provs} provider")

        if not alive_pairs:
            return

        # ── Phase 2: Tool (rate-limited) ──
        eta_t = len(alive_pairs) * (G4F_PERIOD / G4F_RPM)
        _sep("PHASE 2: TOOL TEST")
        log.info(f"  🔧 {len(alive_pairs)} alive pairs, "
                 f"ETA ~{eta_t:.0f}s")

        tool_res = await self._test_all(
            alive_pairs, _TOOL_MSGS, check_tool=True)

        tool_ok_names: set[str] = set()
        log.info("")
        by_m2: dict[str, list] = {}
        for m, pn, pc, ok, ms, err in tool_res:
            by_m2.setdefault(m, []).append((pn, ok, ms))

        for mdl in models:
            items = by_m2.get(mdl, [])
            if not items:
                continue
            n_ok = sum(1 for _, ok, _ in items if ok)
            log.info(f"  📋 {mdl} ({n_ok}/{len(items)})")
            for pn, ok, ms in items:
                if ok:
                    log.info(f"     ✅ {pn:28s} │ {ms:6.0f}ms 🔧")
                else:
                    log.info(f"     ❌ {pn:28s} │ no tool")

        for m, pn, pc, ok, ms, err in tool_res:
            h = self._hp.get(pn)
            if not h:
                continue
            h.tool_tested = True
            if ok:
                h.tool_capable = True
                tool_ok_names.add(pn)
            else:
                h.tool_capable = False

        # Mark untested
        for _, pn, _ in alive_pairs:
            h = self._hp.get(pn)
            if h and not h.tool_tested:
                h.tool_tested = True
                h.tool_capable = False

        # ── Summary ──
        _sep("SCAN COMPLETE")
        log.info(f"  ✅ Alive        : {n_alive}")
        log.info(f"  📋 Models       : {n_models}")
        log.info(f"  👤 Providers    : {n_provs}")
        log.info(f"  🔧 Tool-capable : {len(tool_ok_names)}")
        if tool_ok_names:
            log.info(f"     {', '.join(sorted(tool_ok_names))}")
        _sep()

    async def _test_all(
        self,
        pairs: list[tuple[str, str, Any]],
        msgs: list[dict],
        check_tool: bool = False,
    ) -> list[tuple[str, str, Any, bool, float, str]]:
        """
        Tes semua pairs SECARA BERURUTAN dengan rate limiter.
        Return [(model, pname, pcls, success, latency_ms, error)]
        """
        results = []
        total   = len(pairs)

        for idx, (model, pname, pcls) in enumerate(pairs, 1):
            pct = idx / total * 100
            log.info(f"  [{idx:3d}/{total}] {pct:5.1f}% │ "
                     f"{pname:28s} │ {model}")

            # Rate limit
            await rate_limiter.acquire(f"scan:{pname}")

            t0 = time.time()
            try:
                kw = {"model": model, "messages": msgs,
                      "provider": pcls}
                try:
                    text = await asyncio.wait_for(
                        g4f.ChatCompletion.create_async(**kw),
                        timeout=SCAN_TIMEOUT)
                except (AttributeError, NotImplementedError,
                        TypeError):
                    loop = asyncio.get_running_loop()
                    text = await asyncio.wait_for(
                        loop.run_in_executor(
                            None,
                            lambda: g4f.ChatCompletion.create(**kw)),
                        timeout=SCAN_TIMEOUT)

                if hasattr(text, "__iter__") and \
                   not isinstance(text, str):
                    text = "".join(str(c) for c in text)
                text = str(text).strip() if text else ""
                ms = (time.time() - t0) * 1000

                if not text:
                    raise ValueError("empty")

                if check_tool:
                    ok = _is_tool_resp(text)
                else:
                    ok = True

                if ok:
                    h = self._hp.get(pname)
                    if h and not check_tool:
                        h.ok(ms)

                results.append((model, pname, pcls, ok, ms, ""))

            except asyncio.TimeoutError:
                results.append((model, pname, pcls, False, 0,
                                "timeout"))
            except Exception as exc:
                results.append((model, pname, pcls, False, 0,
                                str(exc)[:40]))

        return results

    # ── chain ─────────────────────────────────────

    def get_chain(self, model: str,
                  need_tools=False) -> list[tuple[str, str, Any]]:
        if model.lower() == "auto":
            return self._auto(need_tools)
        return self._specific(model, need_tools)

    def _auto(self, need_tools) -> list[tuple[str, str, Any]]:
        tc, ot = [], []
        for mdl in PREFERRED_MODELS:
            pairs = self._map.get(mdl, [])
            try:
                idx = PREFERRED_MODELS.index(mdl)
                bonus = 0.25 * (1.0 - idx / len(PREFERRED_MODELS))
            except ValueError:
                bonus = 0.0
            for pn, pcls in pairs:
                h = self._hp.get(pn, Health())
                if not h.healthy:
                    continue
                s = h.score + bonus
                (tc if h.tool_capable else ot).append(
                    (s, mdl, pn, pcls))

        for mdl, pairs in self._map.items():
            if mdl in PREFERRED_MODELS:
                continue
            for pn, pcls in pairs:
                h = self._hp.get(pn, Health())
                if not h.healthy:
                    continue
                (tc if h.tool_capable else ot).append(
                    (h.score, mdl, pn, pcls))

        tc.sort(key=lambda x: x[0], reverse=True)
        ot.sort(key=lambda x: x[0], reverse=True)
        ordered = ((tc + ot) if need_tools
                   else sorted(tc + ot,
                                key=lambda x: x[0], reverse=True))

        chain: list[tuple[str, str, Any]] = []
        seen: set[tuple[str, str]] = set()
        for _, m, pn, pc in ordered:
            key = (m, pn)
            if key in seen:
                continue
            seen.add(key)
            chain.append((m, pn, pc))
            if len(chain) >= 12:
                break

        chain.append(("gpt-4o-mini", "g4f-auto", None))
        return chain

    def _specific(self, model, need_tools
                  ) -> list[tuple[str, str, Any]]:
        pairs = self._map.get(model)
        if not pairs:
            for k in self._map:
                if model in k or k in model:
                    pairs = self._map[k]
                    model = k
                    break

        chain: list[tuple[str, str, Any]] = []
        if pairs:
            tp, op = [], []
            for pn, pc in pairs:
                h = self._hp.get(pn, Health())
                if not h.healthy:
                    continue
                (tp if h.tool_capable else op).append(
                    (h.score, model, pn, pc))
            tp.sort(key=lambda x: x[0], reverse=True)
            op.sort(key=lambda x: x[0], reverse=True)
            o = (tp + op) if need_tools else sorted(
                tp + op, key=lambda x: x[0], reverse=True)
            for _, m, pn, pc in o:
                chain.append((m, pn, pc))

        chain.append((model, "g4f-auto", None))
        return chain

    def rec_ok(self, pn, ms=0.0):
        self._hp.setdefault(pn, Health()).ok(ms)

    def rec_fail(self, pn):
        self._hp.setdefault(pn, Health()).fail()

    def mark_no_tool(self, pn):
        h = self._hp.get(pn)
        if h:
            h.tool_capable = False

    def list_models(self) -> list[str]:
        return sorted(self._map.keys())

    def status(self) -> dict:
        return {
            pn: {
                "healthy": h.healthy,
                "score": round(h.score, 3),
                "tool_capable": h.tool_capable,
                "successes": h.successes,
                "failures": h.failures,
                "avg_latency_ms": round(h.avg_lat, 1),
            }
            for pn, h in sorted(self._hp.items())
        }


reg = Registry()


# ══════════════════════════════════════════════════════════════
#  PYDANTIC
# ══════════════════════════════════════════════════════════════

class FunctionDef(BaseModel):
    name: str
    description: str = ""
    parameters: dict = Field(default_factory=dict)

class ToolDef(BaseModel):
    type: str = "function"
    function: FunctionDef

class MessageIn(BaseModel):
    role: str
    content: Optional[Union[str, list]] = None
    name: Optional[str] = None
    tool_call_id: Optional[str] = None
    tool_calls: Optional[list] = None

class ChatRequest(BaseModel):
    model: str = "auto"
    messages: List[MessageIn]
    tools: Optional[List[ToolDef]] = None
    tool_choice: Optional[Any] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    top_p: Optional[float] = None
    stream: Optional[bool] = False
    model_config = {"extra": "allow"}


# ══════════════════════════════════════════════════════════════
#  UTILITY
# ══════════════════════════════════════════════════════════════

def clean_model(raw: str) -> str:
    return raw.split("/", 1)[-1] if "/" in raw else raw


def _tok(text: str) -> int:
    return max(1, len(text) // 4) if text else 0


def _ptok(msgs: list[dict]) -> int:
    return sum(_tok(m.get("content", "") or "") + 4
               for m in msgs)


# ══════════════════════════════════════════════════════════════
#  CONVERSATION STATE — Loop Detection
# ══════════════════════════════════════════════════════════════

@dataclass
class ConvState:
    trailing: int = 0
    pending: bool = False
    force_text: bool = False


def analyze_conv(messages: List[MessageIn]) -> ConvState:
    cs = ConvState()
    if not messages:
        return cs
    i = len(messages) - 1
    rounds = 0
    while i >= 0:
        if messages[i].role != "tool":
            break
        cs.pending = True
        while i >= 0 and messages[i].role == "tool":
            i -= 1
        if i >= 0 and messages[i].role == "assistant":
            c  = str(messages[i].content or "").strip()
            tc = bool(messages[i].tool_calls)
            if tc or c in ("", "null"):
                rounds += 1; i -= 1; continue
            break
        else:
            break
    cs.trailing = rounds
    cs.force_text = rounds >= MAX_CONSECUTIVE_TOOLS
    return cs


# ══════════════════════════════════════════════════════════════
#  TOOL PROMPT + PREPROCESSOR
# ══════════════════════════════════════════════════════════════

def _tool_prompt(tools: List[ToolDef],
                 tool_choice=None) -> str:
    blocks = []
    for i, t in enumerate(tools, 1):
        fn = t.function
        p  = json.dumps(fn.parameters, ensure_ascii=False)
        blocks.append(
            f"Tool #{i}: {fn.name}\n"
            f"  desc: {fn.description or '-'}\n"
            f"  params: {p}")
    tt = "\n\n".join(blocks)
    ns = [t.function.name for t in tools]

    force = ""
    if isinstance(tool_choice, dict):
        f = tool_choice.get("function", {}).get("name", "")
        force = f"\n⛔ You MUST call '{f}'. No plain text."
    elif tool_choice == "required":
        force = "\n⛔ You MUST call a tool. No plain text."

    return (
        "=== FUNCTION CALLING ===\n"
        f"Tools:\n\n{tt}\n\n"
        f"If you need a tool ({', '.join(ns)}), "
        "respond ONLY with JSON:\n"
        '{"nama_alat":"<name>","argumen":{<params>}}\n'
        "No markdown. No extra text.\n"
        "If just chatting, reply normally.\n"
        f"{force}\n=== END ===")


_POST = ("=== Tool results above. Respond with text. "
         "Do NOT call tools again unless asked. ===")
_STOP = ("=== STOP: You are looping tool calls. "
         "Reply with text NOW. No JSON. No tools. ===")


def preprocess(msgs, tools, tool_choice=None, cs=None):
    result = []
    has   = bool(tools and len(tools) > 0)
    skip  = tool_choice == "none"
    force = cs.force_text if cs else False
    pend  = cs.pending if cs else False

    if has and not skip:
        if force:
            result.append({"role": "system", "content": _STOP})
            log.info(f"  🛑 FORCE TEXT ({cs.trailing} rounds)")
        elif pend:
            result.append({"role": "system",
                           "content": _tool_prompt(tools, tool_choice)})
            result.append({"role": "system", "content": _POST})
            log.info(f"  🔧 Tool + post-result guidance")
        else:
            result.append({"role": "system",
                           "content": _tool_prompt(tools, tool_choice)})
            log.info("  🔧 Tool prompt injected")

    for msg in msgs:
        role = msg.role
        content = msg.content or ""

        if isinstance(content, list):
            parts = []
            for p in content:
                if isinstance(p, dict) and p.get("type") == "text":
                    parts.append(p.get("text", ""))
                elif isinstance(p, str):
                    parts.append(p)
            content = "\n".join(parts)

        if role == "tool":
            nm  = msg.name or "?"
            cid = msg.tool_call_id or "?"
            result.append({"role": "user",
                           "content": f"[Tool '{nm}' (id:{cid})]:\n{content}"})
        elif role == "assistant" and msg.tool_calls:
            ds = []
            for tc in msg.tool_calls:
                fn = tc.get("function", {}) if isinstance(tc, dict) else {}
                ds.append(f"{fn.get('name','?')}({fn.get('arguments','{}')})")
            result.append({"role": "assistant",
                           "content": f"I called: {', '.join(ds)}"})
        elif role == "function":
            result.append({"role": "user",
                           "content": f"[Func '{msg.name}']:{content}"})
        else:
            result.append({"role": role, "content": str(content)})

    return result


# ══════════════════════════════════════════════════════════════
#  JSON EXTRACTOR
# ══════════════════════════════════════════════════════════════

def _strip_md(t):
    return re.sub(r"```(?:json|JSON)?\s*\n?(.*?)\n?\s*```",
                  r"\1", t, flags=re.DOTALL).strip()

def _find_jsons(text):
    res, n, i = [], len(text), 0
    while i < n:
        if text[i] != "{":
            i += 1; continue
        d = ins = esc = 0; j = i
        while j < n:
            c = text[j]
            if esc: esc = 0; j += 1; continue
            if c == "\\" and ins: esc = 1; j += 1; continue
            if c == '"': ins = 1 - ins
            elif not ins:
                if c == "{": d += 1
                elif c == "}":
                    d -= 1
                    if d == 0:
                        try:
                            o = json.loads(text[i:j+1])
                            if isinstance(o, dict): res.append(o)
                        except json.JSONDecodeError: pass
                        break
            j += 1
        i += 1
    return res

_NK = ["nama_alat","name","tool_name","function","function_name"]
_AK = ["argumen","arguments","args","params","parameters"]

def try_parse_tool(raw, tools=None):
    if not raw or not raw.strip():
        return None
    valid = {t.function.name for t in tools} if tools else set()
    cl = _strip_md(raw)
    cands = _find_jsons(cl)
    if not cands:
        try:
            o = json.loads(cl)
            if isinstance(o, dict): cands = [o]
        except: pass
    if not cands:
        return None
    for obj in cands:
        for nk in _NK:
            if nk not in obj: continue
            fn = obj[nk]
            if not isinstance(fn, str): continue
            if valid and fn not in valid: continue
            args = {}
            for ak in _AK:
                if ak in obj: args = obj[ak]; break
            if isinstance(args, str):
                try: args = json.loads(args)
                except: args = {}
            if not isinstance(args, dict): args = {}
            log.info(f"  🎯 Tool: {fn}("
                     f"{json.dumps(args, ensure_ascii=False)})")
            return {"name": fn, "arguments": args}
    return None

def _clean_forced(raw, tools=None):
    tc = try_parse_tool(raw, tools)
    if tc:
        return ("Saya sudah memproses informasi yang tersedia. "
                "Ada yang bisa saya bantu?")
    cl = re.sub(r'^\s*```(?:json)?\s*\{.*?\}\s*```\s*', '',
                raw, flags=re.DOTALL).strip()
    return cl if cl else raw


# ══════════════════════════════════════════════════════════════
#  RESPONSE BUILDERS — Exact OpenAI API Format
# ══════════════════════════════════════════════════════════════
#
#  Dicocokkan persis dengan output real dari OpenAI API
#  agar openai-go library PicoClaw tidak crash.
#
# ══════════════════════════════════════════════════════════════

def resp_chat(model: str, content: str,
              msgs: list[dict] | None = None) -> dict:
    """
    Kondisi A: Chat biasa.

    Contoh output real OpenAI:
    {
      "id": "chatcmpl-...",
      "object": "chat.completion",
      "created": 1711728000,
      "model": "gpt-4o",
      "choices": [{
        "index": 0,
        "message": {
          "role": "assistant",
          "content": "Halo!"
        },
        "finish_reason": "stop",
        "logprobs": null
      }],
      "usage": {...},
      "system_fingerprint": null
    }
    """
    pt = _ptok(msgs) if msgs else 0
    ct = _tok(content)

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:29]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": content,
                    # TIDAK ADA tool_calls di chat biasa
                },
                "finish_reason": "stop",
                "logprobs": None,
            }
        ],
        "usage": {
            "prompt_tokens": pt,
            "completion_tokens": ct,
            "total_tokens": pt + ct,
        },
        "system_fingerprint": None,
    }


def resp_tool(model: str, tool_name: str,
              arguments: dict,
              msgs: list[dict] | None = None) -> dict:
    """
    Kondisi B: Tool call.

    Contoh output real OpenAI:
    {
      "id": "chatcmpl-...",
      "object": "chat.completion",
      "created": 1711728005,
      "model": "gpt-4o",
      "choices": [{
        "index": 0,
        "message": {
          "role": "assistant",
          "content": null,           ← WAJIB null
          "tool_calls": [{           ← WAJIB array
            "id": "call_abc123...",  ← WAJIB ada & unik
            "type": "function",
            "function": {
              "name": "web_search",
              "arguments": "{...}"   ← WAJIB string
            }
          }]
        },
        "finish_reason": "tool_calls",  ← WAJIB
        "logprobs": null
      }],
      "usage": {...},
      "system_fingerprint": null
    }
    """
    # arguments: dict → JSON string
    args_str = json.dumps(arguments, ensure_ascii=False)

    pt = _ptok(msgs) if msgs else 0
    ct = _tok(args_str) + 5

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:29]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": None,            # ← null
                    "tool_calls": [             # ← array
                        {
                            "id": f"call_{uuid.uuid4().hex[:24]}",
                            "type": "function",
                            "function": {
                                "name": tool_name,
                                "arguments": args_str,  # ← string
                            },
                        }
                    ],
                },
                "finish_reason": "tool_calls",  # ← bukan "stop"
                "logprobs": None,
            }
        ],
        "usage": {
            "prompt_tokens": pt,
            "completion_tokens": ct,
            "total_tokens": pt + ct,
        },
        "system_fingerprint": None,
    }


def resp_error(msg, etype="server_error", code=500):
    return {"error": {"message": msg, "type": etype,
                      "param": None, "code": code}}


# ══════════════════════════════════════════════════════════════
#  G4F CALLER — Rate-Limited + Retry + Fallback
# ══════════════════════════════════════════════════════════════

async def _raw(model, messages, provider=None,
               temperature=None, max_tokens=None):
    """Satu kali panggilan g4f (TANPA rate limit — dipanggil dari wrapper)."""
    kw: dict[str, Any] = {"model": model, "messages": messages}
    if provider is not None: kw["provider"] = provider
    if temperature is not None: kw["temperature"] = temperature
    if max_tokens is not None: kw["max_tokens"] = max_tokens

    try:
        r = await g4f.ChatCompletion.create_async(**kw)
        if r: return str(r).strip()
    except (AttributeError, NotImplementedError, TypeError):
        pass

    loop = asyncio.get_running_loop()
    r = await loop.run_in_executor(
        None, lambda: g4f.ChatCompletion.create(**kw))
    if hasattr(r, "__iter__") and not isinstance(r, str):
        return "".join(str(c) for c in r).strip()
    return str(r).strip() if r else ""


async def _retry(model, messages, provider, pname,
                 temperature=None, max_tokens=None):
    """Panggil satu provider dengan retry. RATE LIMITED."""
    last: Exception = RuntimeError("none")
    for att in range(1 + MAX_RETRIES):
        if att > 0:
            w = RETRY_BACKOFF * (2 ** (att - 1))
            log.info(f"       🔄 retry {att}/{MAX_RETRIES} "
                     f"({w:.1f}s)")
            await asyncio.sleep(w)

        # Rate limit sebelum panggilan g4f
        await rate_limiter.acquire(f"call:{pname}")

        try:
            t0 = time.time()
            text = await asyncio.wait_for(
                _raw(model, messages, provider,
                     temperature, max_tokens),
                timeout=G4F_TIMEOUT)
            ms = (time.time() - t0) * 1000
            if not text:
                raise ValueError("empty")
            log.info(f"       ✅ {pname} (att {att+1}, "
                     f"{ms:.0f}ms, {len(text)}ch)")
            return text, ms
        except asyncio.TimeoutError:
            last = TimeoutError("timeout")
            log.warning(f"       ⏱️ {pname} att {att+1}: timeout")
        except Exception as e:
            last = e
            log.warning(f"       ❌ {pname} att {att+1}: {e}")
    raise last


async def call_smart(
    model_raw, messages, tools=None, tool_choice=None,
    temperature=None, max_tokens=None, force_text=False,
) -> tuple[str, str, str]:

    model = clean_model(model_raw)
    need  = bool(tools) and not force_text
    chain = reg.get_chain(model, need)

    # Log rate limiter status
    slots = rate_limiter.available_slots
    wait  = rate_limiter.next_slot_in
    queue = rate_limiter.queue_size

    log.info(f"  🔗 Chain: {len(chain)}" +
             (" (FORCE TEXT)" if force_text else
              " (tool-aware)" if need else ""))
    log.info(f"  🎫 Rate: {slots}/{G4F_RPM} slots free" +
             (f", next in {wait:.0f}s" if wait > 0 else "") +
             (f", {queue} in queue" if queue > 0 else ""))

    for i, (m, pn, _) in enumerate(chain[:5]):
        h   = reg._hp.get(pn, Health())
        tag = " 🔧" if h.tool_capable else ""
        log.info(f"     {i+1}. {pn} ({m}){tag}")
    if len(chain) > 5:
        log.info(f"     … +{len(chain)-5}")

    last_err  = ""
    best_text = ""
    best_mdl  = ""
    best_prov = ""
    ttl       = MAX_TOOL_TRIES if need else 0

    for mdl, pn, pcls in chain:
        log.info(f"  ⏩ [{mdl}] → {pn}")
        try:
            text, ms = await _retry(
                mdl, messages, pcls, pn, temperature, max_tokens)
            reg.rec_ok(pn, ms)

            if force_text:
                return _clean_forced(text, tools), mdl, pn

            if need:
                tc = try_parse_tool(text, tools)
                if tc:
                    log.info(f"  🏆 {pn}: tool call OK")
                    return text, mdl, pn

                log.info(f"  ⚠️  {pn}: plain text")
                if not best_text:
                    best_text = text
                    best_mdl  = mdl
                    best_prov = pn

                h = reg._hp.get(pn)
                if h and h.tool_capable:
                    reg.mark_no_tool(pn)

                ttl -= 1
                if ttl > 0:
                    log.info(f"  🔄 Tool retry ({ttl} left)")
                    continue
                log.info("  ℹ️  Tool retries exhausted")
                return best_text, best_mdl, best_prov

            return text, mdl, pn

        except Exception as e:
            reg.rec_fail(pn)
            last_err = f"{pn}({mdl}): {e}"
            log.warning(f"  ⛔ {last_err}")

    if best_text:
        return best_text, best_mdl, best_prov

    raise HTTPException(503, detail=resp_error(
        f"All failed. Last: {last_err}",
        "all_failed", 503))


# ══════════════════════════════════════════════════════════════
#  FASTAPI APP
# ══════════════════════════════════════════════════════════════

@asynccontextmanager
async def lifespan(app_: FastAPI):
    log.info("🚀 G4F Bridge v7")
    log.info(f"🎫 Rate limit: {G4F_RPM} req / "
             f"{G4F_PERIOD:.0f}s")
    _sep("STARTUP SCAN")
    t0 = time.time()
    await reg.scan_all()
    log.info(f"⏱️  Scan: {time.time()-t0:.1f}s")
    yield
    log.info("👋 Bye.")


app = FastAPI(
    title="G4F Bridge", version="7.0.0",
    description="Rate-Limited + Exact OpenAI + Anti-Loop",
    lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"])


@app.middleware("http")
async def log_http(req: Request, call_next):
    t0 = time.time()
    log.info(f"📡 ← {req.method} {req.url.path}")
    r = await call_next(req)
    ms = (time.time() - t0) * 1000
    log.info(f"📡 → {'✅' if r.status_code < 400 else '❌'} "
             f"{r.status_code} ({ms:.0f}ms)")
    return r


# ══════════════════════════════════════════════════════════════
#  ROUTE: POST /v1/chat/completions
# ══════════════════════════════════════════════════════════════

@app.post("/v1/chat/completions")
@app.post("/chat/completions")
async def chat_completions(req: ChatRequest):
    has_tools = bool(req.tools and len(req.tools) > 0)
    cs = analyze_conv(req.messages)

    rid = f"req-{uuid.uuid4().hex[:10]}"
    _sep(f"REQUEST {rid}")
    mc = clean_model(req.model)

    log.info(f"  📨 model : {req.model} → '{mc}'" +
             (" (AUTO)" if mc.lower() == "auto" else ""))
    log.info(f"  📨 msgs  : {len(req.messages)}")
    log.info(f"  📨 tools : " +
             (f"YA ({len(req.tools)})" if has_tools else "TIDAK"))

    if cs.force_text:
        log.info(f"  🛑 LOOP: {cs.trailing} rounds → FORCE TEXT")
    elif cs.pending:
        log.info(f"  📎 Pending ({cs.trailing} round)")

    _sep("PREPROCESS")
    processed = preprocess(req.messages, req.tools,
                           req.tool_choice, cs)
    log.info(f"  ⚙️  Final: {len(processed)} msgs")

    _sep("G4F CALL")
    log.info(f"  📤 → g4f (model='{mc}', "
             f"force={cs.force_text})")

    try:
        raw, amodel, prov = await call_smart(
            mc, processed,
            tools=req.tools if has_tools else None,
            tool_choice=req.tool_choice,
            temperature=req.temperature,
            max_tokens=req.max_tokens,
            force_text=cs.force_text)
    except HTTPException:
        raise
    except Exception as e:
        log.error(f"  💥 {e}")
        return JSONResponse(502, resp_error(
            f"g4f: {e}", "upstream", 502))

    log.info(f"  📥 ← {prov} ({amodel}, {len(raw)}ch)")
    _log_json("G4F.raw", raw)

    if not raw:
        return JSONResponse(502, resp_error(
            "empty", "empty", 502))

    _sep("PARSE")

    tool_call = None
    if cs.force_text:
        log.info("  🛑 Force text → skip")
        raw = _clean_forced(raw, req.tools)
    elif has_tools:
        log.info("  🔍 Detecting tool…")
        tool_call = try_parse_tool(raw, req.tools)

    if tool_call:
        log.info(f"  ✅ TOOL CALL: {tool_call['name']}")
        response = resp_tool(amodel, tool_call["name"],
                             tool_call["arguments"], processed)
    else:
        log.info(f"  ℹ️  TEXT" +
                 (" (forced)" if cs.force_text else ""))
        response = resp_chat(amodel, raw, processed)

    _sep("RESPONSE")
    fr  = response["choices"][0]["finish_reason"]
    msg = response["choices"][0]["message"]

    log.info(f"  📦 model  = {amodel}")
    log.info(f"  📦 prov   = {prov}")
    log.info(f"  📦 reason = {fr}")
    log.info(f"  📦 usage  = {response['usage']}")

    if fr == "tool_calls":
        cv = msg.get("content")
        tc = msg["tool_calls"][0]
        log.info(f"  📐 content    = {repr(cv)} "
                 f"{'✅' if cv is None else '❌ NOT NULL!'}")
        log.info(f"  📐 tool_calls = ✅ array")
        log.info(f"  📐 call.id    = {tc['id']}")
        log.info(f"  📐 arguments  = "
                 f"{'✅ str' if isinstance(tc['function']['arguments'], str) else '❌'}")
    else:
        has_tc = "tool_calls" in msg
        log.info(f"  📐 tool_calls = "
                 f"{'❌ present!' if has_tc else '✅ absent'}")

    _log_json("FINAL", response)
    _sep()

    return JSONResponse(
        content=response,
        headers={"X-G4F-Provider": prov,
                 "X-G4F-Model": amodel})


# ══════════════════════════════════════════════════════════════
#  OTHER ROUTES
# ══════════════════════════════════════════════════════════════

@app.get("/v1/models")
@app.get("/models")
async def list_models():
    return {
        "object": "list",
        "data": [{"id": m, "object": "model", "created": 0,
                  "owned_by": "g4f"} for m in reg.list_models()]
             + [{"id": "auto", "object": "model", "created": 0,
                 "owned_by": "bridge"}]}


@app.post("/v1/scan")
async def rescan():
    t0 = time.time()
    await reg.scan_all()
    st = reg.status()
    return {
        "duration_s": round(time.time() - t0, 1),
        "alive": sum(1 for v in st.values() if v["successes"] > 0),
        "tool_capable": sum(1 for v in st.values()
                            if v["tool_capable"]),
    }


@app.get("/v1/providers/status")
async def prov_status():
    st = reg.status()
    return {
        "total": len(st),
        "healthy": sum(1 for v in st.values() if v["healthy"]),
        "tool_capable": sum(1 for v in st.values()
                            if v["tool_capable"]),
        "rate_limit": {
            "rpm": G4F_RPM,
            "slots_free": rate_limiter.available_slots,
            "next_slot_in": round(rate_limiter.next_slot_in, 1),
            "queue": rate_limiter.queue_size,
        },
        "providers": st}


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "models": len(reg.list_models()),
        "rate_slots_free": rate_limiter.available_slots,
        "ts": int(time.time())}


@app.get("/")
async def root():
    st = reg.status()
    return {
        "service": "G4F Bridge", "version": "7.0.0",
        "rate_limit": f"{G4F_RPM} req/{G4F_PERIOD:.0f}s",
        "alive": sum(1 for v in st.values() if v["healthy"]),
        "tool_capable": sum(1 for v in st.values()
                            if v["tool_capable"]),
        "endpoints": {
            "chat": "POST /v1/chat/completions",
            "models": "GET /v1/models",
            "scan": "POST /v1/scan",
            "status": "GET /v1/providers/status"}}


# ══════════════════════════════════════════════════════════════
#  ENTRY
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn
    addr = f"http://{HOST}:{PORT}"
    print(f"""
╔════════════════════════════════════════════════════════╗
║            G4F  API  BRIDGE  v7.0                      ║
║   Rate-Limited + Exact OpenAI Format + Anti-Loop       ║
╠════════════════════════════════════════════════════════╣
║  🌐 {addr:<50s}║
║  📖 {(addr + '/docs'):<50s}║
║  🎫 Rate : {(str(G4F_RPM) + ' req/' + str(int(G4F_PERIOD)) + 's'):<44s}║
║  🛑 Loop : {(str(MAX_CONSECUTIVE_TOOLS) + ' rounds'):<44s}║
╚════════════════════════════════════════════════════════╝""")
    uvicorn.run("server:app", host=HOST, port=PORT,
                log_level="warning")