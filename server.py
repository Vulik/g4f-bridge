"""
server.py  v6
═════════════
G4F Bridge — Strict OpenAI JSON + Model-First Scan + Anti-Loop

Perubahan utama dari v5:
  ✦ JSON output 100% identik dengan standar OpenAI API
    - Chat biasa : content=string, TANPA field tool_calls
    - Tool call  : content=null,  DENGAN tool_calls array
    - arguments  = JSON string escaped
    - finish_reason = "stop" | "tool_calls"
    - usage token estimation
  ✦ Scan model-first: discover model → test provider per model
  ✦ Display hasil scan terorganisir per model
  ✦ Anti-loop, tool-aware fallback (tetap dari v4/v5)
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
SCAN_WORKERS          = int(os.getenv("SCAN_WORKERS", "6"))
MAX_RETRIES           = int(os.getenv("MAX_RETRIES", "1"))
RETRY_BACKOFF         = float(os.getenv("RETRY_BACKOFF", "2.0"))
COOLDOWN_BASE         = int(os.getenv("COOLDOWN_BASE", "60"))
COOLDOWN_MAX          = int(os.getenv("COOLDOWN_MAX", "600"))
MAX_TOOL_TRIES        = int(os.getenv("MAX_TOOL_TRIES", "4"))
MAX_CONSECUTIVE_TOOLS = int(os.getenv("MAX_CONSECUTIVE_TOOLS", "2"))
LOG_LEVEL             = os.getenv("LOG_LEVEL", "DEBUG")

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
    datefmt="%Y-%m-%d %H:%M:%S",
))
log.addHandler(_h)
log.propagate = False


def _sep(title: str = "", w: int = 72):
    if title:
        pad = max(1, (w - len(title) - 2) // 2)
        log.info(f"{'─' * pad} {title} {'─' * pad}")
    else:
        log.info("─" * w)


def _log_json(label: str, data: Any, limit: int = 3000):
    try:
        txt = json.dumps(data, ensure_ascii=False, indent=2)
    except (TypeError, ValueError):
        txt = str(data)
    if len(txt) > limit:
        txt = txt[:limit] + f"\n  … [{len(txt)} chars]"
    for line in txt.split("\n"):
        log.debug(f"  {label:>14s} │ {line}")


# ══════════════════════════════════════════════════════════════
#  SCAN TEST PROMPTS
# ══════════════════════════════════════════════════════════════

_ALIVE_TEST = [{"role": "user", "content": "Reply with exactly: ok"}]

_TOOL_TEST = [
    {"role": "system", "content": (
        "You have ONE tool: get_test_info(query: string).\n"
        "You MUST call it. Reply ONLY with this JSON:\n"
        '{"nama_alat": "get_test_info", "argumen": {"query": "<question>"}}\n'
        "NO markdown. ONLY JSON."
    )},
    {"role": "user", "content": "What is Python?"},
]


def _is_tool_response(text: str) -> bool:
    """Validasi apakah respons scan mengandung tool call JSON yang benar."""
    if not text:
        return False
    cleaned = re.sub(r"```(?:json)?\s*\n?(.*?)\n?\s*```",
                     r"\1", text, flags=re.DOTALL).strip()
    try:
        obj = json.loads(cleaned)
        if isinstance(obj, dict):
            for k in ("nama_alat", "name", "tool_name"):
                if obj.get(k) == "get_test_info":
                    return True
    except (json.JSONDecodeError, TypeError):
        pass
    for m in re.finditer(r'\{[^{}]*"(?:nama_alat|name)"[^{}]*\}', cleaned):
        try:
            obj = json.loads(m.group())
            if obj.get("nama_alat") == "get_test_info" or \
               obj.get("name") == "get_test_info":
                return True
        except (json.JSONDecodeError, TypeError):
            pass
    return False


# ══════════════════════════════════════════════════════════════
#  PROVIDER REGISTRY — Model-First Scan
# ══════════════════════════════════════════════════════════════

@dataclass
class HealthRecord:
    successes: int = 0
    failures: int = 0
    consecutive_fails: int = 0
    total_latency_ms: float = 0.0
    last_success: float = 0.0
    last_failure: float = 0.0
    tool_capable: bool = False
    tool_tested: bool = False

    @property
    def is_healthy(self) -> bool:
        if self.consecutive_fails == 0:
            return True
        cd = min(COOLDOWN_MAX, COOLDOWN_BASE * self.consecutive_fails)
        return (time.time() - self.last_failure) > cd

    @property
    def avg_latency(self) -> float:
        return (self.total_latency_ms / self.successes
                ) if self.successes else 9999.0

    @property
    def score(self) -> float:
        total = self.successes + self.failures
        if total == 0:
            return 0.5
        rate  = self.successes / total
        lat   = 1.0 / (1.0 + self.avg_latency / 5000.0)
        fresh = (0.1 if self.last_success
                 and (time.time() - self.last_success) < 300 else 0.0)
        pen   = min(0.5, 0.15 * self.consecutive_fails)
        return rate * 0.5 + lat * 0.3 + fresh - pen

    def record_ok(self, latency_ms: float = 0.0):
        self.successes += 1
        self.total_latency_ms += latency_ms
        self.last_success = time.time()
        self.consecutive_fails = 0

    def record_fail(self):
        self.failures += 1
        self.last_failure = time.time()
        self.consecutive_fails += 1


@dataclass
class ScanResult:
    """Satu hasil tes (model, provider)."""
    model: str
    provider_name: str
    provider_class: Any = field(repr=False, default=None)
    latency_ms: float = 0.0
    alive: bool = False
    tool_ok: bool = False
    error: str = ""


class ProviderRegistry:

    def __init__(self):
        self._model_map: Dict[str, List[Tuple[str, Any]]] = {}
        self._health: Dict[str, HealthRecord] = {}
        self._discover()

    # ── discovery ─────────────────────────────────

    def _discover(self):
        """Discover semua model → [provider, ...] dari g4f."""
        try:
            from g4f.models import ModelUtils
        except ImportError:
            log.warning("⚠️  g4f.models.ModelUtils tidak tersedia")
            return

        for model_name, model_obj in ModelUtils.convert.items():
            provs = self._extract(model_obj)
            if not provs:
                continue
            pairs = []
            for p in provs:
                pn = self._pname(p)
                pairs.append((pn, p))
                if pn not in self._health:
                    self._health[pn] = HealthRecord()
            self._model_map[model_name] = pairs

    @staticmethod
    def _extract(model_obj) -> list:
        bp = getattr(model_obj, "best_provider", None)
        if bp is None:
            return []
        if hasattr(bp, "providers"):
            return [p for p in bp.providers
                    if getattr(p, "working", True)]
        if getattr(bp, "working", True):
            return [bp]
        return []

    @staticmethod
    def _pname(p) -> str:
        return getattr(p, "__name__", type(p).__name__)

    # ══════════════════════════════════════════════
    #  MODEL-FIRST SCAN
    # ══════════════════════════════════════════════

    async def scan_all(self):
        """
        Scan dua fase, hasilnya diorganisir per model:

        Phase 1 — ALIVE: kirim pesan sederhana ke setiap
                  (model, provider), catat siapa yang hidup.

        Phase 2 — TOOL:  dari yang hidup, kirim pesan tool-test,
                  catat siapa yang patuh memanggil tool.
        """
        models   = sorted(self._model_map.keys())
        n_models = len(models)
        n_pairs  = sum(len(v) for v in self._model_map.values())

        # ── Show discovery ──
        _sep("DISCOVERY")
        log.info(f"  📋 {n_models} model ditemukan, {n_pairs} pasangan total\n")

        for mdl in models:
            provs  = self._model_map[mdl]
            pnames = ", ".join(pn for pn, _ in provs)
            if len(pnames) > 55:
                pnames = pnames[:52] + "..."
            log.info(f"  {mdl:30s} │ {len(provs):2d} provider │ {pnames}")

        # ── Phase 1: Alive ──
        _sep("PHASE 1: ALIVE TEST")
        all_pairs = [(m, pn, pc)
                     for m in models
                     for pn, pc in self._model_map[m]]

        log.info(f"  🧪 Testing {len(all_pairs)} pairs "
                 f"({SCAN_WORKERS} concurrent, {SCAN_TIMEOUT}s timeout)\n")

        alive_results = await self._run_tests(all_pairs, _ALIVE_TEST)
        alive_by_model = self._show_results(alive_results)

        # Update health for alive
        for r in alive_results:
            if r.alive:
                self._health.setdefault(
                    r.provider_name, HealthRecord()
                ).record_ok(r.latency_ms)

        alive_pairs = [(r.model, r.provider_name, r.provider_class)
                       for r in alive_results if r.alive]

        total_alive     = len(alive_pairs)
        models_alive    = len(alive_by_model)
        providers_alive = len({pn for _, pn, _ in alive_pairs})

        log.info(f"\n  📊 Alive: {total_alive} pairs, "
                 f"{models_alive} model, {providers_alive} provider")

        if not alive_pairs:
            log.warning("  ⚠️  Tidak ada provider hidup!")
            return

        # ── Phase 2: Tool ──
        _sep("PHASE 2: TOOL CAPABILITY TEST")
        log.info(f"  🔧 Testing {len(alive_pairs)} alive pairs "
                 f"untuk tool capability\n")

        tool_results = await self._run_tests(
            alive_pairs, _TOOL_TEST, validate_tool=True)
        self._show_results(tool_results, show_label="TOOL")

        # Apply tool results
        tool_names: set[str] = set()
        for r in tool_results:
            h = self._health.get(r.provider_name)
            if not h:
                continue
            h.tool_tested = True
            if r.tool_ok:
                h.tool_capable = True
                tool_names.add(r.provider_name)
            else:
                h.tool_capable = False

        # Mark alive but not tool-tested (shouldn't happen, but safety)
        for pn in {pn for _, pn, _ in alive_pairs} - tool_names:
            h = self._health.get(pn)
            if h and not h.tool_tested:
                h.tool_tested = True
                h.tool_capable = False

        # ── Summary ──
        _sep("SCAN COMPLETE")
        log.info(f"  📊 Total pairs   : {n_pairs}")
        log.info(f"  ✅ Alive          : {total_alive}")
        log.info(f"  📋 Models alive   : {models_alive}")
        log.info(f"  👤 Providers alive: {providers_alive}")
        log.info(f"  🔧 Tool-capable   : {len(tool_names)}")
        if tool_names:
            log.info(f"     = {', '.join(sorted(tool_names))}")
        _sep()

    async def _run_tests(
        self,
        pairs: list[tuple[str, str, Any]],
        test_msgs: list[dict],
        validate_tool: bool = False,
    ) -> list[ScanResult]:
        """Tes semua pairs secara concurrent."""
        sem     = asyncio.Semaphore(SCAN_WORKERS)
        results: list[ScanResult] = []

        async def _one(model: str, pname: str, pcls: Any):
            async with sem:
                t0 = time.time()
                try:
                    kw = {"model": model, "messages": test_msgs,
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
                                lambda: g4f.ChatCompletion.create(**kw)
                            ),
                            timeout=SCAN_TIMEOUT)

                    if hasattr(text, "__iter__") and \
                       not isinstance(text, str):
                        text = "".join(str(c) for c in text)
                    text = str(text).strip() if text else ""
                    ms   = (time.time() - t0) * 1000

                    if not text:
                        raise ValueError("empty")

                    tool_ok = (_is_tool_response(text)
                               if validate_tool else False)

                    results.append(ScanResult(
                        model=model, provider_name=pname,
                        provider_class=pcls, latency_ms=ms,
                        alive=True if not validate_tool else tool_ok,
                        tool_ok=tool_ok,
                    ))
                except asyncio.TimeoutError:
                    results.append(ScanResult(
                        model=model, provider_name=pname,
                        provider_class=pcls, alive=False,
                        error="timeout",
                    ))
                except Exception as exc:
                    results.append(ScanResult(
                        model=model, provider_name=pname,
                        provider_class=pcls, alive=False,
                        error=str(exc)[:50],
                    ))

        tasks = [_one(m, pn, pc) for m, pn, pc in pairs]
        await asyncio.gather(*tasks, return_exceptions=True)
        return results

    def _show_results(
        self,
        results: list[ScanResult],
        show_label: str = "",
    ) -> dict[str, list[ScanResult]]:
        """Tampilkan hasil scan terorganisir per model."""
        by_model: dict[str, list[ScanResult]] = {}
        for r in results:
            by_model.setdefault(r.model, []).append(r)

        for model in sorted(by_model.keys()):
            items = by_model[model]
            ok    = sum(1 for r in items if r.alive)
            if ok == 0 and show_label == "":
                continue  # skip model tanpa provider hidup (phase 1)

            log.info(f"  📋 {model} ({ok}/{len(items)})")
            for r in sorted(items,
                            key=lambda x: (-x.alive, x.latency_ms)):
                if r.alive:
                    tag = " 🔧" if r.tool_ok else ""
                    log.info(f"     ✅ {r.provider_name:28s} │ "
                             f"{r.latency_ms:6.0f}ms{tag}")
                else:
                    err = f" │ {r.error}" if r.error else ""
                    log.info(f"     ❌ {r.provider_name:28s}{err}")

        # Return only models with alive providers
        return {m: rs for m, rs in by_model.items()
                if any(r.alive for r in rs)}

    # ── chain building ────────────────────────────

    def get_chain(self, model: str,
                  need_tools: bool = False
                  ) -> list[tuple[str, str, Any]]:
        if model.lower() == "auto":
            return self._auto_chain(need_tools)
        return self._model_chain(model, need_tools)

    def _auto_chain(self, need_tools: bool
                    ) -> list[tuple[str, str, Any]]:
        buckets: dict[str, list[tuple[float, str, str, Any]]] = {
            "tool": [], "other": []
        }

        for mdl in PREFERRED_MODELS:
            pairs = self._model_map.get(mdl, [])
            try:
                idx   = PREFERRED_MODELS.index(mdl)
                bonus = 0.25 * (1.0 - idx / len(PREFERRED_MODELS))
            except ValueError:
                bonus = 0.0
            for pn, pcls in pairs:
                h = self._health.get(pn, HealthRecord())
                if not h.is_healthy:
                    continue
                b = "tool" if h.tool_capable else "other"
                buckets[b].append((h.score + bonus, mdl, pn, pcls))

        for mdl, pairs in self._model_map.items():
            if mdl in PREFERRED_MODELS:
                continue
            for pn, pcls in pairs:
                h = self._health.get(pn, HealthRecord())
                if not h.is_healthy:
                    continue
                b = "tool" if h.tool_capable else "other"
                buckets[b].append((h.score, mdl, pn, pcls))

        for b in buckets.values():
            b.sort(key=lambda x: x[0], reverse=True)

        ordered = ((buckets["tool"] + buckets["other"])
                   if need_tools
                   else sorted(buckets["tool"] + buckets["other"],
                               key=lambda x: x[0], reverse=True))

        chain: list[tuple[str, str, Any]] = []
        seen: set[tuple[str, str]] = set()
        for _, mdl, pn, pcls in ordered:
            key = (mdl, pn)
            if key in seen:
                continue
            seen.add(key)
            chain.append((mdl, pn, pcls))
            if len(chain) >= 15:
                break

        chain.append(("gpt-4o-mini", "g4f-auto", None))

        if chain:
            tc = sum(1 for _, pn, _ in chain
                     if self._health.get(pn, HealthRecord()
                                         ).tool_capable)
            log.info(f"  🎯 Auto: {len(chain)} candidates, "
                     f"{tc} tool-capable, "
                     f"top = {chain[0][1]} ({chain[0][0]})")
        return chain

    def _model_chain(self, model: str, need_tools: bool
                     ) -> list[tuple[str, str, Any]]:
        pairs = self._model_map.get(model)
        if not pairs:
            for key in self._model_map:
                if model in key or key in model:
                    pairs = self._model_map[key]
                    model = key
                    break

        chain: list[tuple[str, str, Any]] = []
        if pairs:
            tool_p, other_p = [], []
            for pn, pcls in pairs:
                h = self._health.get(pn, HealthRecord())
                if not h.is_healthy:
                    continue
                bucket = tool_p if h.tool_capable else other_p
                bucket.append((h.score, model, pn, pcls))

            tool_p.sort(key=lambda x: x[0], reverse=True)
            other_p.sort(key=lambda x: x[0], reverse=True)

            ordered = ((tool_p + other_p) if need_tools
                       else sorted(tool_p + other_p,
                                   key=lambda x: x[0], reverse=True))
            for _, m, pn, pcls in ordered:
                chain.append((m, pn, pcls))

        chain.append((model, "g4f-auto", None))
        return chain

    # ── health ────────────────────────────────────

    def record_success(self, pn, ms=0.0):
        self._health.setdefault(pn, HealthRecord()).record_ok(ms)

    def record_failure(self, pn):
        self._health.setdefault(pn, HealthRecord()).record_fail()

    def mark_tool_failed(self, pn):
        h = self._health.get(pn)
        if h:
            h.tool_capable = False
            log.info(f"  ⚠️  {pn} → not tool-capable (runtime)")

    def list_models(self) -> list[str]:
        return sorted(self._model_map.keys())

    def status(self) -> dict:
        return {
            pn: {
                "healthy": h.is_healthy,
                "score": round(h.score, 3),
                "tool_capable": h.tool_capable,
                "successes": h.successes,
                "failures": h.failures,
                "consecutive_fails": h.consecutive_fails,
                "avg_latency_ms": round(h.avg_latency, 1),
            }
            for pn, h in sorted(self._health.items())
        }


registry = ProviderRegistry()


# ══════════════════════════════════════════════════════════════
#  PYDANTIC — Input Models
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
#  UTILITAS
# ══════════════════════════════════════════════════════════════

def clean_model(raw: str) -> str:
    """'openai/gpt-4o' → 'gpt-4o'"""
    return raw.split("/", 1)[-1] if "/" in raw else raw


def _estimate_tokens(text: str) -> int:
    """Estimasi kasar: ~4 karakter = 1 token."""
    return max(1, len(text) // 4) if text else 0


def _prompt_tokens(messages: list[dict]) -> int:
    """Estimasi total prompt tokens dari semua messages."""
    total = 0
    for m in messages:
        total += _estimate_tokens(m.get("content", "") or "")
        total += 4  # overhead per message
    return total


# ══════════════════════════════════════════════════════════════
#  CONVERSATION STATE — Loop Detection
# ══════════════════════════════════════════════════════════════

@dataclass
class ConvState:
    trailing_rounds: int = 0
    has_pending: bool = False
    force_text: bool = False


def analyze_conv(messages: List[MessageIn]) -> ConvState:
    """
    Hitung berapa ronde tool berturut-turut di akhir conversation.
    Jika >= MAX_CONSECUTIVE_TOOLS → force_text = True.
    """
    st = ConvState()
    if not messages:
        return st

    i = len(messages) - 1
    rounds = 0

    while i >= 0:
        if messages[i].role != "tool":
            break
        st.has_pending = True

        # Skip consecutive tool results
        while i >= 0 and messages[i].role == "tool":
            i -= 1

        # Expect assistant with tool_calls/empty content
        if i >= 0 and messages[i].role == "assistant":
            content = str(messages[i].content or "").strip()
            has_tc  = bool(messages[i].tool_calls)
            if has_tc or content in ("", "null"):
                rounds += 1
                i -= 1
                continue
            break
        else:
            break

    st.trailing_rounds = rounds
    st.force_text = rounds >= MAX_CONSECUTIVE_TOOLS
    return st


# ══════════════════════════════════════════════════════════════
#  TOOL PROMPT + MESSAGE PREPROCESSOR
# ══════════════════════════════════════════════════════════════

def _build_tool_prompt(tools: List[ToolDef],
                       tool_choice: Any = None) -> str:
    blocks = []
    for i, t in enumerate(tools, 1):
        fn = t.function
        p  = json.dumps(fn.parameters, ensure_ascii=False, indent=2)
        blocks.append(
            f"Tool #{i}:\n  name: {fn.name}\n"
            f"  description: {fn.description or '-'}\n"
            f"  parameters: {p}")
    tools_text = "\n\n".join(blocks)
    names = [t.function.name for t in tools]

    force = ""
    if isinstance(tool_choice, dict):
        fn = tool_choice.get("function", {}).get("name", "")
        force = (f"\n\n⛔ You MUST call tool '{fn}'. "
                 f"Do NOT reply with plain text.")
    elif tool_choice == "required":
        force = ("\n\n⛔ You MUST call one tool. "
                 "Do NOT reply with plain text.")

    return (
        "=== FUNCTION CALLING INSTRUCTION ===\n\n"
        f"Available tools:\n\n{tools_text}\n\n"
        "RULES:\n"
        f"1. If you need a tool ({', '.join(names)}), "
        "respond ONLY with one-line JSON:\n"
        '   {"nama_alat": "<name>", "argumen": {<params>}}\n\n'
        f'   Example: {{"nama_alat": "{names[0]}", '
        f'"argumen": {{"key": "value"}}}}\n\n'
        "2. No markdown code blocks.\n"
        "3. No text before/after JSON.\n"
        "4. Only tools from the list.\n"
        "5. If just chatting, reply normally.\n"
        f"{force}\n=== END ===")


_POST_RESULT = (
    "=== IMPORTANT ===\n"
    "Tool results above are ready.\n"
    "Now respond to the user with helpful text.\n"
    "Do NOT call another tool unless explicitly asked.\n"
    "=== END ===")

_FORCE_TEXT = (
    "=== CRITICAL: STOP CALLING TOOLS ===\n"
    "You have been looping tool calls.\n"
    "You MUST respond with natural text NOW.\n"
    "Do NOT output JSON. Do NOT call tools.\n"
    "=== END ===")


def preprocess(
    messages: List[MessageIn],
    tools: Optional[List[ToolDef]],
    tool_choice: Any = None,
    cs: Optional[ConvState] = None,
) -> list[dict[str, str]]:
    """Siapkan messages untuk g4f."""
    result: list[dict[str, str]] = []

    has_tools  = bool(tools and len(tools) > 0)
    skip       = tool_choice == "none"
    force_text = cs.force_text if cs else False
    pending    = cs.has_pending if cs else False
    rounds     = cs.trailing_rounds if cs else 0

    if has_tools and not skip:
        if force_text:
            result.append({"role": "system", "content": _FORCE_TEXT})
            log.info(f"  🛑 FORCE TEXT ({rounds} rounds)")
        elif pending:
            result.append({"role": "system",
                           "content": _build_tool_prompt(tools, tool_choice)})
            result.append({"role": "system", "content": _POST_RESULT})
            log.info(f"  🔧 Tool prompt + post-result ({rounds} round(s))")
        else:
            result.append({"role": "system",
                           "content": _build_tool_prompt(tools, tool_choice)})
            log.info("  🔧 Tool prompt injected")

    for msg in messages:
        role    = msg.role
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
            result.append({
                "role": "user",
                "content": f"[Tool result '{nm}' (id:{cid})]:\n{content}",
            })
        elif role == "assistant" and msg.tool_calls:
            descs = []
            for tc in msg.tool_calls:
                fn = (tc.get("function", {})
                      if isinstance(tc, dict) else {})
                descs.append(
                    f"{fn.get('name','?')}({fn.get('arguments','{}')})")
            result.append({
                "role": "assistant",
                "content": f"I called: {', '.join(descs)}",
            })
        elif role == "function":
            result.append({
                "role": "user",
                "content": f"[Function '{msg.name or '?'}']: {content}",
            })
        else:
            result.append({"role": role, "content": str(content)})

    return result


# ══════════════════════════════════════════════════════════════
#  JSON EXTRACTOR — Deteksi Tool Call dari Teks g4f
# ══════════════════════════════════════════════════════════════

def _strip_md(text: str) -> str:
    return re.sub(r"```(?:json|JSON)?\s*\n?(.*?)\n?\s*```",
                  r"\1", text, flags=re.DOTALL).strip()


def _find_jsons(text: str) -> list[dict]:
    results = []
    n, i = len(text), 0
    while i < n:
        if text[i] != "{":
            i += 1; continue
        depth = in_str = esc = 0
        j = i
        while j < n:
            ch = text[j]
            if esc:
                esc = 0; j += 1; continue
            if ch == "\\" and in_str:
                esc = 1; j += 1; continue
            if ch == '"':
                in_str = 1 - in_str
            elif not in_str:
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        try:
                            obj = json.loads(text[i:j + 1])
                            if isinstance(obj, dict):
                                results.append(obj)
                        except json.JSONDecodeError:
                            pass
                        break
            j += 1
        i += 1
    return results


_NK = ["nama_alat", "name", "tool_name", "function", "function_name"]
_AK = ["argumen", "arguments", "args", "params", "parameters"]


def try_parse_tool(
    raw: str,
    tools: Optional[List[ToolDef]] = None,
) -> Optional[Dict[str, Any]]:
    """
    Coba deteksi tool call dari teks g4f.
    Return {"name": ..., "arguments": {...}} atau None.
    """
    if not raw or not raw.strip():
        return None

    valid = {t.function.name for t in tools} if tools else set()
    cleaned    = _strip_md(raw)
    candidates = _find_jsons(cleaned)

    if not candidates:
        try:
            obj = json.loads(cleaned)
            if isinstance(obj, dict):
                candidates = [obj]
        except (json.JSONDecodeError, TypeError):
            pass

    if not candidates:
        return None

    for obj in candidates:
        for nk in _NK:
            if nk not in obj:
                continue
            fn = obj[nk]
            if not isinstance(fn, str):
                continue
            if valid and fn not in valid:
                continue

            args = {}
            for ak in _AK:
                if ak in obj:
                    args = obj[ak]; break
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {}
            if not isinstance(args, dict):
                args = {}

            log.info(f"  🎯 Tool: {fn}("
                     f"{json.dumps(args, ensure_ascii=False)})")
            return {"name": fn, "arguments": args}

    return None


def _clean_forced(raw: str,
                  tools: Optional[List[ToolDef]] = None) -> str:
    """Bersihkan respons force-text dari JSON artifacts."""
    tc = try_parse_tool(raw, tools)
    if tc:
        log.warning("  🛑 Model masih output tool di force-text")
        return ("Saya sudah memproses informasi yang tersedia. "
                "Silakan sampaikan pertanyaan Anda.")
    cleaned = re.sub(r'^\s*```(?:json)?\s*\{.*?\}\s*```\s*', '',
                     raw, flags=re.DOTALL).strip()
    return cleaned if cleaned else raw


# ══════════════════════════════════════════════════════════════
#  RESPONSE BUILDERS — 100% Standar OpenAI API
# ══════════════════════════════════════════════════════════════
#
#  Format ini harus IDENTIK dengan output asli OpenAI
#  agar PicoClaw (openai-go library) tidak crash.
#
#  KONDISI A (chat biasa):
#    message = {role, content}          ← TANPA tool_calls
#    finish_reason = "stop"
#
#  KONDISI B (tool call):
#    message = {role, content:null, tool_calls:[...]}
#    finish_reason = "tool_calls"
#    arguments = JSON STRING escaped
#
# ══════════════════════════════════════════════════════════════

def resp_chat(
    model: str,
    content: str,
    messages: list[dict] | None = None,
) -> dict:
    """
    Kondisi A: Respons teks biasa.

    ┌────────────────────────────────────────┐
    │  message:                              │
    │    role: "assistant"                   │
    │    content: "teks jawaban"             │
    │    (TANPA field tool_calls)            │
    │                                        │
    │  finish_reason: "stop"                 │
    └────────────────────────────────────────┘
    """
    pt = _prompt_tokens(messages) if messages else 0
    ct = _estimate_tokens(content)

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": content,
                    # ← TIDAK ADA "tool_calls" di sini
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": pt,
            "completion_tokens": ct,
            "total_tokens": pt + ct,
        },
    }


def resp_tool(
    model: str,
    tool_name: str,
    arguments: dict,
    messages: list[dict] | None = None,
) -> dict:
    """
    Kondisi B: Respons tool/function call.

    ┌─────────────────────────────────────────────────┐
    │  message:                                       │
    │    role: "assistant"                             │
    │    content: null         ← WAJIB null, bukan "" │
    │    tool_calls: [         ← WAJIB array          │
    │      {                                          │
    │        id: "call_..."    ← WAJIB unik           │
    │        type: "function"                         │
    │        function:                                │
    │          name: "web_search"                     │
    │          arguments: "{\"q\":\"x\"}"  ← STRING!  │
    │      }                                          │
    │    ]                                            │
    │                                                 │
    │  finish_reason: "tool_calls"  ← BUKAN "stop"   │
    └─────────────────────────────────────────────────┘
    """
    #  arguments → JSON string yang di-escape
    #  dict {"query":"kucing"} → string '{"query":"kucing"}'
    #  saat JSONResponse serialize lagi → "{\"query\":\"kucing\"}"
    args_string = json.dumps(arguments, ensure_ascii=False)

    pt = _prompt_tokens(messages) if messages else 0
    ct = _estimate_tokens(args_string) + 5

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": None,    # ← null di JSON, BUKAN ""
                    "tool_calls": [     # ← WAJIB array
                        {
                            "id": f"call_{uuid.uuid4().hex[:24]}",
                            "type": "function",
                            "function": {
                                "name": tool_name,
                                "arguments": args_string,
                            },
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {
            "prompt_tokens": pt,
            "completion_tokens": ct,
            "total_tokens": pt + ct,
        },
    }


def resp_error(msg: str, etype: str = "server_error",
               code: int = 500) -> dict:
    return {"error": {"message": msg, "type": etype,
                      "param": None, "code": code}}


# ══════════════════════════════════════════════════════════════
#  G4F CALLER — Retry + Fallback + Tool-Aware
# ══════════════════════════════════════════════════════════════

async def _raw_call(model, messages, provider=None,
                    temperature=None, max_tokens=None) -> str:
    kw: dict[str, Any] = {"model": model, "messages": messages}
    if provider is not None:
        kw["provider"] = provider
    if temperature is not None:
        kw["temperature"] = temperature
    if max_tokens is not None:
        kw["max_tokens"] = max_tokens

    try:
        r = await g4f.ChatCompletion.create_async(**kw)
        if r:
            return str(r).strip()
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
    last: Exception = RuntimeError("none")
    for attempt in range(1 + MAX_RETRIES):
        if attempt > 0:
            w = RETRY_BACKOFF * (2 ** (attempt - 1))
            log.info(f"       🔄 retry {attempt}/{MAX_RETRIES} "
                     f"({w:.1f}s)")
            await asyncio.sleep(w)
        try:
            t0 = time.time()
            text = await asyncio.wait_for(
                _raw_call(model, messages, provider,
                          temperature, max_tokens),
                timeout=G4F_TIMEOUT)
            ms = (time.time() - t0) * 1000
            if not text:
                raise ValueError("empty")
            log.info(f"       ✅ {pname} (attempt {attempt + 1}, "
                     f"{ms:.0f}ms, {len(text)}ch)")
            return text, ms
        except asyncio.TimeoutError:
            last = TimeoutError(f"timeout")
            log.warning(f"       ⏱️ {pname} attempt "
                        f"{attempt + 1}: timeout")
        except Exception as e:
            last = e
            log.warning(f"       ❌ {pname} attempt "
                        f"{attempt + 1}: {e}")
    raise last


async def call_smart(
    model_raw: str,
    messages: list[dict],
    tools: list[ToolDef] | None = None,
    tool_choice: Any = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    force_text: bool = False,
) -> tuple[str, str, str]:
    """
    Returns (response_text, actual_model, provider_name).
    Raises HTTPException 503 jika semua gagal.
    """
    model      = clean_model(model_raw)
    need_tools = bool(tools) and not force_text
    chain      = registry.get_chain(model, need_tools)

    log.info(f"  🔗 Chain: {len(chain)}" +
             (" (FORCE TEXT)" if force_text else
              " (tool-aware)" if need_tools else ""))
    for i, (m, pn, _) in enumerate(chain[:5]):
        h   = registry._health.get(pn, HealthRecord())
        tag = " 🔧" if h.tool_capable else ""
        log.info(f"     {i + 1}. {pn} ({m}){tag}")
    if len(chain) > 5:
        log.info(f"     … +{len(chain) - 5} more")

    last_err        = ""
    best_text       = ""
    best_model      = ""
    best_prov       = ""
    tool_tries_left = MAX_TOOL_TRIES if need_tools else 0

    for mdl, pn, pcls in chain:
        log.info(f"  ⏩ [{mdl}] → {pn}")
        try:
            text, ms = await _retry(
                mdl, messages, pcls, pn, temperature, max_tokens)
            registry.record_success(pn, ms)

            if force_text:
                return _clean_forced(text, tools), mdl, pn

            if need_tools:
                tc = try_parse_tool(text, tools)
                if tc:
                    log.info(f"  🏆 {pn}: valid tool call")
                    return text, mdl, pn

                log.info(f"  ⚠️  {pn}: plain text (no tool)")
                if not best_text:
                    best_text  = text
                    best_model = mdl
                    best_prov  = pn

                h = registry._health.get(pn)
                if h and h.tool_capable:
                    registry.mark_tool_failed(pn)

                tool_tries_left -= 1
                if tool_tries_left > 0:
                    log.info(f"  🔄 Tool retry "
                             f"({tool_tries_left} left)")
                    continue
                log.info("  ℹ️  Tool retries exhausted")
                return best_text, best_model, best_prov

            return text, mdl, pn

        except Exception as e:
            registry.record_failure(pn)
            last_err = f"{pn}({mdl}): {e}"
            log.warning(f"  ⛔ {last_err}")

    if best_text:
        return best_text, best_model, best_prov

    raise HTTPException(503, detail=resp_error(
        f"All providers failed. Last: {last_err}",
        "all_providers_failed", 503))


# ══════════════════════════════════════════════════════════════
#  FASTAPI APP
# ══════════════════════════════════════════════════════════════

@asynccontextmanager
async def lifespan(app_: FastAPI):
    log.info("🚀 G4F Bridge v6 starting...")
    _sep("STARTUP SCAN")
    t0 = time.time()
    await registry.scan_all()
    log.info(f"⏱️  Total scan: {time.time() - t0:.1f}s")
    yield
    log.info("👋 Bye.")


app = FastAPI(
    title="G4F Bridge", version="6.0.0",
    description="Strict OpenAI JSON + Model-First Scan + Anti-Loop",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)


@app.middleware("http")
async def log_http(request: Request, call_next):
    t0 = time.time()
    log.info(f"📡 ← {request.method} {request.url.path}")
    resp = await call_next(request)
    ms   = (time.time() - t0) * 1000
    e    = "✅" if resp.status_code < 400 else "❌"
    log.info(f"📡 → {e} {resp.status_code} ({ms:.0f}ms)")
    return resp


# ══════════════════════════════════════════════════════════════
#  ROUTE: POST /v1/chat/completions
# ══════════════════════════════════════════════════════════════

@app.post("/v1/chat/completions")
@app.post("/chat/completions")
async def chat_completions(req: ChatRequest):

    has_tools = bool(req.tools and len(req.tools) > 0)
    cs        = analyze_conv(req.messages)

    # ═══ LOG ═══
    rid = f"req-{uuid.uuid4().hex[:12]}"
    _sep(f"REQUEST {rid}")
    mc = clean_model(req.model)

    log.info(f"  📨 model       : {req.model} → '{mc}'" +
             (" (AUTO)" if mc.lower() == "auto" else ""))
    log.info(f"  📨 messages    : {len(req.messages)}")
    log.info(f"  📨 tools       : " +
             (f"YA ({len(req.tools)})" if has_tools else "TIDAK"))
    log.info(f"  📨 tool_choice : {req.tool_choice}")

    if cs.force_text:
        log.info(f"  🛑 LOOP: {cs.trailing_rounds} rounds → FORCE TEXT")
    elif cs.has_pending:
        log.info(f"  📎 Pending results ({cs.trailing_rounds} round)")
    else:
        log.info("  ℹ️  Fresh conversation")

    if has_tools:
        for t in req.tools:
            log.info(f"     🔧 {t.function.name}")

    # ═══ PREPROCESS ═══
    _sep("PREPROCESS")
    processed = preprocess(req.messages, req.tools,
                           req.tool_choice, cs)
    log.info(f"  ⚙️  Final messages: {len(processed)}")

    # ═══ CALL G4F ═══
    _sep("G4F CALL")
    log.info(f"  📤 → g4f (model='{mc}', "
             f"force_text={cs.force_text})")

    try:
        raw, actual_model, prov = await call_smart(
            model_raw=mc, messages=processed,
            tools=req.tools if has_tools else None,
            tool_choice=req.tool_choice,
            temperature=req.temperature,
            max_tokens=req.max_tokens,
            force_text=cs.force_text)
    except HTTPException:
        raise
    except Exception as e:
        log.error(f"  💥 {type(e).__name__}: {e}")
        return JSONResponse(502, resp_error(
            f"g4f: {e}", "upstream_error", 502))

    log.info(f"  📥 ← {prov} ({actual_model}, {len(raw)}ch)")
    _log_json("G4F.raw", raw)

    if not raw:
        return JSONResponse(502, resp_error(
            "empty response", "empty", 502))

    # ═══ PARSE ═══
    _sep("PARSE")

    tool_call = None
    if cs.force_text:
        log.info("  🛑 Force text → skip tool parse")
        raw = _clean_forced(raw, req.tools)
    elif has_tools:
        log.info("  🔍 Detecting tool call…")
        tool_call = try_parse_tool(raw, req.tools)

    # ═══ BUILD RESPONSE ═══
    if tool_call:
        log.info(f"  ✅ KONDISI B: Tool Call")
        log.info(f"     name = {tool_call['name']}")
        log.info(f"     args = {json.dumps(tool_call['arguments'], ensure_ascii=False)}")
        response = resp_tool(
            actual_model,
            tool_call["name"],
            tool_call["arguments"],
            processed,
        )
    else:
        label = ("FORCED" if cs.force_text else
                 "tools ada, tidak dipanggil" if has_tools else "")
        log.info(f"  ℹ️  KONDISI A: Teks biasa"
                 + (f" ({label})" if label else ""))
        response = resp_chat(actual_model, raw, processed)

    # ═══ LOG & RETURN ═══
    _sep("RESPONSE → CLIENT")
    fr  = response["choices"][0]["finish_reason"]
    msg = response["choices"][0]["message"]

    log.info(f"  📦 model         = {actual_model}")
    log.info(f"  📦 provider      = {prov}")
    log.info(f"  📦 finish_reason = {fr}")
    log.info(f"  📦 usage         = {response['usage']}")

    # Verifikasi format kritis
    if fr == "tool_calls":
        cv = msg.get("content")
        log.info(f"  📐 content       = {repr(cv)}"
                 f"  ({'✅ null' if cv is None else '❌ NOT null!'})")
        log.info(f"  📐 tool_calls    = "
                 f"{'✅ array' if isinstance(msg.get('tool_calls'), list) else '❌'}")
        tc = msg["tool_calls"][0]
        log.info(f"  📐 call.id       = {tc['id']}")
        log.info(f"  📐 arguments     = "
                 f"{'✅ string' if isinstance(tc['function']['arguments'], str) else '❌ NOT string!'}")
    else:
        has_tc = "tool_calls" in msg
        log.info(f"  📐 tool_calls    = "
                 f"{'❌ PRESENT (bug!)' if has_tc else '✅ absent'}")

    _log_json("RESPONSE", response)
    _sep()

    return JSONResponse(
        content=response,
        headers={"X-G4F-Provider": prov,
                 "X-G4F-Model": actual_model})


# ══════════════════════════════════════════════════════════════
#  OTHER ROUTES
# ══════════════════════════════════════════════════════════════

@app.get("/v1/models")
@app.get("/models")
async def list_models():
    return {
        "object": "list",
        "data": [{"id": m, "object": "model", "created": 0,
                  "owned_by": "g4f"} for m in registry.list_models()]
             + [{"id": "auto", "object": "model", "created": 0,
                 "owned_by": "bridge"}],
    }


@app.post("/v1/scan")
async def rescan():
    t0 = time.time()
    await registry.scan_all()
    st = registry.status()
    return {
        "duration_s": round(time.time() - t0, 1),
        "alive": sum(1 for v in st.values() if v["successes"] > 0),
        "tool_capable": sum(1 for v in st.values()
                            if v["tool_capable"]),
        "models": len(registry.list_models()),
    }


@app.get("/v1/providers/status")
async def prov_status():
    st = registry.status()
    return {
        "total": len(st),
        "healthy": sum(1 for v in st.values() if v["healthy"]),
        "tool_capable": sum(1 for v in st.values()
                            if v["tool_capable"]),
        "providers": st,
    }


@app.get("/health")
async def health():
    return {"status": "ok",
            "models": len(registry.list_models()),
            "ts": int(time.time())}


@app.get("/")
async def root():
    st = registry.status()
    return {
        "service": "G4F Bridge", "version": "6.0.0",
        "alive": sum(1 for v in st.values() if v["healthy"]),
        "tool_capable": sum(1 for v in st.values()
                            if v["tool_capable"]),
        "endpoints": {
            "chat": "POST /v1/chat/completions",
            "models": "GET /v1/models",
            "scan": "POST /v1/scan",
            "status": "GET /v1/providers/status",
        },
    }


# ══════════════════════════════════════════════════════════════
#  ENTRY
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn

    addr = f"http://{HOST}:{PORT}"
    print(f"""
╔════════════════════════════════════════════════════╗
║           G4F  API  BRIDGE  v6.0                   ║
║  Strict OpenAI JSON + Model-First Scan + Anti-Loop ║
╠════════════════════════════════════════════════════╣
║  🌐 {addr:<46s}║
║  📖 {(addr + '/docs'):<46s}║
╚════════════════════════════════════════════════════╝""")

    uvicorn.run("server:app", host=HOST, port=PORT,
                log_level="warning")