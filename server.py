"""
server.py  v3
═════════════
G4F Bridge — Auto-Model + Tool-Scan + Tool-Aware Fallback

Fix dari v2:
  ✦ Double logging fixed (handler guard)
  ✦ Startup scan: tes setiap provider DENGAN tool → tandai tool_capable
  ✦ Runtime: request dgn tools → prioritaskan tool_capable provider
  ✦ Tool-retry: jika provider tidak panggil tool, coba provider lain
  ✦ System prompt diperkuat agar model lebih patuh
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

# ══════════════════════════════════════════════════════════════
#  DEPENDENCY CHECK
# ══════════════════════════════════════════════════════════════

try:
    import g4f
except ImportError:
    sys.exit("\n  ❌ pip install g4f\n")

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field


# ══════════════════════════════════════════════════════════════
#  KONFIGURASI
# ══════════════════════════════════════════════════════════════

HOST           = os.getenv("HOST", "0.0.0.0")
PORT           = int(os.getenv("PORT", "8820"))
G4F_TIMEOUT    = int(os.getenv("G4F_TIMEOUT", "120"))
SCAN_TIMEOUT   = int(os.getenv("SCAN_TIMEOUT", "25"))
SCAN_WORKERS   = int(os.getenv("SCAN_WORKERS", "6"))
MAX_RETRIES    = int(os.getenv("MAX_RETRIES", "1"))
RETRY_BACKOFF  = float(os.getenv("RETRY_BACKOFF", "2.0"))
COOLDOWN_BASE  = int(os.getenv("COOLDOWN_BASE", "60"))
COOLDOWN_MAX   = int(os.getenv("COOLDOWN_MAX", "600"))
MAX_TOOL_TRIES = int(os.getenv("MAX_TOOL_TRIES", "4"))
LOG_LEVEL      = os.getenv("LOG_LEVEL", "DEBUG")

PREFERRED_MODELS: list[str] = [
    "gpt-4o-mini", "gpt-4o", "gpt-4", "gpt-3.5-turbo",
    "claude-3.5-sonnet", "claude-3-haiku",
    "llama-3.1-70b", "llama-3.1-8b", "mixtral-8x7b",
]


# ══════════════════════════════════════════════════════════════
#  LOGGING  — FIX: guard against duplicate handlers
# ══════════════════════════════════════════════════════════════

log = logging.getLogger("bridge")
log.handlers.clear()                       # ← FIX: hapus handler lama
log.setLevel(getattr(logging, LOG_LEVEL, logging.DEBUG))

_h = logging.StreamHandler(sys.stdout)
_h.setFormatter(logging.Formatter(
    "%(asctime)s │ %(levelname)-7s │ %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
))
log.addHandler(_h)
log.propagate = False                      # ← jangan naik ke root logger


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
#  TOOL SCAN — Prompt & Validator untuk tes tool capability
# ══════════════════════════════════════════════════════════════

_TOOL_TEST_SYSTEM = (
    "You have ONE tool: get_test_info(query: string).\n"
    "You MUST call it for ANY question.\n"
    "Reply ONLY with this exact JSON, nothing else:\n"
    '{"nama_alat": "get_test_info", "argumen": {"query": "<user question>"}}\n'
    "NO markdown. NO explanation. ONLY the JSON."
)

_TOOL_TEST_USER = "What is Python?"

_TOOL_TEST_MESSAGES: list[dict] = [
    {"role": "system", "content": _TOOL_TEST_SYSTEM},
    {"role": "user",   "content": _TOOL_TEST_USER},
]

_BASIC_TEST_MESSAGES: list[dict] = [
    {"role": "user", "content": "Reply with exactly: ok"},
]


def _is_valid_tool_response(text: str) -> bool:
    """Cek apakah respons mengandung JSON tool call yang valid."""
    if not text:
        return False
    cleaned = re.sub(r"```(?:json)?\s*\n?(.*?)\n?\s*```", r"\1",
                     text, flags=re.DOTALL).strip()
    try:
        obj = json.loads(cleaned)
        if isinstance(obj, dict):
            for nk in ("nama_alat", "name", "tool_name", "function"):
                if nk in obj and obj[nk] == "get_test_info":
                    return True
    except (json.JSONDecodeError, TypeError):
        pass

    # brace-match fallback
    for match in re.finditer(r'\{[^{}]*"(?:nama_alat|name)"[^{}]*\}', cleaned):
        try:
            obj = json.loads(match.group())
            if obj.get("nama_alat") == "get_test_info" or obj.get("name") == "get_test_info":
                return True
        except (json.JSONDecodeError, TypeError):
            pass

    return False


# ══════════════════════════════════════════════════════════════
#  PROVIDER REGISTRY — Discovery + Health + Tool Capability
# ══════════════════════════════════════════════════════════════

@dataclass
class HealthRecord:
    successes: int = 0
    failures: int = 0
    consecutive_fails: int = 0
    total_latency_ms: float = 0.0
    last_success: float = 0.0
    last_failure: float = 0.0
    tool_capable: bool = False          # ← BARU: apakah bisa tool call
    tool_tested: bool = False           # ← BARU: sudah di-scan tool?

    @property
    def is_healthy(self) -> bool:
        if self.consecutive_fails == 0:
            return True
        cd = min(COOLDOWN_MAX, COOLDOWN_BASE * self.consecutive_fails)
        return (time.time() - self.last_failure) > cd

    @property
    def avg_latency(self) -> float:
        return (self.total_latency_ms / self.successes) if self.successes else 9999.0

    @property
    def score(self) -> float:
        total = self.successes + self.failures
        if total == 0:
            return 0.5
        rate = self.successes / total
        lat  = 1.0 / (1.0 + self.avg_latency / 5000.0)
        fresh = 0.1 if self.last_success and (time.time() - self.last_success) < 300 else 0.0
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


class ProviderRegistry:

    def __init__(self):
        self._model_map: Dict[str, List[Tuple[str, Any]]] = {}
        self._health: Dict[str, HealthRecord] = {}
        self._scan_done = False
        self._discover()

    # ── discovery ───────────────────────────────────

    def _discover(self):
        try:
            from g4f.models import ModelUtils
        except ImportError:
            log.warning("⚠️  g4f.models.ModelUtils tidak tersedia")
            return

        count = 0
        for model_name, model_obj in ModelUtils.convert.items():
            providers = self._extract(model_obj)
            if not providers:
                continue
            pairs = []
            for p in providers:
                pn = self._pname(p)
                pairs.append((pn, p))
                if pn not in self._health:
                    self._health[pn] = HealthRecord()
                count += 1
            self._model_map[model_name] = pairs

        log.info(
            f"📚 Registry: {len(self._model_map)} model, "
            f"{len(self._health)} provider, {count} pasangan"
        )

    @staticmethod
    def _extract(model_obj) -> list:
        bp = getattr(model_obj, "best_provider", None)
        if bp is None:
            return []
        if hasattr(bp, "providers"):
            return [p for p in bp.providers if getattr(p, "working", True)]
        if getattr(bp, "working", True):
            return [bp]
        return []

    @staticmethod
    def _pname(p) -> str:
        return getattr(p, "__name__", type(p).__name__)

    # ══════════════════════════════════════════════
    #  STARTUP SCAN — Tes hidup + tes tool
    # ══════════════════════════════════════════════

    async def scan_all(self):
        """
        Scan semua provider:
          Phase 1: Basic alive test  (kirim pesan simple)
          Phase 2: Tool test         (kirim pesan + tool instruction)
        """
        all_pairs = self._all_pairs()
        if not all_pairs:
            log.warning("⚠️  Tidak ada pasangan untuk di-scan")
            return

        _sep("PHASE 1: ALIVE SCAN")
        log.info(f"  🧪 Menguji {len(all_pairs)} pasangan (max {SCAN_WORKERS} parallel)")
        alive = await self._scan_phase(all_pairs, _BASIC_TEST_MESSAGES, "ALIVE")

        _sep("PHASE 2: TOOL CAPABILITY SCAN")
        if not alive:
            log.warning("  ⚠️  Tidak ada provider hidup, skip tool scan")
            self._scan_done = True
            return

        log.info(f"  🔧 Menguji {len(alive)} provider hidup untuk tool capability")
        tool_ok = await self._scan_phase(alive, _TOOL_TEST_MESSAGES, "TOOL")

        # Tandai tool_capable
        tool_names = set()
        for model, pname, pcls, _ in tool_ok:
            h = self._health.get(pname)
            if h:
                h.tool_capable = True
                h.tool_tested = True
                tool_names.add(pname)

        # Tandai yang tested tapi gagal tool
        alive_names = {pname for _, pname, _, _ in alive}
        for pname in alive_names - tool_names:
            h = self._health.get(pname)
            if h:
                h.tool_tested = True
                h.tool_capable = False

        # Summary
        _sep("SCAN COMPLETE")
        log.info(f"  📊 Total pasangan  : {len(all_pairs)}")
        log.info(f"  ✅ Hidup           : {len(alive)}")
        log.info(f"  🔧 Tool-capable    : {len(tool_ok)}")
        for model, pname, _, lat in tool_ok:
            log.info(f"     🏆 {pname:28s} ({model}) — {lat:.0f}ms")
        _sep()
        self._scan_done = True

    def _all_pairs(self) -> list[tuple[str, str, Any]]:
        pairs = []
        seen: set[tuple[str, str]] = set()
        for model, provs in self._model_map.items():
            for pname, pcls in provs:
                key = (model, pname)
                if key not in seen:
                    seen.add(key)
                    pairs.append((model, pname, pcls))
        return pairs

    async def _scan_phase(
        self,
        pairs: list,
        test_messages: list[dict],
        phase: str,
    ) -> list[tuple[str, str, Any, float]]:
        """
        Tes setiap pasangan dengan test_messages.
        Return: [(model, pname, pcls, latency_ms), ...] yang berhasil.
        """
        sem = asyncio.Semaphore(SCAN_WORKERS)
        results: list[tuple[str, str, Any, float]] = []

        async def _test(model: str, pname: str, pcls: Any):
            async with sem:
                t0 = time.time()
                try:
                    kw: dict[str, Any] = {
                        "model": model,
                        "messages": test_messages,
                        "provider": pcls,
                    }
                    try:
                        text = await asyncio.wait_for(
                            g4f.ChatCompletion.create_async(**kw),
                            timeout=SCAN_TIMEOUT,
                        )
                    except (AttributeError, NotImplementedError, TypeError):
                        loop = asyncio.get_running_loop()
                        text = await asyncio.wait_for(
                            loop.run_in_executor(
                                None,
                                lambda: g4f.ChatCompletion.create(**kw),
                            ),
                            timeout=SCAN_TIMEOUT,
                        )

                    if hasattr(text, "__iter__") and not isinstance(text, str):
                        text = "".join(str(c) for c in text)
                    text = str(text).strip() if text else ""

                    ms = (time.time() - t0) * 1000

                    if not text:
                        raise ValueError("empty")

                    # Untuk TOOL phase: validasi apakah JSON tool call
                    if phase == "TOOL":
                        if _is_valid_tool_response(text):
                            log.info(f"  ✅ {phase:5s} │ {pname:28s} │ {model:24s} │ {ms:6.0f}ms │ 🔧 tool OK")
                            results.append((model, pname, pcls, ms))
                        else:
                            log.debug(f"  ❌ {phase:5s} │ {pname:28s} │ {model:24s} │ {ms:6.0f}ms │ no tool JSON")
                    else:
                        log.info(f"  ✅ {phase:5s} │ {pname:28s} │ {model:24s} │ {ms:6.0f}ms")
                        results.append((model, pname, pcls, ms))
                        # record health
                        h = self._health.get(pname)
                        if h:
                            h.record_ok(ms)

                except asyncio.TimeoutError:
                    log.debug(f"  ⏱️ {phase:5s} │ {pname:28s} │ {model:24s} │ timeout")
                except Exception as exc:
                    log.debug(f"  ❌ {phase:5s} │ {pname:28s} │ {model:24s} │ {str(exc)[:60]}")

        # Handle 3-tuple or 4-tuple input
        tasks = []
        for item in pairs:
            if len(item) == 3:
                m, pn, pc = item
            else:
                m, pn, pc, _ = item
            tasks.append(_test(m, pn, pc))

        await asyncio.gather(*tasks, return_exceptions=True)
        return results

    # ── chain building (TOOL-AWARE) ─────────────────

    def get_chain(
        self, model: str, has_tools: bool = False
    ) -> List[Tuple[str, str, Any]]:
        """
        Build fallback chain.
        Jika has_tools=True → tool_capable providers PERTAMA.
        """
        if model.lower() == "auto":
            return self._auto_chain(has_tools)
        return self._model_chain(model, has_tools)

    def _auto_chain(self, has_tools: bool) -> List[Tuple[str, str, Any]]:
        tool_first: list[tuple[float, str, str, Any]] = []
        tool_no:    list[tuple[float, str, str, Any]] = []

        for mdl in PREFERRED_MODELS:
            pairs = self._model_map.get(mdl, [])
            try:
                idx = PREFERRED_MODELS.index(mdl)
                bonus = 0.25 * (1.0 - idx / len(PREFERRED_MODELS))
            except ValueError:
                bonus = 0.0

            for pn, pcls in pairs:
                h = self._health.get(pn, HealthRecord())
                if not h.is_healthy:
                    continue
                s = h.score + bonus
                if h.tool_capable:
                    tool_first.append((s, mdl, pn, pcls))
                else:
                    tool_no.append((s, mdl, pn, pcls))

        # remaining models
        for mdl, pairs in self._model_map.items():
            if mdl in PREFERRED_MODELS:
                continue
            for pn, pcls in pairs:
                h = self._health.get(pn, HealthRecord())
                if not h.is_healthy:
                    continue
                bucket = tool_first if h.tool_capable else tool_no
                bucket.append((h.score, mdl, pn, pcls))

        tool_first.sort(key=lambda x: x[0], reverse=True)
        tool_no.sort(key=lambda x: x[0], reverse=True)

        # Jika ada tools → tool_capable dulu
        if has_tools:
            ordered = tool_first + tool_no
        else:
            # Campur berdasarkan skor
            ordered = sorted(tool_first + tool_no, key=lambda x: x[0], reverse=True)

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
            tc = sum(1 for m, pn, _ in chain
                     if self._health.get(pn, HealthRecord()).tool_capable)
            log.info(
                f"  🎯 Auto chain: {len(chain)} total, "
                f"{tc} tool-capable, "
                f"top = {chain[0][1]} ({chain[0][0]})"
            )
        return chain

    def _model_chain(self, model: str, has_tools: bool) -> List[Tuple[str, str, Any]]:
        pairs = self._model_map.get(model)
        if not pairs:
            for key in self._model_map:
                if model in key or key in model:
                    pairs = self._model_map[key]
                    model = key
                    log.info(f"  🔍 Fuzzy: → '{model}'")
                    break

        chain: list[tuple[str, str, Any]] = []
        if pairs:
            tool_p = []
            other  = []
            for pn, pcls in pairs:
                h = self._health.get(pn, HealthRecord())
                if not h.is_healthy:
                    continue
                if h.tool_capable:
                    tool_p.append((h.score, model, pn, pcls))
                else:
                    other.append((h.score, model, pn, pcls))

            tool_p.sort(key=lambda x: x[0], reverse=True)
            other.sort(key=lambda x: x[0], reverse=True)

            if has_tools:
                ordered = tool_p + other
            else:
                ordered = sorted(tool_p + other, key=lambda x: x[0], reverse=True)

            for _, m, pn, pcls in ordered:
                chain.append((m, pn, pcls))

        chain.append((model, "g4f-auto", None))
        return chain

    # ── reporting ───────────────────────────────────

    def record_success(self, pname: str, latency_ms: float = 0.0):
        self._health.setdefault(pname, HealthRecord()).record_ok(latency_ms)

    def record_failure(self, pname: str):
        self._health.setdefault(pname, HealthRecord()).record_fail()

    def mark_tool_failed(self, pname: str):
        """Runtime: provider gagal memanggil tool padahal diminta."""
        h = self._health.get(pname)
        if h:
            h.tool_capable = False
            log.info(f"  ⚠️  {pname} ditandai BUKAN tool-capable (runtime)")

    def list_models(self) -> list[str]:
        return sorted(self._model_map.keys())

    def status(self) -> dict:
        return {
            pn: {
                "healthy": h.is_healthy,
                "score": round(h.score, 3),
                "tool_capable": h.tool_capable,
                "tool_tested": h.tool_tested,
                "successes": h.successes,
                "failures": h.failures,
                "consecutive_fails": h.consecutive_fails,
                "avg_latency_ms": round(h.avg_latency, 1),
            }
            for pn, h in sorted(self._health.items())
        }


# global
registry = ProviderRegistry()


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
#  UTILITAS
# ══════════════════════════════════════════════════════════════

def _id(prefix: str = "chatcmpl") -> str:
    return f"{prefix}-{uuid.uuid4().hex[:12]}"

def clean_model(raw: str) -> str:
    return raw.split("/", 1)[-1] if "/" in raw else raw


# ══════════════════════════════════════════════════════════════
#  TOOL PROMPT — Diperkuat agar model lebih patuh
# ══════════════════════════════════════════════════════════════

def build_tool_prompt(tools: List[ToolDef], tool_choice: Any = None) -> str:
    blocks = []
    for i, t in enumerate(tools, 1):
        fn = t.function
        p  = json.dumps(fn.parameters, ensure_ascii=False, indent=2)
        blocks.append(
            f"Tool #{i}:\n"
            f"  name: {fn.name}\n"
            f"  description: {fn.description or '-'}\n"
            f"  parameters: {p}"
        )
    tools_text = "\n\n".join(blocks)
    names = [t.function.name for t in tools]

    force = ""
    if isinstance(tool_choice, dict):
        forced = tool_choice.get("function", {}).get("name", "")
        force = f"\n\n⛔ MANDATORY: You MUST call tool '{forced}'. Do NOT reply with plain text."
    elif tool_choice == "required":
        force = "\n\n⛔ MANDATORY: You MUST call one of the tools. Do NOT reply with plain text."

    return (
        "=== INTERNAL INSTRUCTION: FUNCTION CALLING ===\n\n"
        "You have access to these tools:\n\n"
        f"{tools_text}\n\n"
        "STRICT RULES:\n"
        f"1. If the user's request can be answered by calling a tool ({', '.join(names)}), "
        "you MUST respond with ONLY a single-line JSON in this exact format:\n\n"
        '   {"nama_alat": "<tool_name>", "argumen": {<parameters as key-value>}}\n\n'
        f'   Example: {{"nama_alat": "{names[0]}", "argumen": {{"key": "value"}}}}\n\n'
        "2. Do NOT wrap in markdown code blocks (no ```).\n"
        "3. Do NOT add any text before or after the JSON.\n"
        "4. The JSON must be parseable by json.loads().\n"
        "5. Only use tool names from the list above.\n"
        "6. If you genuinely do NOT need any tool, reply normally.\n"
        f"{force}\n"
        "=== END INSTRUCTION ==="
    )


def preprocess_messages(
    messages: List[MessageIn],
    tools: Optional[List[ToolDef]],
    tool_choice: Any = None,
) -> List[Dict[str, str]]:
    result: list[dict[str, str]] = []

    has_tools  = bool(tools and len(tools) > 0)
    skip_tools = tool_choice == "none"

    if has_tools and not skip_tools:
        prompt = build_tool_prompt(tools, tool_choice)
        result.append({"role": "system", "content": prompt})
        log.info("  🔧 Tool system prompt injected")

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
            name = msg.name or "?"
            cid  = msg.tool_call_id or "?"
            result.append({
                "role": "user",
                "content": f"[Tool result '{name}' (id:{cid})]:\n{content}",
            })
        elif role == "assistant" and msg.tool_calls:
            descs = []
            for tc in msg.tool_calls:
                fn = tc.get("function", {}) if isinstance(tc, dict) else {}
                descs.append(f"{fn.get('name','?')}({fn.get('arguments','{}')})")
            result.append({
                "role": "assistant",
                "content": f"I called: {', '.join(descs)}",
            })
        elif role == "function":
            result.append({
                "role": "user",
                "content": f"[Function result '{msg.name or '?'}']: {content}",
            })
        else:
            result.append({"role": role, "content": str(content)})

    return result


# ══════════════════════════════════════════════════════════════
#  JSON EXTRACTOR
# ══════════════════════════════════════════════════════════════

def _strip_md(text: str) -> str:
    return re.sub(r"```(?:json|JSON)?\s*\n?(.*?)\n?\s*```",
                  r"\1", text, flags=re.DOTALL).strip()

def _find_jsons(text: str) -> list[dict]:
    results = []
    n = len(text)
    i = 0
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
                if ch == "{": depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        try:
                            obj = json.loads(text[i:j+1])
                            if isinstance(obj, dict):
                                results.append(obj)
                        except json.JSONDecodeError:
                            pass
                        break
            j += 1
        i += 1
    return results

_NAME_KEYS = ["nama_alat", "name", "tool_name", "function", "function_name"]
_ARGS_KEYS = ["argumen", "arguments", "args", "params", "parameters"]

def try_parse_tool_call(
    raw: str,
    tools: Optional[List[ToolDef]] = None,
) -> Optional[Dict[str, Any]]:
    if not raw or not raw.strip():
        return None

    valid_names = {t.function.name for t in tools} if tools else set()

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

    log.debug(f"  🔍 {len(candidates)} JSON candidate(s)")

    for obj in candidates:
        for nk in _NAME_KEYS:
            if nk not in obj:
                continue
            fn_name = obj[nk]
            if not isinstance(fn_name, str):
                continue
            if valid_names and fn_name not in valid_names:
                log.debug(f"  ⚠️ '{fn_name}' not in {valid_names}")
                continue

            arguments = {}
            for ak in _ARGS_KEYS:
                if ak in obj:
                    arguments = obj[ak]; break
            if isinstance(arguments, str):
                try: arguments = json.loads(arguments)
                except json.JSONDecodeError: arguments = {}
            if not isinstance(arguments, dict):
                arguments = {}

            log.info(f"  🎯 Tool call: {fn_name}({json.dumps(arguments, ensure_ascii=False)})")
            return {"name": fn_name, "arguments": arguments}

    return None


# ══════════════════════════════════════════════════════════════
#  RESPONSE BUILDERS
# ══════════════════════════════════════════════════════════════

_ZERO_USAGE = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

def resp_chat(model: str, content: str) -> dict:
    return {
        "id": _id(), "object": "chat.completion",
        "created": int(time.time()), "model": model,
        "choices": [{"index": 0,
                     "message": {"role": "assistant", "content": content},
                     "finish_reason": "stop"}],
        "usage": _ZERO_USAGE,
    }

def resp_tool(model: str, tool_name: str, arguments: dict) -> dict:
    return {
        "id": _id(), "object": "chat.completion",
        "created": int(time.time()), "model": model,
        "choices": [{"index": 0,
                     "message": {
                         "role": "assistant",
                         "content": "",
                         "tool_calls": [{
                             "id": _id("call"),
                             "type": "function",
                             "function": {
                                 "name": tool_name,
                                 "arguments": json.dumps(arguments, ensure_ascii=False),
                             },
                         }],
                     },
                     "finish_reason": "tool_calls"}],
        "usage": _ZERO_USAGE,
    }

def resp_error(msg: str, etype: str = "server_error", code: int = 500) -> dict:
    return {"error": {"message": msg, "type": etype, "param": None, "code": code}}


# ══════════════════════════════════════════════════════════════
#  G4F CALLER — Dengan Tool-Aware Fallback
# ══════════════════════════════════════════════════════════════

async def _raw_call(model, messages, provider=None, temperature=None, max_tokens=None):
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
        None, lambda: g4f.ChatCompletion.create(**kw),
    )
    if hasattr(r, "__iter__") and not isinstance(r, str):
        return "".join(str(c) for c in r).strip()
    return str(r).strip() if r else ""


async def _call_retry(model, messages, provider, pname, temperature=None, max_tokens=None):
    last_err: Exception = RuntimeError("no attempt")
    for attempt in range(1 + MAX_RETRIES):
        if attempt > 0:
            wait = RETRY_BACKOFF * (2 ** (attempt - 1))
            log.info(f"       🔄 retry {attempt}/{MAX_RETRIES} — wait {wait:.1f}s")
            await asyncio.sleep(wait)
        try:
            t0 = time.time()
            text = await asyncio.wait_for(
                _raw_call(model, messages, provider, temperature, max_tokens),
                timeout=G4F_TIMEOUT,
            )
            ms = (time.time() - t0) * 1000
            if not text: raise ValueError("empty response")
            log.info(f"       ✅ {pname} OK (attempt {attempt+1}, {ms:.0f}ms, {len(text)} chars)")
            return text, ms
        except asyncio.TimeoutError:
            last_err = TimeoutError(f"{pname} timeout"); log.warning(f"       ⏱️ {pname} attempt {attempt+1}: timeout")
        except Exception as exc:
            last_err = exc; log.warning(f"       ❌ {pname} attempt {attempt+1}: {exc}")
    raise last_err


async def call_g4f_smart(
    model_raw: str,
    messages: list[dict],
    tools: Optional[List[ToolDef]] = None,
    tool_choice: Any = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> Tuple[str, str, str]:
    """
    TOOL-AWARE FALLBACK:
      1. Bangun chain (tool_capable providers duluan jika ada tools)
      2. Panggil provider
      3. Jika tools diminta tapi respons = teks biasa:
         → coba provider berikutnya (max MAX_TOOL_TRIES kali)
      4. Jika semua tool_capable gagal: return teks biasa terbaik
    """
    model     = clean_model(model_raw)
    has_tools = bool(tools and len(tools) > 0)
    chain     = registry.get_chain(model, has_tools)

    # Apakah tool wajib?
    tool_required = tool_choice in ("required",) or isinstance(tool_choice, dict)

    log.info(f"  🔗 Chain: {len(chain)} candidates" +
             (f" (tool_required={tool_required})" if has_tools else ""))
    for i, (m, pn, _) in enumerate(chain[:6]):
        h = registry._health.get(pn, HealthRecord())
        tag = " 🔧" if h.tool_capable else ""
        log.info(f"     {i+1}. {pn} ({m}){tag}")
    if len(chain) > 6:
        log.info(f"     … +{len(chain)-6} more")

    last_err           = ""
    best_plain_text    = ""
    best_plain_model   = ""
    best_plain_prov    = ""
    tool_retries_left  = MAX_TOOL_TRIES if has_tools else 0

    for mdl, pname, pcls in chain:
        log.info(f"  ⏩ [{mdl}] trying: {pname}")
        try:
            text, latency = await _call_retry(
                mdl, messages, pcls, pname, temperature, max_tokens,
            )
            registry.record_success(pname, latency)

            # ── Cek apakah tool call berhasil ──
            if has_tools:
                tool_call = try_parse_tool_call(text, tools)

                if tool_call:
                    # Tool call berhasil!
                    log.info(f"  🏆 {pname} returned valid tool call!")
                    return text, mdl, pname

                # Provider merespons teks biasa...
                log.info(f"  ⚠️  {pname} responded with plain text (no tool call)")

                # Simpan sebagai fallback terbaik
                if not best_plain_text:
                    best_plain_text  = text
                    best_plain_model = mdl
                    best_plain_prov  = pname

                # Tandai provider ini sebagai non-tool-capable
                h = registry._health.get(pname)
                if h and h.tool_capable:
                    registry.mark_tool_failed(pname)

                tool_retries_left -= 1
                if tool_retries_left > 0:
                    log.info(f"  🔄 Tool retry ({tool_retries_left} left) → trying next provider...")
                    continue
                else:
                    # Habis retry, return teks biasa
                    log.info(f"  ℹ️  Tool retries exhausted, using plain text")
                    return best_plain_text, best_plain_model, best_plain_prov

            # Tidak ada tools → langsung return
            return text, mdl, pname

        except Exception as exc:
            registry.record_failure(pname)
            last_err = f"{pname}({mdl}): {exc}"
            log.warning(f"  ⛔ {pname}({mdl}) failed: {exc}")

    # Jika kita punya plain text fallback
    if best_plain_text:
        return best_plain_text, best_plain_model, best_plain_prov

    raise HTTPException(
        status_code=503,
        detail=resp_error(
            f"Semua provider gagal. Terakhir: {last_err}",
            "all_providers_failed", 503,
        ),
    )


# ══════════════════════════════════════════════════════════════
#  FASTAPI APP
# ══════════════════════════════════════════════════════════════

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app_: FastAPI):
    """Startup: jalankan scan provider + tool capability test."""
    log.info("🚀 G4F API Bridge v3 starting...")
    _sep("STARTUP SCAN")
    t0 = time.time()
    await registry.scan_all()
    log.info(f"⏱️  Scan selesai dalam {time.time()-t0:.1f}s")
    yield
    log.info("👋 Shutting down.")

app = FastAPI(
    title="G4F Bridge", version="3.0.0",
    description="Auto-Model + Tool-Scan + Tool-Aware Fallback",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

@app.middleware("http")
async def log_requests(request: Request, call_next):
    t0 = time.time()
    log.info(f"📡 ← {request.method} {request.url.path}")
    response = await call_next(request)
    ms = (time.time() - t0) * 1000
    emoji = "✅" if response.status_code < 400 else "❌"
    log.info(f"📡 → {emoji} {response.status_code} ({ms:.0f}ms)")
    return response


# ══════════════════════════════════════════════════════════════
#  ROUTE: POST /v1/chat/completions
# ══════════════════════════════════════════════════════════════

@app.post("/v1/chat/completions")
@app.post("/chat/completions")
async def chat_completions(req: ChatRequest):

    rid = _id("req")
    has_tools = bool(req.tools and len(req.tools) > 0)

    # ═══ LOG REQUEST ═══
    _sep(f"REQUEST {rid}")
    model_clean = clean_model(req.model)
    is_auto = model_clean.lower() == "auto"

    log.info(f"  📨 model       : {req.model} → '{model_clean}'" +
             (" (AUTO)" if is_auto else ""))
    log.info(f"  📨 messages    : {len(req.messages)}")
    log.info(f"  📨 tools       : {'YA (' + str(len(req.tools)) + ')' if has_tools else 'TIDAK'}")
    log.info(f"  📨 tool_choice : {req.tool_choice}")
    if has_tools:
        for t in req.tools:
            log.info(f"     🔧 {t.function.name} — {t.function.description or '-'}")

    _log_json("REQ.body", {
        "model": req.model,
        "messages": [
            {"role": m.role,
             "content": (str(m.content)[:120] + "…" if m.content and len(str(m.content)) > 120 else m.content)}
            for m in req.messages
        ],
        "tools": [t.function.name for t in req.tools] if has_tools else None,
    })

    # ═══ PREPROCESS ═══
    _sep("PREPROCESSING")
    processed = preprocess_messages(req.messages, req.tools, req.tool_choice)
    log.info(f"  ⚙️  Messages final: {len(processed)}")
    _log_json("PROCESSED", processed)

    # ═══ CALL G4F (tool-aware) ═══
    _sep("G4F CALL")
    log.info(f"  📤 → g4f (model='{model_clean}', timeout={G4F_TIMEOUT}s)")

    try:
        raw, actual_model, provider_used = await call_g4f_smart(
            model_raw=model_clean,
            messages=processed,
            tools=req.tools if has_tools else None,
            tool_choice=req.tool_choice,
            temperature=req.temperature,
            max_tokens=req.max_tokens,
        )
    except HTTPException:
        raise
    except Exception as exc:
        log.error(f"  💥 {type(exc).__name__}: {exc}")
        return JSONResponse(status_code=502,
                            content=resp_error(f"g4f error: {exc}", "upstream_error", 502))

    log.info(f"  📥 ← via {provider_used} (model={actual_model}, {len(raw)} chars)")
    _log_json("G4F.raw", raw)

    if not raw:
        return JSONResponse(status_code=502,
                            content=resp_error("empty response", "empty_response", 502))

    # ═══ PARSE: teks biasa vs tool call ═══
    _sep("PARSING")

    tool_call = None
    if has_tools:
        log.info("  🔍 Mencoba deteksi tool call…")
        tool_call = try_parse_tool_call(raw, req.tools)

    if tool_call:
        log.info(f"  ✅ KONDISI B: Tool Call")
        log.info(f"     name      = {tool_call['name']}")
        log.info(f"     arguments = {json.dumps(tool_call['arguments'], ensure_ascii=False)}")
        response = resp_tool(actual_model, tool_call["name"], tool_call["arguments"])
    else:
        log.info(f"  ℹ️  KONDISI A: Teks biasa" +
                 (" (tools tersedia tapi tidak dipanggil)" if has_tools else ""))
        response = resp_chat(actual_model, raw)

    # ═══ RETURN ═══
    _sep("RESPONSE → CLIENT")
    log.info(f"  📦 model         = {actual_model}")
    log.info(f"  📦 provider      = {provider_used}")
    log.info(f"  📦 finish_reason = {response['choices'][0]['finish_reason']}")
    _log_json("RESPONSE", response)
    _sep()

    return JSONResponse(
        content=response,
        headers={"X-G4F-Provider": provider_used, "X-G4F-Model": actual_model},
    )


# ══════════════════════════════════════════════════════════════
#  ROUTE: OTHER ENDPOINTS
# ══════════════════════════════════════════════════════════════

@app.get("/v1/models")
@app.get("/models")
async def list_models():
    models = registry.list_models()
    return {
        "object": "list",
        "data": [{"id": m, "object": "model", "created": 0, "owned_by": "g4f"} for m in models]
             + [{"id": "auto", "object": "model", "created": 0, "owned_by": "bridge",
                 "description": "Auto-select best model + provider"}],
    }

@app.post("/v1/scan")
async def trigger_rescan():
    log.info("🔄 Manual rescan…")
    t0 = time.time()
    await registry.scan_all()
    elapsed = time.time() - t0

    st = registry.status()
    alive = sum(1 for v in st.values() if v["successes"] > 0)
    tools = sum(1 for v in st.values() if v["tool_capable"])
    return {
        "status": "done",
        "duration_s": round(elapsed, 1),
        "providers_alive": alive,
        "providers_tool_capable": tools,
        "models": len(registry.list_models()),
    }

@app.get("/v1/providers/status")
async def provider_status():
    report = registry.status()
    return {
        "total": len(report),
        "healthy": sum(1 for v in report.values() if v["healthy"]),
        "tool_capable": sum(1 for v in report.values() if v["tool_capable"]),
        "providers": report,
    }

@app.get("/health")
async def health():
    return {"status": "ok", "models": len(registry.list_models()), "ts": int(time.time())}

@app.get("/")
async def root():
    st = registry.status()
    return {
        "service": "G4F API Bridge",
        "version": "3.0.0",
        "auto_model": True,
        "tool_scan": True,
        "providers_alive": sum(1 for v in st.values() if v["healthy"]),
        "providers_tool_capable": sum(1 for v in st.values() if v["tool_capable"]),
        "endpoints": {
            "chat":   "POST /v1/chat/completions",
            "models": "GET  /v1/models",
            "scan":   "POST /v1/scan",
            "status": "GET  /v1/providers/status",
        },
    }


# ══════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn

    addr = f"http://{HOST}:{PORT}"
    print(f"""
╔═══════════════════════════════════════════════════╗
║           G4F  API  BRIDGE  v3.0                  ║
║   Auto-Model + Tool-Scan + Tool-Aware Fallback    ║
╠═══════════════════════════════════════════════════╣
║  🌐 Server  : {addr:<35s}║
║  📖 Docs    : {addr + '/docs':<35s}║
║  ⏱️  Timeout : {str(G4F_TIMEOUT) + 's':<35s}║
║  🔧 ToolTry : {str(MAX_TOOL_TRIES) + ' providers max':<35s}║
╚═══════════════════════════════════════════════════╝""")

    uvicorn.run("server:app", host=HOST, port=PORT, log_level="warning")