"""
server.py  v5
═════════════
G4F Bridge — Fixed null content for tool_calls

Changelog v4 → v5:
  ✦ FIX: tool_calls content → null (bukan "")  [CRITICAL - penyebab loop]
  ✦ FIX: Custom JSON serializer yang bisa omit atau null-kan content
  ✦ FIX: Response builder juga untuk force-text cleaning
  ✦ FEAT: TOOL_CONTENT_MODE env var ("null" | "omit")
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
from fastapi.responses import JSONResponse, Response
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

# ← FIX: Mode content untuk tool_calls response
# "null"  → {"content": null, "tool_calls": [...]}    (standar OpenAI)
# "omit"  → {"tool_calls": [...]}                     (tanpa content sama sekali)
TOOL_CONTENT_MODE = os.getenv("TOOL_CONTENT_MODE", "null")

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
#  ← FIX: CUSTOM JSON RESPONSE — Handles null vs omit content
# ══════════════════════════════════════════════════════════════

_SENTINEL = object()   # marker internal untuk "hapus field ini"


class PicoResponse(Response):
    """
    JSONResponse khusus yang:
      - Menghasilkan "content": null   (bukan "content": "")
      - Atau menghapus "content" dari message jika mode = "omit"

    Ini menghindari crash di PicoClaw yang tidak bisa handle
    content="" berdampingan dengan tool_calls.
    """
    media_type = "application/json"

    def __init__(
        self,
        content: Any,
        status_code: int = 200,
        headers: dict | None = None,
    ):
        body = self._serialize(content)
        super().__init__(
            content=body,
            status_code=status_code,
            headers=headers,
            media_type=self.media_type,
        )

    @staticmethod
    def _serialize(data: Any) -> bytes:
        """
        Serialisasi dict → JSON bytes.
        Menangani _SENTINEL untuk penghapusan field.
        """
        def _clean(obj):
            if isinstance(obj, dict):
                cleaned = {}
                for k, v in obj.items():
                    if v is _SENTINEL:
                        continue       # ← omit field ini
                    cleaned[k] = _clean(v)
                return cleaned
            elif isinstance(obj, list):
                return [_clean(item) for item in obj]
            return obj

        cleaned = _clean(data)
        return json.dumps(
            cleaned,
            ensure_ascii=False,
            separators=(",", ":"),     # compact JSON
        ).encode("utf-8")


# ══════════════════════════════════════════════════════════════
#  TOOL SCAN — Prompt & Validator
# ══════════════════════════════════════════════════════════════

_TOOL_TEST_SYSTEM = (
    "You have ONE tool: get_test_info(query: string).\n"
    "You MUST call it for ANY question.\n"
    "Reply ONLY with this exact JSON, nothing else:\n"
    '{"nama_alat": "get_test_info", "argumen": {"query": "<user question>"}}\n'
    "NO markdown. NO explanation. ONLY the JSON."
)
_TOOL_TEST_MESSAGES: list[dict] = [
    {"role": "system", "content": _TOOL_TEST_SYSTEM},
    {"role": "user",   "content": "What is Python?"},
]
_BASIC_TEST_MESSAGES: list[dict] = [
    {"role": "user", "content": "Reply with exactly: ok"},
]


def _is_valid_tool_response(text: str) -> bool:
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
    for match in re.finditer(r'\{[^{}]*"(?:nama_alat|name)"[^{}]*\}', cleaned):
        try:
            obj = json.loads(match.group())
            if obj.get("nama_alat") == "get_test_info" or obj.get("name") == "get_test_info":
                return True
        except (json.JSONDecodeError, TypeError):
            pass
    return False


# ══════════════════════════════════════════════════════════════
#  PROVIDER REGISTRY
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
        log.info(f"📚 Registry: {len(self._model_map)} model, "
                 f"{len(self._health)} provider, {count} pasangan")

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

    async def scan_all(self):
        all_pairs = self._all_pairs()
        if not all_pairs:
            log.warning("⚠️  Tidak ada pasangan untuk di-scan")
            return

        _sep("PHASE 1: ALIVE SCAN")
        log.info(f"  🧪 Menguji {len(all_pairs)} pasangan "
                 f"(max {SCAN_WORKERS} parallel)")
        alive = await self._scan_phase(all_pairs, _BASIC_TEST_MESSAGES, "ALIVE")

        _sep("PHASE 2: TOOL CAPABILITY SCAN")
        if not alive:
            log.warning("  ⚠️  Tidak ada provider hidup")
            self._scan_done = True
            return

        log.info(f"  🔧 Menguji {len(alive)} provider hidup "
                 f"untuk tool capability")
        tool_ok = await self._scan_phase(alive, _TOOL_TEST_MESSAGES, "TOOL")

        tool_names = set()
        for model, pname, pcls, _ in tool_ok:
            h = self._health.get(pname)
            if h:
                h.tool_capable = True
                h.tool_tested = True
                tool_names.add(pname)

        alive_names = {pname for _, pname, _, _ in alive}
        for pname in alive_names - tool_names:
            h = self._health.get(pname)
            if h:
                h.tool_tested = True
                h.tool_capable = False

        _sep("SCAN COMPLETE")
        log.info(f"  📊 Total: {len(all_pairs)} │ "
                 f"Hidup: {len(alive)} │ Tool-capable: {len(tool_ok)}")
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

    async def _scan_phase(self, pairs, test_messages, phase):
        sem = asyncio.Semaphore(SCAN_WORKERS)
        results: list[tuple[str, str, Any, float]] = []

        async def _test(model, pname, pcls):
            async with sem:
                t0 = time.time()
                try:
                    kw = {"model": model, "messages": test_messages,
                          "provider": pcls}
                    try:
                        text = await asyncio.wait_for(
                            g4f.ChatCompletion.create_async(**kw),
                            timeout=SCAN_TIMEOUT)
                    except (AttributeError, NotImplementedError, TypeError):
                        loop = asyncio.get_running_loop()
                        text = await asyncio.wait_for(
                            loop.run_in_executor(
                                None,
                                lambda: g4f.ChatCompletion.create(**kw)),
                            timeout=SCAN_TIMEOUT)

                    if hasattr(text, "__iter__") and not isinstance(text, str):
                        text = "".join(str(c) for c in text)
                    text = str(text).strip() if text else ""
                    ms = (time.time() - t0) * 1000
                    if not text:
                        raise ValueError("empty")

                    if phase == "TOOL":
                        if _is_valid_tool_response(text):
                            log.info(f"  ✅ {phase:5s} │ {pname:28s} │ "
                                     f"{model:24s} │ {ms:6.0f}ms │ 🔧 OK")
                            results.append((model, pname, pcls, ms))
                        else:
                            log.debug(f"  ❌ {phase:5s} │ {pname:28s} │ "
                                      f"{model:24s} │ no tool JSON")
                    else:
                        log.info(f"  ✅ {phase:5s} │ {pname:28s} │ "
                                 f"{model:24s} │ {ms:6.0f}ms")
                        results.append((model, pname, pcls, ms))
                        h = self._health.get(pname)
                        if h:
                            h.record_ok(ms)

                except asyncio.TimeoutError:
                    log.debug(f"  ⏱️ {phase:5s} │ {pname:28s} │ "
                              f"{model:24s} │ timeout")
                except Exception as exc:
                    log.debug(f"  ❌ {phase:5s} │ {pname:28s} │ "
                              f"{model:24s} │ {str(exc)[:60]}")

        tasks = [_test(item[0], item[1], item[2]) for item in pairs]
        await asyncio.gather(*tasks, return_exceptions=True)
        return results

    def get_chain(self, model: str,
                  has_tools: bool = False) -> List[Tuple[str, str, Any]]:
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
                (tool_first if h.tool_capable else tool_no).append(
                    (s, mdl, pn, pcls))

        for mdl, pairs in self._model_map.items():
            if mdl in PREFERRED_MODELS:
                continue
            for pn, pcls in pairs:
                h = self._health.get(pn, HealthRecord())
                if not h.is_healthy:
                    continue
                (tool_first if h.tool_capable else tool_no).append(
                    (h.score, mdl, pn, pcls))

        tool_first.sort(key=lambda x: x[0], reverse=True)
        tool_no.sort(key=lambda x: x[0], reverse=True)
        ordered = ((tool_first + tool_no) if has_tools
                   else sorted(tool_first + tool_no,
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
            tc = sum(1 for m, pn, _ in chain
                     if self._health.get(pn, HealthRecord()).tool_capable)
            log.info(f"  🎯 Auto chain: {len(chain)} total, "
                     f"{tc} tool-capable, "
                     f"top = {chain[0][1]} ({chain[0][0]})")
        return chain

    def _model_chain(self, model: str,
                     has_tools: bool) -> List[Tuple[str, str, Any]]:
        pairs = self._model_map.get(model)
        if not pairs:
            for key in self._model_map:
                if model in key or key in model:
                    pairs = self._model_map[key]
                    model = key
                    break

        chain: list[tuple[str, str, Any]] = []
        if pairs:
            tool_p, other = [], []
            for pn, pcls in pairs:
                h = self._health.get(pn, HealthRecord())
                if not h.is_healthy:
                    continue
                (tool_p if h.tool_capable else other).append(
                    (h.score, model, pn, pcls))
            tool_p.sort(key=lambda x: x[0], reverse=True)
            other.sort(key=lambda x: x[0], reverse=True)
            ordered = ((tool_p + other) if has_tools
                       else sorted(tool_p + other,
                                   key=lambda x: x[0], reverse=True))
            for _, m, pn, pcls in ordered:
                chain.append((m, pn, pcls))

        chain.append((model, "g4f-auto", None))
        return chain

    def record_success(self, pname, latency_ms=0.0):
        self._health.setdefault(pname, HealthRecord()).record_ok(latency_ms)

    def record_failure(self, pname):
        self._health.setdefault(pname, HealthRecord()).record_fail()

    def mark_tool_failed(self, pname):
        h = self._health.get(pname)
        if h:
            h.tool_capable = False
            log.info(f"  ⚠️  {pname} ditandai BUKAN tool-capable")

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
#  CONVERSATION STATE ANALYZER
# ══════════════════════════════════════════════════════════════

@dataclass
class ConversationState:
    trailing_tool_rounds: int = 0
    has_pending_results: bool = False
    force_text: bool = False
    last_user_text: str = ""


def analyze_conversation(messages: List[MessageIn]) -> ConversationState:
    state = ConversationState()
    if not messages:
        return state

    for msg in reversed(messages):
        if msg.role == "user" and not msg.tool_call_id:
            state.last_user_text = str(msg.content or "")
            break

    i = len(messages) - 1
    rounds = 0

    while i >= 0:
        msg = messages[i]
        if msg.role == "tool":
            state.has_pending_results = True
            while i >= 0 and messages[i].role == "tool":
                i -= 1
            if i >= 0 and messages[i].role == "assistant":
                content = str(messages[i].content or "").strip()
                has_tc  = bool(messages[i].tool_calls)
                is_empty = content == "" or content == "null"
                if has_tc or is_empty:
                    rounds += 1
                    i -= 1
                    continue
                else:
                    break
            else:
                break
        else:
            break

    state.trailing_tool_rounds = rounds
    state.force_text = rounds >= MAX_CONSECUTIVE_TOOLS
    return state


# ══════════════════════════════════════════════════════════════
#  TOOL PROMPT
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
            f"  parameters: {p}")
    tools_text = "\n\n".join(blocks)
    names = [t.function.name for t in tools]

    force = ""
    if isinstance(tool_choice, dict):
        forced = tool_choice.get("function", {}).get("name", "")
        force = (f"\n\n⛔ MANDATORY: You MUST call tool '{forced}'. "
                 f"Do NOT reply with plain text.")
    elif tool_choice == "required":
        force = ("\n\n⛔ MANDATORY: You MUST call one of the tools. "
                 "Do NOT reply with plain text.")

    return (
        "=== INTERNAL INSTRUCTION: FUNCTION CALLING ===\n\n"
        "You have access to these tools:\n\n"
        f"{tools_text}\n\n"
        "STRICT RULES:\n"
        f"1. If the user's request REQUIRES a tool ({', '.join(names)}), "
        "respond with ONLY a single-line JSON:\n\n"
        '   {"nama_alat": "<tool_name>", "argumen": {<parameters>}}\n\n'
        f'   Example: {{"nama_alat": "{names[0]}", '
        f'"argumen": {{"key": "value"}}}}\n\n'
        "2. Do NOT wrap in markdown code blocks.\n"
        "3. Do NOT add any text before or after the JSON.\n"
        "4. Only use tool names from the list above.\n"
        "5. If the user is just chatting, reply normally "
        "WITHOUT calling any tool.\n"
        f"{force}\n"
        "=== END INSTRUCTION ===")


_POST_RESULT_PROMPT = (
    "=== IMPORTANT ===\n"
    "Tool results from your previous call are included above.\n"
    "You MUST now:\n"
    "  1. Read and process the tool results\n"
    "  2. Respond to the user with a helpful, natural text message\n"
    "  3. Do NOT call another tool unless the user EXPLICITLY asks\n"
    "  4. NEVER repeat the same tool call\n"
    "=== END ===")

_FORCE_TEXT_PROMPT = (
    "=== CRITICAL: STOP CALLING TOOLS ===\n"
    "You have been calling tools repeatedly without answering.\n"
    "This is a LOOP and must stop NOW.\n\n"
    "You MUST respond with a natural text message.\n"
    "Do NOT output any JSON.\n"
    "Do NOT call any tool.\n"
    "Use whatever information you already have.\n"
    "=== END ===")


def preprocess_messages(
    messages: List[MessageIn],
    tools: Optional[List[ToolDef]],
    tool_choice: Any = None,
    conv_state: Optional[ConversationState] = None,
) -> List[Dict[str, str]]:
    result: list[dict[str, str]] = []

    has_tools   = bool(tools and len(tools) > 0)
    skip_tools  = tool_choice == "none"
    force_text  = conv_state.force_text if conv_state else False
    has_pending = conv_state.has_pending_results if conv_state else False
    rounds      = conv_state.trailing_tool_rounds if conv_state else 0

    if has_tools and not skip_tools:
        if force_text:
            result.append({"role": "system", "content": _FORCE_TEXT_PROMPT})
            log.info(f"  🛑 FORCE TEXT: anti-loop prompt "
                     f"({rounds} rounds)")
        elif has_pending:
            prompt = build_tool_prompt(tools, tool_choice)
            result.append({"role": "system", "content": prompt})
            result.append({"role": "system", "content": _POST_RESULT_PROMPT})
            log.info(f"  🔧 Tool prompt + post-result guidance "
                     f"({rounds} round(s))")
        else:
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
                descs.append(
                    f"{fn.get('name','?')}({fn.get('arguments','{}')})")
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
            i += 1
            continue
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
                    arguments = obj[ak]
                    break
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    arguments = {}
            if not isinstance(arguments, dict):
                arguments = {}

            log.info(f"  🎯 Tool call: {fn_name}("
                     f"{json.dumps(arguments, ensure_ascii=False)})")
            return {"name": fn_name, "arguments": arguments}

    return None


def _clean_forced_text(raw: str,
                       tools: Optional[List[ToolDef]] = None) -> str:
    tc = try_parse_tool_call(raw, tools)
    if tc:
        log.warning(f"  🛑 Model masih output tool call di force-text, "
                    f"cleaning: {tc['name']}")
        return ("Saya sudah memproses informasi yang tersedia. "
                "Silakan sampaikan apa yang ingin Anda ketahui.")

    cleaned = re.sub(r'^\s*```(?:json)?\s*\{.*?\}\s*```\s*', '',
                     raw, flags=re.DOTALL).strip()
    return cleaned if cleaned else raw


# ══════════════════════════════════════════════════════════════
#  ← FIX: RESPONSE BUILDERS — null content, not empty string
# ══════════════════════════════════════════════════════════════

_ZERO_USAGE = {
    "prompt_tokens": 0,
    "completion_tokens": 0,
    "total_tokens": 0,
}


def resp_chat(model: str, content: str) -> dict:
    """KONDISI A: Teks biasa. content = string."""
    return {
        "id": _id(),
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": content,         # ← string normal
            },
            "finish_reason": "stop",
        }],
        "usage": _ZERO_USAGE,
    }


def resp_tool(model: str, tool_name: str, arguments: dict) -> dict:
    """
    KONDISI B: Tool call.

    ← FIX KRITIS:
    content harus null (None) atau dihilangkan,
    BUKAN string kosong "".

    PicoClaw crash jika content="" bersama tool_calls.
    """
    args_string = json.dumps(arguments, ensure_ascii=False)

    # ← FIX: Bangun message berdasarkan TOOL_CONTENT_MODE
    message: dict[str, Any] = {
        "role": "assistant",
    }

    if TOOL_CONTENT_MODE == "omit":
        # Mode "omit": tidak ada field "content" sama sekali
        pass  # content tidak ditambahkan ke dict
    else:
        # Mode "null" (default): content = null
        message["content"] = None              # ← FIX: None → null di JSON

    message["tool_calls"] = [{
        "id": _id("call"),
        "type": "function",
        "function": {
            "name": tool_name,
            "arguments": args_string,
        },
    }]

    log.info(f"  📐 content mode = '{TOOL_CONTENT_MODE}' → "              # ← FIX: log
             f"{'null' if TOOL_CONTENT_MODE != 'omit' else 'omitted'}")

    return {
        "id": _id(),
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "message": message,
            "finish_reason": "tool_calls",
        }],
        "usage": _ZERO_USAGE,
    }


def resp_error(msg, etype="server_error", code=500):
    return {
        "error": {
            "message": msg, "type": etype,
            "param": None, "code": code,
        }
    }


# ══════════════════════════════════════════════════════════════
#  G4F CALLER
# ══════════════════════════════════════════════════════════════

async def _raw_call(model, messages, provider=None,
                    temperature=None, max_tokens=None):
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


async def _call_retry(model, messages, provider, pname,
                      temperature=None, max_tokens=None):
    last_err: Exception = RuntimeError("no attempt")
    for attempt in range(1 + MAX_RETRIES):
        if attempt > 0:
            wait = RETRY_BACKOFF * (2 ** (attempt - 1))
            log.info(f"       🔄 retry {attempt}/{MAX_RETRIES} "
                     f"— wait {wait:.1f}s")
            await asyncio.sleep(wait)
        try:
            t0 = time.time()
            text = await asyncio.wait_for(
                _raw_call(model, messages, provider,
                          temperature, max_tokens),
                timeout=G4F_TIMEOUT)
            ms = (time.time() - t0) * 1000
            if not text:
                raise ValueError("empty response")
            log.info(f"       ✅ {pname} OK (attempt {attempt+1}, "
                     f"{ms:.0f}ms, {len(text)} chars)")
            return text, ms
        except asyncio.TimeoutError:
            last_err = TimeoutError(f"{pname} timeout")
            log.warning(f"       ⏱️ {pname} attempt {attempt+1}: timeout")
        except Exception as exc:
            last_err = exc
            log.warning(f"       ❌ {pname} attempt {attempt+1}: {exc}")
    raise last_err


async def call_g4f_smart(
    model_raw, messages, tools=None, tool_choice=None,
    temperature=None, max_tokens=None, force_text=False,
) -> Tuple[str, str, str]:

    model     = clean_model(model_raw)
    has_tools = bool(tools and len(tools) > 0) and not force_text
    chain     = registry.get_chain(model, has_tools)

    log.info(f"  🔗 Chain: {len(chain)} candidates" +
             (" (FORCE TEXT)" if force_text else
              " (tool_aware)" if has_tools else ""))
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
                mdl, messages, pcls, pname, temperature, max_tokens)
            registry.record_success(pname, latency)

            if force_text:
                cleaned = _clean_forced_text(text, tools)
                return cleaned, mdl, pname

            if has_tools:
                tool_call = try_parse_tool_call(text, tools)
                if tool_call:
                    log.info(f"  🏆 {pname} returned valid tool call!")
                    return text, mdl, pname

                log.info(f"  ⚠️  {pname} responded plain text")
                if not best_plain_text:
                    best_plain_text  = text
                    best_plain_model = mdl
                    best_plain_prov  = pname

                h = registry._health.get(pname)
                if h and h.tool_capable:
                    registry.mark_tool_failed(pname)

                tool_retries_left -= 1
                if tool_retries_left > 0:
                    log.info(f"  🔄 Tool retry ({tool_retries_left} left)")
                    continue
                else:
                    log.info("  ℹ️  Tool retries exhausted, using plain text")
                    return (best_plain_text,
                            best_plain_model, best_plain_prov)

            return text, mdl, pname

        except Exception as exc:
            registry.record_failure(pname)
            last_err = f"{pname}({mdl}): {exc}"
            log.warning(f"  ⛔ {pname}({mdl}) failed: {exc}")

    if best_plain_text:
        return best_plain_text, best_plain_model, best_plain_prov

    raise HTTPException(
        status_code=503,
        detail=resp_error(
            f"Semua provider gagal. Terakhir: {last_err}",
            "all_providers_failed", 503))


# ══════════════════════════════════════════════════════════════
#  FASTAPI APP
# ══════════════════════════════════════════════════════════════

@asynccontextmanager
async def lifespan(app_: FastAPI):
    log.info("🚀 G4F API Bridge v5 starting...")
    log.info(f"📐 TOOL_CONTENT_MODE = '{TOOL_CONTENT_MODE}'")     # ← FIX: log mode
    _sep("STARTUP SCAN")
    t0 = time.time()
    await registry.scan_all()
    log.info(f"⏱️  Scan: {time.time()-t0:.1f}s")
    yield
    log.info("👋 Shutting down.")


app = FastAPI(
    title="G4F Bridge", version="5.0.0",
    description="Fixed null content + Anti-Loop",
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

    conv_state = analyze_conversation(req.messages)

    _sep(f"REQUEST {rid}")
    model_clean = clean_model(req.model)
    is_auto = model_clean.lower() == "auto"

    log.info(f"  📨 model       : {req.model} → '{model_clean}'" +
             (" (AUTO)" if is_auto else ""))
    log.info(f"  📨 messages    : {len(req.messages)}")
    log.info(f"  📨 tools       : "
             f"{'YA (' + str(len(req.tools)) + ')' if has_tools else 'TIDAK'}")
    log.info(f"  📨 tool_choice : {req.tool_choice}")

    if conv_state.force_text:
        log.info(f"  🛑 LOOP DETECTED: {conv_state.trailing_tool_rounds} "
                 f"rounds → FORCE TEXT")
    elif conv_state.has_pending_results:
        log.info(f"  📎 Pending results: "
                 f"{conv_state.trailing_tool_rounds} round(s)")
    else:
        log.info("  ℹ️  Fresh conversation")

    if has_tools:
        for t in req.tools:
            log.info(f"     🔧 {t.function.name} — "
                     f"{(t.function.description or '-')[:60]}")

    # ═══ PREPROCESS ═══
    _sep("PREPROCESSING")
    processed = preprocess_messages(
        req.messages, req.tools, req.tool_choice, conv_state)
    log.info(f"  ⚙️  Messages final: {len(processed)}")

    # ═══ CALL G4F ═══
    _sep("G4F CALL")
    log.info(f"  📤 → g4f (model='{model_clean}', "
             f"force_text={conv_state.force_text})")

    try:
        raw, actual_model, provider_used = await call_g4f_smart(
            model_raw=model_clean,
            messages=processed,
            tools=req.tools if has_tools else None,
            tool_choice=req.tool_choice,
            temperature=req.temperature,
            max_tokens=req.max_tokens,
            force_text=conv_state.force_text,
        )
    except HTTPException:
        raise
    except Exception as exc:
        log.error(f"  💥 {type(exc).__name__}: {exc}")
        return JSONResponse(
            status_code=502,
            content=resp_error(f"g4f error: {exc}", "upstream_error", 502))

    log.info(f"  📥 ← via {provider_used} "
             f"(model={actual_model}, {len(raw)} chars)")
    _log_json("G4F.raw", raw)

    if not raw:
        return JSONResponse(
            status_code=502,
            content=resp_error("empty response", "empty_response", 502))

    # ═══ PARSE ═══
    _sep("PARSING")

    tool_call = None
    if conv_state.force_text:
        log.info("  🛑 Force text → skip tool parsing")
        raw = _clean_forced_text(raw, req.tools)
    elif has_tools:
        log.info("  🔍 Mencoba deteksi tool call…")
        tool_call = try_parse_tool_call(raw, req.tools)

    if tool_call:
        log.info("  ✅ KONDISI B: Tool Call")
        log.info(f"     name      = {tool_call['name']}")
        log.info(f"     arguments = "
                 f"{json.dumps(tool_call['arguments'], ensure_ascii=False)}")
        response = resp_tool(
            actual_model, tool_call["name"], tool_call["arguments"])
    else:
        log.info("  ℹ️  KONDISI A: Teks biasa" +
                 (" (FORCED)" if conv_state.force_text else
                  " (tools ada tapi tidak dipanggil)" if has_tools else ""))
        response = resp_chat(actual_model, raw)

    # ═══ RETURN ═══
    _sep("RESPONSE → CLIENT")
    fr = response["choices"][0]["finish_reason"]
    log.info(f"  📦 model         = {actual_model}")
    log.info(f"  📦 provider      = {provider_used}")
    log.info(f"  📦 finish_reason = {fr}")

    # ← FIX: Log verifikasi content field
    msg_out = response["choices"][0]["message"]
    if "tool_calls" in msg_out:
        content_val = msg_out.get("content")
        content_present = "content" in msg_out
        log.info(f"  📐 content field = "                                   # ← FIX
                 f"{'present' if content_present else 'OMITTED'}, "
                 f"value = {repr(content_val)}")

    _log_json("RESPONSE", response)
    _sep()

    # ← FIX: Gunakan PicoResponse untuk serialisasi yang benar
    return PicoResponse(
        content=response,
        headers={
            "X-G4F-Provider": provider_used,
            "X-G4F-Model": actual_model,
        },
    )


# ══════════════════════════════════════════════════════════════
#  OTHER ROUTES
# ══════════════════════════════════════════════════════════════

@app.get("/v1/models")
@app.get("/models")
async def list_models():
    models = registry.list_models()
    return {
        "object": "list",
        "data": [{"id": m, "object": "model",
                  "created": 0, "owned_by": "g4f"} for m in models]
             + [{"id": "auto", "object": "model",
                 "created": 0, "owned_by": "bridge",
                 "description": "Auto-select best model + provider"}],
    }

@app.post("/v1/scan")
async def trigger_rescan():
    t0 = time.time()
    await registry.scan_all()
    elapsed = time.time() - t0
    st = registry.status()
    return {
        "status": "done",
        "duration_s": round(elapsed, 1),
        "providers_alive": sum(
            1 for v in st.values() if v["successes"] > 0),
        "providers_tool_capable": sum(
            1 for v in st.values() if v["tool_capable"]),
        "models": len(registry.list_models()),
    }

@app.get("/v1/providers/status")
async def provider_status():
    report = registry.status()
    return {
        "total": len(report),
        "healthy": sum(1 for v in report.values() if v["healthy"]),
        "tool_capable": sum(
            1 for v in report.values() if v["tool_capable"]),
        "providers": report,
    }

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "models": len(registry.list_models()),
        "tool_content_mode": TOOL_CONTENT_MODE,
        "ts": int(time.time()),
    }

@app.get("/")
async def root():
    st = registry.status()
    return {
        "service": "G4F API Bridge", "version": "5.0.0",
        "features": ["auto-model", "tool-scan", "anti-loop",
                      "null-content-fix"],
        "tool_content_mode": TOOL_CONTENT_MODE,
        "anti_loop_threshold": MAX_CONSECUTIVE_TOOLS,
        "providers_alive": sum(
            1 for v in st.values() if v["healthy"]),
        "providers_tool_capable": sum(
            1 for v in st.values() if v["tool_capable"]),
        "endpoints": {
            "chat": "POST /v1/chat/completions",
            "models": "GET /v1/models",
            "scan": "POST /v1/scan",
            "status": "GET /v1/providers/status",
        },
    }


# ══════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn

    addr = f"http://{HOST}:{PORT}"
    print(f"""
╔═══════════════════════════════════════════════════════╗
║            G4F  API  BRIDGE  v5.0                     ║
║   Null-Content Fix + Anti-Loop + Tool-Scan            ║
╠═══════════════════════════════════════════════════════╣
║  🌐 Server    : {addr:<37s}║
║  📖 Docs      : {(addr + '/docs'):<37s}║
║  📐 Content   : {TOOL_CONTENT_MODE:<37s}║
║  🛑 Loop limit: {(str(MAX_CONSECUTIVE_TOOLS) + ' rounds'):<37s}║
╚═══════════════════════════════════════════════════════╝""")

    uvicorn.run("server:app", host=HOST, port=PORT, log_level="warning")