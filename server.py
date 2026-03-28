"""
server.py
═════════
G4F API Bridge — Single-File, Auto-Model, Function Calling Interceptor

Fitur:
  ✦ Auto-model: model="auto" → otomatis cari provider/model yang hidup
  ✦ Fallback chain: jika provider A gagal, coba B, C, dst.
  ✦ Retry per provider dengan exponential backoff
  ✦ Function Calling interceptor (g4f hanya teks → kita akali)
  ✦ JSON extractor robust (markdown, brace-matching, multi-format)
  ✦ Health tracking: provider yang sering gagal turun prioritas
  ✦ Cooldown: provider bermasalah diistirahatkan sementara
  ✦ Log detail setiap tahap (siap copy-paste untuk debugging)

Jalankan:
  pip install fastapi uvicorn g4f
  python server.py
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
    sys.exit("\n  ❌ 'g4f' belum terinstall. → pip install g4f\n")

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field


# ══════════════════════════════════════════════════════════════
#  KONFIGURASI
# ══════════════════════════════════════════════════════════════

HOST          = os.getenv("HOST", "0.0.0.0")
PORT          = int(os.getenv("PORT", "8820"))
G4F_TIMEOUT   = int(os.getenv("G4F_TIMEOUT", "120"))
MAX_RETRIES   = int(os.getenv("MAX_RETRIES", "1"))
RETRY_BACKOFF = float(os.getenv("RETRY_BACKOFF", "2.0"))
COOLDOWN_BASE = int(os.getenv("COOLDOWN_BASE", "60"))
COOLDOWN_MAX  = int(os.getenv("COOLDOWN_MAX", "600"))
LOG_LEVEL     = os.getenv("LOG_LEVEL", "DEBUG")

PREFERRED_MODELS: list[str] = [
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


# ══════════════════════════════════════════════════════════════
#  LOGGING
# ══════════════════════════════════════════════════════════════

log = logging.getLogger("bridge")
log.setLevel(getattr(logging, LOG_LEVEL, logging.DEBUG))

_h = logging.StreamHandler(sys.stdout)
_h.setFormatter(logging.Formatter(
    "%(asctime)s │ %(levelname)-7s │ %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
))
log.addHandler(_h)
log.propagate = False


def _sep(title: str = "", width: int = 72):
    if title:
        pad = max(1, (width - len(title) - 2) // 2)
        log.info(f"{'─' * pad} {title} {'─' * pad}")
    else:
        log.info("─" * width)


def _log_json(label: str, data: Any, limit: int = 3000):
    try:
        txt = json.dumps(data, ensure_ascii=False, indent=2)
    except (TypeError, ValueError):
        txt = str(data)
    if len(txt) > limit:
        txt = txt[:limit] + f"\n  … [truncated, {len(txt)} chars total]"
    for line in txt.split("\n"):
        log.debug(f"  {label:>14s} │ {line}")


# ══════════════════════════════════════════════════════════════
#  PROVIDER REGISTRY — Auto-Discovery + Health Tracking
# ══════════════════════════════════════════════════════════════

@dataclass
class HealthRecord:
    """Track runtime health for one provider."""
    successes: int = 0
    failures: int = 0
    consecutive_fails: int = 0
    total_latency_ms: float = 0.0
    last_success: float = 0.0
    last_failure: float = 0.0

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
            return 0.5  # untested → netral
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
    """
    Discovers all (model, provider) pairs from g4f internals,
    tracks runtime health, builds fallback chains.
    """

    def __init__(self):
        # model_name → [(provider_name, provider_class), ...]
        self._model_map: Dict[str, List[Tuple[str, Any]]] = {}
        # provider_name → HealthRecord
        self._health: Dict[str, HealthRecord] = {}

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
            f"{len(self._health)} provider unik, "
            f"{count} pasangan"
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

    # ── chain building ──────────────────────────────

    def get_chain(
        self, model: str
    ) -> List[Tuple[str, str, Any]]:
        """
        Return [(model, provider_name, provider_class_or_None), ...]
        sorted by score, ready for fallback iteration.
        """
        if model.lower() == "auto":
            return self._auto_chain()
        return self._model_chain(model)

    def _auto_chain(self) -> List[Tuple[str, str, Any]]:
        """Pick best (model, provider) across all models."""
        scored: list[tuple[float, str, str, Any]] = []

        for mdl in PREFERRED_MODELS:
            pairs = self._model_map.get(mdl, [])
            # priority bonus: higher for earlier models
            try:
                idx = PREFERRED_MODELS.index(mdl)
                bonus = 0.25 * (1.0 - idx / len(PREFERRED_MODELS))
            except ValueError:
                bonus = 0.0

            for pn, pcls in pairs:
                h = self._health.get(pn, HealthRecord())
                if not h.is_healthy:
                    continue
                scored.append((h.score + bonus, mdl, pn, pcls))

        # add remaining models (not in priority list)
        for mdl, pairs in self._model_map.items():
            if mdl in PREFERRED_MODELS:
                continue
            for pn, pcls in pairs:
                h = self._health.get(pn, HealthRecord())
                if not h.is_healthy:
                    continue
                scored.append((h.score, mdl, pn, pcls))

        scored.sort(key=lambda x: x[0], reverse=True)

        # deduplicate and limit
        chain: list[tuple[str, str, Any]] = []
        seen: set[tuple[str, str]] = set()
        for _, mdl, pn, pcls in scored:
            key = (mdl, pn)
            if key in seen:
                continue
            seen.add(key)
            chain.append((mdl, pn, pcls))
            if len(chain) >= 12:
                break

        # final safety net: g4f tanpa provider spesifik
        chain.append(("gpt-4o-mini", "g4f-auto", None))

        if chain:
            log.info(
                f"🎯 Auto chain: {len(chain)} candidates, "
                f"top = {chain[0][1]} ({chain[0][0]})"
            )
        return chain

    def _model_chain(self, model: str) -> List[Tuple[str, str, Any]]:
        """Build chain for a specific model."""
        pairs = self._model_map.get(model)

        # fuzzy match
        if not pairs:
            for key in self._model_map:
                if model in key or key in model:
                    pairs = self._model_map[key]
                    model = key  # use canonical name
                    log.info(f"🔍 Fuzzy: '{model}' matched")
                    break

        chain: list[tuple[str, str, Any]] = []
        if pairs:
            ranked = sorted(
                pairs,
                key=lambda x: self._health.get(x[0], HealthRecord()).score,
                reverse=True,
            )
            for pn, pcls in ranked:
                h = self._health.get(pn, HealthRecord())
                if h.is_healthy:
                    chain.append((model, pn, pcls))

        # fallback: no specific provider
        chain.append((model, "g4f-auto", None))
        return chain

    # ── health reporting ────────────────────────────

    def record_success(self, pname: str, latency_ms: float = 0.0):
        self._health.setdefault(pname, HealthRecord()).record_ok(latency_ms)

    def record_failure(self, pname: str):
        self._health.setdefault(pname, HealthRecord()).record_fail()

    def list_models(self) -> list[str]:
        return sorted(self._model_map.keys())

    def status(self) -> dict:
        return {
            pn: {
                "healthy": h.is_healthy,
                "score": round(h.score, 3),
                "successes": h.successes,
                "failures": h.failures,
                "consecutive_fails": h.consecutive_fails,
                "avg_latency_ms": round(h.avg_latency, 1),
            }
            for pn, h in sorted(self._health.items())
        }


# global registry
registry = ProviderRegistry()


# ══════════════════════════════════════════════════════════════
#  PYDANTIC — Request Models
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
    """'openai/gpt-4o' → 'gpt-4o'"""
    return raw.split("/", 1)[-1] if "/" in raw else raw


# ══════════════════════════════════════════════════════════════
#  TOOL INTERCEPTOR — System Prompt Builder
# ══════════════════════════════════════════════════════════════

def build_tool_prompt(
    tools: List[ToolDef],
    tool_choice: Any = None,
) -> str:
    """
    Bangun system prompt rahasia yang mengajarkan model g4f
    cara 'berpura-pura' mendukung function calling.
    """
    blocks = []
    for i, t in enumerate(tools, 1):
        fn = t.function
        p  = json.dumps(fn.parameters, ensure_ascii=False, indent=2)
        blocks.append(
            f"  {i}. Nama  : {fn.name}\n"
            f"     Desc  : {fn.description or '-'}\n"
            f"     Params:\n{p}"
        )
    tools_text = "\n\n".join(blocks)
    names      = [t.function.name for t in tools]

    force = ""
    if isinstance(tool_choice, dict):
        forced = tool_choice.get("function", {}).get("name", "")
        force = (
            f"\n⚠️ WAJIB panggil tool '{forced}'. "
            "Jangan jawab teks biasa."
        )
    elif tool_choice == "required":
        force = (
            "\n⚠️ WAJIB panggil salah satu tool. "
            "Jangan jawab teks biasa."
        )

    return (
        "═══ INSTRUKSI INTERNAL: FUNCTION CALLING ═══\n"
        "Kamu punya akses ke tool berikut:\n\n"
        f"{tools_text}\n\n"
        "ATURAN:\n"
        f"• Jika PERLU memanggil tool ({', '.join(names)}), "
        "balas HANYA dengan satu baris JSON murni:\n"
        '  {"nama_alat": "<nama>", "argumen": {<parameter>}}\n\n'
        '  Contoh: {"nama_alat": "get_weather", "argumen": {"lokasi": "Tokyo"}}\n\n'
        "• JANGAN bungkus dalam ``` code block.\n"
        "• Jika TIDAK perlu tool, jawab natural.\n"
        "• Gunakan HANYA tool dari daftar.\n"
        f"{force}\n"
        "═══ AKHIR INSTRUKSI ═══"
    )


def preprocess_messages(
    messages: List[MessageIn],
    tools: Optional[List[ToolDef]],
    tool_choice: Any = None,
) -> List[Dict[str, str]]:
    """
    Siapkan messages untuk g4f:
    - Inject tool prompt jika tools ada
    - role="tool"       → role="user"
    - assistant+tool_calls → plain text
    - content selalu str
    """
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

        # list content → string
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
                "content": f"[Hasil tool '{name}' (id:{cid})]:\n{content}",
            })
        elif role == "assistant" and msg.tool_calls:
            descs = []
            for tc in msg.tool_calls:
                fn = tc.get("function", {}) if isinstance(tc, dict) else {}
                descs.append(f"{fn.get('name','?')}({fn.get('arguments','{}')})")
            result.append({
                "role": "assistant",
                "content": f"Saya memanggil: {', '.join(descs)}",
            })
        elif role == "function":
            result.append({
                "role": "user",
                "content": f"[Hasil fungsi '{msg.name or '?'}']: {content}",
            })
        else:
            result.append({"role": role, "content": str(content)})

    return result


# ══════════════════════════════════════════════════════════════
#  JSON EXTRACTOR — Deteksi Tool Call dari Teks g4f
# ══════════════════════════════════════════════════════════════

def _strip_md(text: str) -> str:
    return re.sub(
        r"```(?:json|JSON)?\s*\n?(.*?)\n?\s*```",
        r"\1", text, flags=re.DOTALL,
    ).strip()


def _find_jsons(text: str) -> list[dict]:
    """Brace-matching JSON extractor, handles escaped strings."""
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


# ── KEY PATTERNS for tool call detection ──
_NAME_KEYS = ["nama_alat", "name", "tool_name", "function", "function_name"]
_ARGS_KEYS = ["argumen", "arguments", "args", "params", "parameters"]


def try_parse_tool_call(
    raw: str,
    tools: Optional[List[ToolDef]] = None,
) -> Optional[Dict[str, Any]]:
    """
    Coba deteksi tool call JSON dari teks g4f.
    Return {"name": ..., "arguments": {...}} atau None.
    """
    if not raw or not raw.strip():
        return None

    valid_names = {t.function.name for t in tools} if tools else set()

    cleaned    = _strip_md(raw)
    candidates = _find_jsons(cleaned)

    # fallback: parse seluruh teks
    if not candidates:
        try:
            obj = json.loads(cleaned)
            if isinstance(obj, dict):
                candidates = [obj]
        except (json.JSONDecodeError, TypeError):
            pass

    if not candidates:
        return None

    log.debug(f"  🔍 {len(candidates)} JSON candidate(s) ditemukan")

    for obj in candidates:
        for nk in _NAME_KEYS:
            if nk not in obj:
                continue
            fn_name = obj[nk]
            if not isinstance(fn_name, str):
                continue
            if valid_names and fn_name not in valid_names:
                log.debug(f"  ⚠️ '{fn_name}' not in valid tools {valid_names}")
                continue

            # find arguments
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

            log.info(f"  🎯 Tool call: {fn_name}({json.dumps(arguments, ensure_ascii=False)})")
            return {"name": fn_name, "arguments": arguments}

    return None


# ══════════════════════════════════════════════════════════════
#  RESPONSE BUILDERS — Format OpenAI Baku
# ══════════════════════════════════════════════════════════════

def resp_chat(model: str, content: str) -> dict:
    """KONDISI A: Teks biasa."""
    return {
        "id": _id(), "object": "chat.completion",
        "created": int(time.time()), "model": model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": content},
            "finish_reason": "stop",
        }],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


def resp_tool(model: str, tool_name: str, arguments: dict) -> dict:
    """KONDISI B: Tool call.  content="" , finish_reason="tool_calls"."""
    return {
        "id": _id(), "object": "chat.completion",
        "created": int(time.time()), "model": model,
        "choices": [{
            "index": 0,
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
            "finish_reason": "tool_calls",
        }],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


def resp_error(msg: str, etype: str = "server_error", code: int = 500) -> dict:
    return {"error": {"message": msg, "type": etype, "param": None, "code": code}}


# ══════════════════════════════════════════════════════════════
#  G4F CALLER — Raw → Retry → Fallback
# ══════════════════════════════════════════════════════════════

async def _raw_call(
    model: str,
    messages: list[dict],
    provider: Any | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> str:
    """Satu kali panggilan ke g4f (async atau sync fallback)."""
    kw: dict[str, Any] = {"model": model, "messages": messages}
    if provider is not None:
        kw["provider"] = provider
    if temperature is not None:
        kw["temperature"] = temperature
    if max_tokens is not None:
        kw["max_tokens"] = max_tokens

    # async
    try:
        r = await g4f.ChatCompletion.create_async(**kw)
        if r:
            return str(r).strip()
    except (AttributeError, NotImplementedError, TypeError):
        pass

    # sync di threadpool
    loop = asyncio.get_running_loop()
    r = await loop.run_in_executor(
        None, lambda: g4f.ChatCompletion.create(**kw),
    )
    if hasattr(r, "__iter__") and not isinstance(r, str):
        return "".join(str(c) for c in r).strip()
    return str(r).strip() if r else ""


async def _call_with_retry(
    model: str,
    messages: list[dict],
    provider: Any | None,
    pname: str,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> str:
    """
    Panggil satu provider dengan retry.
    Retry ke-N menunggu RETRY_BACKOFF × 2^(N-1) detik.
    """
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

            if not text:
                raise ValueError("empty response")

            log.info(f"       ✅ {pname} OK (attempt {attempt+1}, {ms:.0f}ms, {len(text)} chars)")
            return text

        except asyncio.TimeoutError:
            last_err = TimeoutError(f"{pname} timeout {G4F_TIMEOUT}s")
            log.warning(f"       ⏱️ {pname} attempt {attempt+1}: timeout")

        except Exception as exc:
            last_err = exc
            log.warning(f"       ❌ {pname} attempt {attempt+1}: {exc}")

    raise last_err


async def call_g4f_smart(
    model_raw: str,
    messages: list[dict],
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> Tuple[str, str, str]:
    """
    Panggil g4f dengan fallback chain.

    Returns: (response_text, actual_model, provider_name)
    Raises:  HTTPException 503 jika semua gagal
    """
    model = clean_model(model_raw)
    chain = registry.get_chain(model)

    log.info(f"  🔗 Fallback chain: {len(chain)} candidates")
    for i, (m, pn, _) in enumerate(chain[:5]):
        log.info(f"     {i+1}. {pn} ({m})")
    if len(chain) > 5:
        log.info(f"     … dan {len(chain)-5} lagi")

    last_err = ""

    for mdl, pname, pcls in chain:
        log.info(f"  ⏩ [{mdl}] trying: {pname}")
        try:
            t0 = time.time()
            text = await _call_with_retry(
                model=mdl,
                messages=messages,
                provider=pcls,
                pname=pname,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            latency = (time.time() - t0) * 1000
            registry.record_success(pname, latency)
            return text, mdl, pname

        except Exception as exc:
            registry.record_failure(pname)
            last_err = f"{pname}({mdl}): {exc}"
            log.warning(f"  ⛔ {pname}({mdl}) failed completely: {exc}")

    raise HTTPException(
        status_code=503,
        detail=resp_error(
            f"Semua provider gagal untuk model='{model_raw}'. "
            f"Terakhir: {last_err}",
            "all_providers_failed", 503,
        ),
    )


# ══════════════════════════════════════════════════════════════
#  FASTAPI APP
# ══════════════════════════════════════════════════════════════

app = FastAPI(
    title="G4F Bridge",
    description="OpenAI-compatible bridge + Auto-Model + Function Calling",
    version="2.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    t0     = time.time()
    method = request.method
    path   = request.url.path
    log.info(f"📡 ← {method} {path}")

    response = await call_next(request)

    ms    = (time.time() - t0) * 1000
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

    # ═══ STEP 1 — LOG REQUEST ═══
    _sep(f"REQUEST {rid}")
    model_clean = clean_model(req.model)
    is_auto = model_clean.lower() == "auto"

    log.info(f"  📨 model       : {req.model} → '{model_clean}'" +
             (" (AUTO-SELECT)" if is_auto else ""))
    log.info(f"  📨 messages    : {len(req.messages)}")
    log.info(f"  📨 tools       : {'YA (' + str(len(req.tools)) + ')' if has_tools else 'TIDAK'}")
    log.info(f"  📨 tool_choice : {req.tool_choice}")
    log.info(f"  📨 temperature : {req.temperature}")
    log.info(f"  📨 max_tokens  : {req.max_tokens}")
    if has_tools:
        for t in req.tools:
            log.info(f"     🔧 {t.function.name} — {t.function.description or '-'}")

    _log_json("REQ.body", {
        "model": req.model,
        "messages": [
            {"role": m.role,
             "content": (str(m.content)[:120] + "…" if m.content and len(str(m.content)) > 120 else m.content),
             **({"tool_call_id": m.tool_call_id} if m.tool_call_id else {})}
            for m in req.messages
        ],
        "tools": [t.function.name for t in req.tools] if has_tools else None,
        "tool_choice": req.tool_choice,
    })

    # ═══ STEP 2 — PREPROCESS ═══
    _sep("PREPROCESSING")
    processed = preprocess_messages(req.messages, req.tools, req.tool_choice)
    log.info(f"  ⚙️  Messages final: {len(processed)}")
    _log_json("PROCESSED", processed)

    # ═══ STEP 3 — CALL G4F (auto / specific) ═══
    _sep("G4F CALL")
    log.info(f"  📤 → g4f (model='{model_clean}', timeout={G4F_TIMEOUT}s)")

    try:
        raw, actual_model, provider_used = await call_g4f_smart(
            model_raw=model_clean,
            messages=processed,
            temperature=req.temperature,
            max_tokens=req.max_tokens,
        )
    except HTTPException:
        raise
    except Exception as exc:
        log.error(f"  💥 Unexpected: {type(exc).__name__}: {exc}")
        return JSONResponse(
            status_code=502,
            content=resp_error(f"g4f error: {exc}", "upstream_error", 502),
        )

    log.info(f"  📥 ← g4f via {provider_used} (model={actual_model}, {len(raw)} chars)")
    _log_json("G4F.raw", raw)

    if not raw:
        return JSONResponse(
            status_code=502,
            content=resp_error("g4f returned empty response", "empty_response", 502),
        )

    # ═══ STEP 4 — PARSE: teks biasa vs tool call ═══
    _sep("PARSING")

    tool_call = None
    if has_tools:
        log.info("  🔍 Request punya tools → mencoba deteksi tool call…")
        tool_call = try_parse_tool_call(raw, req.tools)

    if tool_call:
        log.info("  ✅ KONDISI B: Tool Call")
        log.info(f"     name      = {tool_call['name']}")
        log.info(f"     arguments = {json.dumps(tool_call['arguments'], ensure_ascii=False)}")
        response = resp_tool(actual_model, tool_call["name"], tool_call["arguments"])
    else:
        log.info(f"  ℹ️  KONDISI A: Teks biasa" +
                 (" (tools tersedia tapi tidak dipanggil)" if has_tools else ""))
        response = resp_chat(actual_model, raw)

    # ═══ STEP 5 — RETURN ═══
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
#  ROUTE: MODEL LIST
# ══════════════════════════════════════════════════════════════

@app.get("/v1/models")
@app.get("/models")
async def list_models():
    models = registry.list_models()
    return {
        "object": "list",
        "data": [
            {"id": m, "object": "model", "created": 0, "owned_by": "g4f-bridge"}
            for m in models
        ] + [
            {"id": "auto", "object": "model", "created": 0,
             "owned_by": "g4f-bridge",
             "description": "Otomatis pilih model + provider terbaik"},
        ],
    }


# ══════════════════════════════════════════════════════════════
#  ROUTE: PROVIDER STATUS  (kirim output ini untuk debugging)
# ══════════════════════════════════════════════════════════════

@app.get("/v1/providers/status")
async def provider_status():
    report = registry.status()
    healthy = sum(1 for v in report.values() if v["healthy"])
    return {
        "total": len(report),
        "healthy": healthy,
        "providers": report,
    }


# ══════════════════════════════════════════════════════════════
#  ROUTE: HEALTH / ROOT
# ══════════════════════════════════════════════════════════════

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "models": len(registry.list_models()),
        "ts": int(time.time()),
    }

@app.get("/")
async def root():
    return {
        "service": "G4F API Bridge",
        "version": "2.0.0",
        "auto_model": True,
        "models_available": len(registry.list_models()),
        "endpoints": {
            "chat":     "POST /v1/chat/completions",
            "models":   "GET  /v1/models",
            "status":   "GET  /v1/providers/status",
            "health":   "GET  /health",
            "docs":     "GET  /docs",
        },
    }


# ══════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn

    addr = f"http://{HOST}:{PORT}"

    print(f"""
╔════════════════════════════════════════════════╗
║          G4F  API  BRIDGE  v2.0               ║
║   OpenAI ↔ g4f  +  Auto-Model  +  Tools      ║
╠════════════════════════════════════════════════╣
║  🌐 Server  : {addr:<32s} ║
║  📖 Docs    : {addr + '/docs':<32s} ║
║  ⏱️  Timeout : {G4F_TIMEOUT:<32} ║
║  🔄 Retries : {MAX_RETRIES:<32} ║
║  📋 Models  : {len(registry.list_models()):<32} ║
╚════════════════════════════════════════════════╝""")

    uvicorn.run("server:app", host=HOST, port=PORT, log_level="warning")