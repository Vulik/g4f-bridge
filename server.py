"""
server.py
═════════
G4F API Bridge — OpenAI-Compatible Middleware + Function Calling Interceptor

Satu file. Satu endpoint utama: POST /v1/chat/completions.

Fitur:
  ✦ Input/Output 100% format OpenAI Chat Completions
  ✦ Function Calling Interceptor (g4f hanya teks → kita akali)
  ✦ JSON Extractor robust (handle markdown, multi-format)
  ✦ Logging detail setiap tahap (copy-paste untuk debugging)
  ✦ Error handling lengkap (timeout, empty, upstream error)

Jalankan:
  pip install fastapi uvicorn g4f
  python server.py
  → http://0.0.0.0:8820/docs
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
from typing import Any, Dict, List, Optional, Union

# ══════════════════════════════════════════════════════════════
#  DEPENDENCY CHECK
# ══════════════════════════════════════════════════════════════

try:
    import g4f
except ImportError:
    sys.exit(
        "\n  ❌  Library 'g4f' belum terinstall.\n"
        "     Jalankan: pip install g4f\n"
    )

from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field


# ══════════════════════════════════════════════════════════════
#  KONFIGURASI
# ══════════════════════════════════════════════════════════════

HOST        = os.getenv("HOST", "0.0.0.0")
PORT        = int(os.getenv("PORT", "8820"))
G4F_TIMEOUT = int(os.getenv("G4F_TIMEOUT", "120"))
LOG_LEVEL   = os.getenv("LOG_LEVEL", "DEBUG")


# ══════════════════════════════════════════════════════════════
#  LOGGING — Terstruktur agar mudah di-copy untuk debugging
# ══════════════════════════════════════════════════════════════

log = logging.getLogger("bridge")
log.setLevel(getattr(logging, LOG_LEVEL, logging.DEBUG))

_handler = logging.StreamHandler(sys.stdout)
_handler.setFormatter(logging.Formatter(
    fmt="%(asctime)s │ %(levelname)-7s │ %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
))
log.addHandler(_handler)
log.propagate = False


def _sep(title: str = "", w: int = 72):
    """Cetak separator garis di log."""
    if title:
        pad = max(1, (w - len(title) - 2) // 2)
        log.info(f"{'─' * pad} {title} {'─' * pad}")
    else:
        log.info("─" * w)


def _log_json(label: str, data: Any, max_chars: int = 3000):
    """Log object sebagai JSON indented (truncated jika terlalu panjang)."""
    try:
        text = json.dumps(data, ensure_ascii=False, indent=2)
    except (TypeError, ValueError):
        text = str(data)
    if len(text) > max_chars:
        text = text[:max_chars] + f"\n  … [dipotong, total {len(text)} char]"
    for line in text.split("\n"):
        log.debug(f"  {label:>12s} │ {line}")


# ══════════════════════════════════════════════════════════════
#  PYDANTIC — Model Request (OpenAI-Compatible)
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
    model: str = "gpt-4o-mini"
    messages: List[MessageIn]
    tools: Optional[List[ToolDef]] = None
    tool_choice: Optional[Any] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    top_p: Optional[float] = None
    stream: Optional[bool] = False

    model_config = {"extra": "allow"}


# ══════════════════════════════════════════════════════════════
#  UTILITAS UMUM
# ══════════════════════════════════════════════════════════════

def _gen_id(prefix: str = "chatcmpl") -> str:
    return f"{prefix}-{uuid.uuid4().hex[:12]}"


def extract_model_name(raw: str) -> str:
    """
    'openai/gpt-4o'  → 'gpt-4o'
    'google/gemini'   → 'gemini'
    'gpt-3.5-turbo'   → 'gpt-3.5-turbo'
    """
    return raw.split("/", 1)[-1] if "/" in raw else raw


# ══════════════════════════════════════════════════════════════
#  TOOL INTERCEPTOR — System Prompt Builder
# ══════════════════════════════════════════════════════════════

def build_tool_system_prompt(
    tools: List[ToolDef],
    tool_choice: Any = None,
) -> str:
    """
    Bangun system prompt rahasia yang mengajarkan model g4f
    cara 'berpura-pura' mendukung function calling.

    Prompt ini di-inject di awal messages sebelum dikirim ke g4f.
    """

    # ── Daftar tool lengkap ──
    tool_lines: list[str] = []
    for i, t in enumerate(tools, 1):
        fn = t.function
        params = json.dumps(fn.parameters, ensure_ascii=False, indent=2)
        tool_lines.append(
            f"  {i}. Nama  : {fn.name}\n"
            f"     Desc  : {fn.description or '-'}\n"
            f"     Params: {params}"
        )
    tools_block = "\n\n".join(tool_lines)
    tool_names  = [t.function.name for t in tools]

    # ── Instruksi berdasarkan tool_choice ──
    if tool_choice == "none":
        # Seharusnya sudah di-handle sebelum fungsi ini dipanggil
        return ""

    if isinstance(tool_choice, dict):
        # tool_choice = {"type": "function", "function": {"name": "xxx"}}
        forced = tool_choice.get("function", {}).get("name", "")
        force_line = (
            f"\n⚠️  WAJIB: Kamu HARUS memanggil tool '{forced}' "
            f"untuk pesan ini. Jangan jawab dengan teks biasa."
        )
    elif tool_choice == "required":
        force_line = (
            "\n⚠️  WAJIB: Kamu HARUS memanggil salah satu tool. "
            "Jangan jawab dengan teks biasa."
        )
    else:
        force_line = ""

    return (
        "═══ INSTRUKSI INTERNAL: FUNCTION CALLING ═══\n"
        "Kamu adalah AI assistant yang memiliki akses ke tool berikut:\n"
        "\n"
        f"{tools_block}\n"
        "\n"
        "ATURAN KETAT:\n"
        f"1. Jika kamu PERLU memanggil salah satu tool ({', '.join(tool_names)}) "
        "untuk menjawab user, balas HANYA dengan satu baris JSON murni "
        "dalam format ini — TANPA teks lain, TANPA markdown, TANPA penjelasan:\n"
        '\n'
        '   {"nama_alat": "<nama_fungsi>", "argumen": {<isi parameter>}}\n'
        '\n'
        '   Contoh: {"nama_alat": "get_weather", "argumen": {"lokasi": "Tokyo"}}\n'
        "\n"
        "2. Jika kamu TIDAK perlu tool, balas natural seperti biasa.\n"
        "3. JANGAN bungkus JSON dalam ``` code block.\n"
        "4. Pastikan JSON valid.\n"
        "5. Gunakan HANYA nama tool yang ada di daftar.\n"
        f"{force_line}\n"
        "═══ AKHIR INSTRUKSI ═══"
    )


def preprocess_messages(
    messages: List[MessageIn],
    tools: Optional[List[ToolDef]],
    tool_choice: Any = None,
) -> List[Dict[str, str]]:
    """
    Siapkan messages untuk g4f:

    1. Inject system prompt tool (jika tools ada & tool_choice ≠ "none")
    2. Konversi pesan role="tool" → role="user" (g4f tak kenal)
    3. Konversi pesan assistant+tool_calls → teks biasa
    4. Pastikan content selalu string
    """
    result: List[Dict[str, str]] = []

    # ── 1. Inject tool prompt ──
    has_tools  = tools is not None and len(tools) > 0
    skip_tools = (tool_choice == "none")

    if has_tools and not skip_tools:
        prompt = build_tool_system_prompt(tools, tool_choice)
        if prompt:
            result.append({"role": "system", "content": prompt})
            log.info("🔧 System prompt tool di-inject")

    # ── 2. Proses setiap message ──
    for msg in messages:
        role    = msg.role
        content = msg.content or ""

        # Handle content list (vision/multimodal)
        if isinstance(content, list):
            parts = []
            for p in content:
                if isinstance(p, dict) and p.get("type") == "text":
                    parts.append(p.get("text", ""))
                elif isinstance(p, str):
                    parts.append(p)
            content = "\n".join(parts)

        # Role: tool → user (g4f tak kenal role "tool")
        if role == "tool":
            tool_name = msg.name or "unknown"
            call_id   = msg.tool_call_id or "?"
            result.append({
                "role": "user",
                "content": (
                    f"[Hasil tool '{tool_name}' "
                    f"(call_id: {call_id})]:\n{content}"
                ),
            })
            log.debug(f"  📎 tool msg '{tool_name}' → user msg")

        # Role: assistant + tool_calls → plain text
        elif role == "assistant" and msg.tool_calls:
            descs = []
            for tc in msg.tool_calls:
                fn = tc.get("function", {}) if isinstance(tc, dict) else {}
                descs.append(f"{fn.get('name', '?')}({fn.get('arguments', '{}')})")
            result.append({
                "role": "assistant",
                "content": f"Saya memanggil: {', '.join(descs)}"
            })
            log.debug("  📎 assistant tool_calls → plain text")

        # Role: function (legacy)
        elif role == "function":
            fn_name = msg.name or "unknown"
            result.append({
                "role": "user",
                "content": f"[Hasil fungsi '{fn_name}']: {content}",
            })

        # Normal
        else:
            result.append({"role": role, "content": str(content)})

    return result


# ══════════════════════════════════════════════════════════════
#  JSON EXTRACTOR — Deteksi Tool Call dari Teks g4f
# ══════════════════════════════════════════════════════════════

def _strip_markdown(text: str) -> str:
    """Hapus marker code-block markdown: ```json ... ``` → isinya saja."""
    return re.sub(
        r"```(?:json|JSON)?\s*\n?(.*?)\n?\s*```",
        r"\1",
        text,
        flags=re.DOTALL,
    ).strip()


def _find_json_objects(text: str) -> List[dict]:
    """
    Temukan semua objek JSON {} valid di dalam teks.
    Menggunakan brace-matching yang sadar string literal.
    """
    results: list[dict] = []
    n = len(text)
    i = 0

    while i < n:
        if text[i] != "{":
            i += 1
            continue

        depth       = 0
        in_string   = False
        escape_next = False
        j           = i

        while j < n:
            ch = text[j]

            if escape_next:
                escape_next = False
                j += 1
                continue

            if ch == "\\" and in_string:
                escape_next = True
                j += 1
                continue

            if ch == '"':
                in_string = not in_string
            elif not in_string:
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        candidate = text[i : j + 1]
                        try:
                            obj = json.loads(candidate)
                            if isinstance(obj, dict):
                                results.append(obj)
                        except json.JSONDecodeError:
                            pass
                        break
            j += 1
        i += 1

    return results


def try_parse_tool_call(
    raw_text: str,
    tools: Optional[List[ToolDef]] = None,
) -> Optional[Dict[str, Any]]:
    """
    Coba ekstrak pemanggilan tool dari teks respons g4f.

    Strategi pencarian (berurutan):
      1. Parse seluruh teks sebagai JSON
      2. Hapus markdown code blocks, coba lagi
      3. Cari objek JSON di dalam teks (brace-matching)

    Validasi:
      - Harus punya key yang dikenal (nama_alat / name / tool_name / function)
      - Nama tool harus ada di daftar yang tersedia

    Returns:
        {"name": "tool_name", "arguments": {...}}  — jika terdeteksi
        None                                        — jika bukan tool call
    """
    if not raw_text or not raw_text.strip():
        return None

    # Kumpulkan nama tool valid
    valid_names: set[str] = set()
    if tools:
        valid_names = {t.function.name for t in tools}

    # ── Kumpulkan kandidat JSON ──
    cleaned    = _strip_markdown(raw_text)
    candidates = _find_json_objects(cleaned)

    # Fallback: coba parse seluruh teks
    if not candidates:
        try:
            obj = json.loads(cleaned)
            if isinstance(obj, dict):
                candidates = [obj]
            elif isinstance(obj, list) and obj:
                candidates = [o for o in obj if isinstance(o, dict)]
        except (json.JSONDecodeError, TypeError):
            pass

    if not candidates:
        log.debug("  🔍 Tidak ada JSON ditemukan dalam respons g4f")
        return None

    log.debug(f"  🔍 Ditemukan {len(candidates)} kandidat JSON")

    # ── Cek setiap kandidat ──
    # Mapping key yang dikenali → (name_key, args_key)
    KEY_PATTERNS = [
        ("nama_alat",     ["argumen", "arguments", "args", "params"]),
        ("name",          ["arguments", "argumen", "args", "params"]),
        ("tool_name",     ["arguments", "argumen", "args", "params"]),
        ("function",      ["parameters", "arguments", "argumen", "params"]),
        ("function_name", ["arguments", "argumen", "args", "params"]),
    ]

    for obj in candidates:
        for name_key, args_keys in KEY_PATTERNS:
            if name_key not in obj:
                continue

            fn_name = obj[name_key]
            if not isinstance(fn_name, str):
                continue

            # Validasi nama tool
            if valid_names and fn_name not in valid_names:
                log.debug(
                    f"  ⚠️  '{fn_name}' tidak ada di daftar tool: "
                    f"{valid_names}"
                )
                continue

            # Cari arguments
            arguments = {}
            for ak in args_keys:
                if ak in obj:
                    arguments = obj[ak]
                    break

            # arguments bisa berupa string JSON
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    arguments = {}

            if not isinstance(arguments, dict):
                arguments = {}

            log.info(
                f"  🎯 Tool call terdeteksi: "
                f"{fn_name}({json.dumps(arguments, ensure_ascii=False)})"
            )
            return {"name": fn_name, "arguments": arguments}

    log.debug("  🔍 Kandidat JSON ada tapi tidak cocok format tool call")
    return None


# ══════════════════════════════════════════════════════════════
#  RESPONSE BUILDERS — Format OpenAI Baku
# ══════════════════════════════════════════════════════════════

def build_chat_response(model: str, content: str) -> dict:
    """
    KONDISI A: Respons teks biasa.

    {
      "id": "chatcmpl-...",
      "object": "chat.completion",
      "created": ...,
      "model": "gpt-4o",
      "choices": [{
        "index": 0,
        "message": {"role": "assistant", "content": "teks"},
        "finish_reason": "stop"
      }],
      "usage": {...}
    }
    """
    return {
        "id": _gen_id(),
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": content,
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        },
    }


def build_tool_call_response(
    model: str,
    tool_name: str,
    arguments: dict,
) -> dict:
    """
    KONDISI B: Respons function/tool calling.

    ⚠️  content HARUS "" (kosong).
    ⚠️  arguments HARUS JSON-string (di-escape).
    ⚠️  finish_reason HARUS "tool_calls".

    {
      "id": "chatcmpl-...",
      "choices": [{
        "message": {
          "role": "assistant",
          "content": "",
          "tool_calls": [{
            "id": "call_...",
            "type": "function",
            "function": {
              "name": "get_weather",
              "arguments": "{\"lokasi\": \"Tokyo\"}"
            }
          }]
        },
        "finish_reason": "tool_calls"
      }]
    }
    """
    # arguments → JSON string yang di-escape
    args_string = json.dumps(arguments, ensure_ascii=False)

    return {
        "id": _gen_id(),
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "",           # HARUS kosong
                    "tool_calls": [
                        {
                            "id": _gen_id("call"),
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
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        },
    }


def build_error_response(
    message: str,
    err_type: str = "server_error",
    code: int = 500,
) -> dict:
    """Format error standar OpenAI."""
    return {
        "error": {
            "message": message,
            "type": err_type,
            "param": None,
            "code": code,
        }
    }


# ══════════════════════════════════════════════════════════════
#  G4F CALLER — Panggil g4f dengan timeout & fallback
# ══════════════════════════════════════════════════════════════

async def call_g4f(
    model: str,
    messages: List[Dict[str, str]],
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
) -> str:
    """
    Panggil g4f dan kembalikan teks respons.
    Coba create_async dulu; fallback ke sync di thread pool.
    """
    kwargs: Dict[str, Any] = {
        "model": model,
        "messages": messages,
    }
    if temperature is not None:
        kwargs["temperature"] = temperature
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens

    # ── Coba async ──
    try:
        result = await asyncio.wait_for(
            g4f.ChatCompletion.create_async(**kwargs),
            timeout=G4F_TIMEOUT,
        )
        if result:
            return str(result).strip()
    except (AttributeError, NotImplementedError, TypeError):
        # create_async tidak tersedia → fallback sync
        pass
    except asyncio.TimeoutError:
        raise TimeoutError(f"g4f timeout ({G4F_TIMEOUT}s)")

    # ── Fallback sync via thread pool ──
    loop = asyncio.get_running_loop()
    result = await asyncio.wait_for(
        loop.run_in_executor(
            None,
            lambda: g4f.ChatCompletion.create(**kwargs),
        ),
        timeout=G4F_TIMEOUT,
    )

    # Handle generator (beberapa provider kembalikan generator)
    if hasattr(result, "__iter__") and not isinstance(result, str):
        return "".join(str(chunk) for chunk in result).strip()

    return str(result).strip() if result else ""


# ══════════════════════════════════════════════════════════════
#  FASTAPI APPLICATION
# ══════════════════════════════════════════════════════════════

app = FastAPI(
    title="G4F Bridge",
    description="OpenAI-compatible bridge + Function Calling interceptor",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request/Response Logger Middleware ──

@app.middleware("http")
async def log_http(request: Request, call_next):
    """Log setiap HTTP request masuk dan response keluar."""
    t0 = time.time()
    method = request.method
    path   = request.url.path

    log.info(f"📡 ← {method} {path}")

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
    """
    Endpoint utama bridge.

    Alur:
      1. Log request masuk
      2. Ekstrak & bersihkan model name
      3. Deteksi tools → inject system prompt
      4. Preprocess messages
      5. Kirim ke g4f
      6. Parse respons (teks biasa vs tool call)
      7. Bangun & kirim respons format OpenAI
    """
    rid = _gen_id("req")

    # ═══════════════════════════════════════════════
    #  STEP 1 — LOG REQUEST MASUK
    # ═══════════════════════════════════════════════
    _sep(f"REQUEST {rid}")

    raw_model = req.model
    model     = extract_model_name(raw_model)
    has_tools = bool(req.tools and len(req.tools) > 0)

    log.info(f"  📨 model       : {raw_model} → '{model}'")
    log.info(f"  📨 messages    : {len(req.messages)} buah")
    log.info(f"  📨 tools       : {'YA (' + str(len(req.tools)) + ')' if has_tools else 'TIDAK'}")
    log.info(f"  📨 tool_choice : {req.tool_choice}")
    log.info(f"  📨 stream      : {req.stream}")
    log.info(f"  📨 temperature : {req.temperature}")
    log.info(f"  📨 max_tokens  : {req.max_tokens}")

    if has_tools:
        for t in req.tools:
            log.info(f"     🔧 {t.function.name} — {t.function.description or '-'}")

    # Log body ringkas
    _log_json("REQ.body", {
        "model": raw_model,
        "messages": [
            {
                "role": m.role,
                "content": (
                    str(m.content)[:120] + "…"
                    if m.content and len(str(m.content)) > 120
                    else m.content
                ),
                **({"tool_calls": "…"} if m.tool_calls else {}),
                **({"tool_call_id": m.tool_call_id} if m.tool_call_id else {}),
            }
            for m in req.messages
        ],
        "tools": [t.function.name for t in req.tools] if has_tools else None,
        "tool_choice": req.tool_choice,
    })

    # ═══════════════════════════════════════════════
    #  STEP 2 — PREPROCESS MESSAGES
    # ═══════════════════════════════════════════════
    _sep("PREPROCESSING")

    processed = preprocess_messages(
        messages=req.messages,
        tools=req.tools,
        tool_choice=req.tool_choice,
    )

    log.info(f"  ⚙️  Messages final: {len(processed)} buah")
    _log_json("PROCESSED", processed)

    # ═══════════════════════════════════════════════
    #  STEP 3 — PANGGIL G4F
    # ═══════════════════════════════════════════════
    _sep("G4F CALL")
    log.info(f"  📤 → g4f (model='{model}', timeout={G4F_TIMEOUT}s)")

    t0 = time.time()
    try:
        raw = await call_g4f(
            model=model,
            messages=processed,
            temperature=req.temperature,
            max_tokens=req.max_tokens,
        )
        elapsed_ms = (time.time() - t0) * 1000

    except TimeoutError as e:
        elapsed_ms = (time.time() - t0) * 1000
        log.error(f"  ⏱️  TIMEOUT setelah {elapsed_ms:.0f}ms — {e}")
        return JSONResponse(
            status_code=504,
            content=build_error_response(str(e), "timeout", 504),
        )
    except Exception as e:
        elapsed_ms = (time.time() - t0) * 1000
        log.error(
            f"  💥 ERROR setelah {elapsed_ms:.0f}ms — "
            f"{type(e).__name__}: {e}"
        )
        return JSONResponse(
            status_code=502,
            content=build_error_response(
                f"g4f error: {type(e).__name__}: {str(e)[:500]}",
                "upstream_error",
                502,
            ),
        )

    log.info(f"  📥 ← g4f merespons ({elapsed_ms:.0f}ms, {len(raw)} chars)")
    _log_json("G4F.raw", raw)

    if not raw:
        log.warning("  ⚠️  g4f mengembalikan respons KOSONG")
        return JSONResponse(
            status_code=502,
            content=build_error_response(
                "g4f returned empty response",
                "empty_response",
                502,
            ),
        )

    # ═══════════════════════════════════════════════
    #  STEP 4 — PARSE: Teks Biasa vs Tool Call?
    # ═══════════════════════════════════════════════
    _sep("PARSING")

    tool_call = None
    if has_tools:
        log.info("  🔍 Request punya tools → mencoba deteksi tool call…")
        tool_call = try_parse_tool_call(raw, req.tools)

    if tool_call:
        # ── KONDISI B: Tool Call ──
        log.info("  ✅ Terdeteksi sebagai TOOL CALL")
        log.info(f"     name      = {tool_call['name']}")
        log.info(
            f"     arguments = "
            f"{json.dumps(tool_call['arguments'], ensure_ascii=False)}"
        )

        response = build_tool_call_response(
            model=model,
            tool_name=tool_call["name"],
            arguments=tool_call["arguments"],
        )
    else:
        # ── KONDISI A: Teks Biasa ──
        if has_tools:
            log.info("  ℹ️  Tidak ada tool call → respons teks biasa")
        else:
            log.info("  ℹ️  Respons teks biasa (tanpa tools)")

        response = build_chat_response(model=model, content=raw)

    # ═══════════════════════════════════════════════
    #  STEP 5 — LOG & KIRIM RESPONSE
    # ═══════════════════════════════════════════════
    _sep("RESPONSE → CLIENT")

    finish = response["choices"][0]["finish_reason"]
    log.info(f"  📦 finish_reason = {finish}")
    _log_json("RESPONSE", response)

    _sep()  # penutup

    return JSONResponse(content=response)


# ══════════════════════════════════════════════════════════════
#  ROUTE: GET /v1/models
# ══════════════════════════════════════════════════════════════

@app.get("/v1/models")
@app.get("/models")
async def list_models():
    """Daftar model — kompatibel format OpenAI."""
    models = [
        "gpt-4o", "gpt-4o-mini", "gpt-4", "gpt-3.5-turbo",
    ]
    try:
        from g4f.models import ModelUtils
        models = list(ModelUtils.convert.keys())
    except Exception:
        pass

    return {
        "object": "list",
        "data": [
            {
                "id": m,
                "object": "model",
                "created": 0,
                "owned_by": "g4f-bridge",
            }
            for m in sorted(set(models))
        ],
    }


# ══════════════════════════════════════════════════════════════
#  ROUTE: Utilitas
# ══════════════════════════════════════════════════════════════

@app.get("/health")
async def health():
    return {"status": "ok", "ts": int(time.time())}


@app.get("/")
async def root():
    return {
        "service": "G4F API Bridge",
        "version": "1.0.0",
        "docs": f"http://{HOST}:{PORT}/docs",
        "endpoints": {
            "chat": "POST /v1/chat/completions",
            "models": "GET  /v1/models",
            "health": "GET  /health",
        },
    }


# ══════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn

    banner = f"""
╔════════════════════════════════════════════╗
║          G4F  API  BRIDGE  v1.0           ║
║     OpenAI ↔ g4f + Function Calling       ║
╠════════════════════════════════════════════╣
║  🌐 Server  : http://{HOST}:{PORT:<13s}   ║
║  📖 Docs    : http://{HOST}:{PORT}/docs{' ' * (4)}  ║
║  ⏱️  Timeout : {G4F_TIMEOUT}s{' ' * (27 - len(str(G4F_TIMEOUT)))}║
╚════════════════════════════════════════════╝"""
    print(banner)

    uvicorn.run(
        "server:app",
        host=HOST,
        port=PORT,
        reload=False,
        log_level="warning",
    )