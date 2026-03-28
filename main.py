"""
main.py
~~~~~~~
API Bridge utama yang:
  • Menerima POST /v1/chat/completions (format OpenAI)
  • Mencari provider terbaik via ProviderManager
  • Meneruskan ke g4f dengan fallback otomatis
  • Mengembalikan respons dalam format OpenAI

Mendukung mode streaming (SSE) dan non-streaming.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from typing import Any, AsyncGenerator, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

import g4f

from provider_manager import ProviderManager
from schemas import (
    ChatCompletionChunk,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    Choice,
    DeltaMessage,
    StreamChoice,
    Usage,
)

# ══════════════════════════════════════════════════
#  SETUP
# ══════════════════════════════════════════════════

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-7s │ %(name)s │ %(message)s",
)
log = logging.getLogger("g4f-bridge")

manager = ProviderManager()

PROVIDER_TIMEOUT = 60        # detik per percobaan provider


app = FastAPI(
    title="G4F API Bridge",
    description="Menjembatani PicoClaw ↔ g4f dengan format OpenAI-compatible",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ══════════════════════════════════════════════════
#  HELPER: PANGGIL G4F
# ══════════════════════════════════════════════════

def _build_kwargs(
    model: str,
    messages: list[dict],
    provider: Any | None,
    temperature: float | None,
    max_tokens: int | None,
    stream: bool = False,
) -> dict:
    """Menyusun kwargs untuk g4f.ChatCompletion.create / create_async."""
    kw: dict[str, Any] = {"model": model, "messages": messages}
    if provider is not None:
        kw["provider"] = provider
    if temperature is not None:
        kw["temperature"] = temperature
    if max_tokens is not None:
        kw["max_tokens"] = max_tokens
    if stream:
        kw["stream"] = True
    return kw


async def _call_g4f(
    model: str,
    messages: list[dict],
    provider: Any | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> str:
    """
    Panggil g4f (non-streaming).
    Coba async terlebih dahulu; jika tidak tersedia, jalankan sync di thread.
    """
    kw = _build_kwargs(model, messages, provider, temperature, max_tokens)

    # Coba versi async (g4f ≥ 0.3)
    try:
        return await g4f.ChatCompletion.create_async(**kw)
    except (AttributeError, NotImplementedError, TypeError):
        pass

    # Fallback: sync di executor agar tidak memblokir event loop
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        None, lambda: g4f.ChatCompletion.create(**kw)
    )


async def _stream_g4f(
    model: str,
    messages: list[dict],
    provider: Any | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> AsyncGenerator[str, None]:
    """
    Streaming dari g4f.
    g4f.ChatCompletion.create(stream=True) mengembalikan sync generator,
    jadi kita jembatani lewat asyncio.Queue agar FastAPI bisa async-iterate.
    """
    kw = _build_kwargs(
        model, messages, provider, temperature, max_tokens, stream=True
    )
    queue: asyncio.Queue[tuple[str, Any]] = asyncio.Queue()
    loop = asyncio.get_running_loop()

    def _producer():
        try:
            resp = g4f.ChatCompletion.create(**kw)
            # resp bisa berupa string (provider tak support stream)
            # atau generator
            if isinstance(resp, str):
                loop.call_soon_threadsafe(queue.put_nowait, ("chunk", resp))
            else:
                for token in resp:
                    if token:
                        loop.call_soon_threadsafe(
                            queue.put_nowait, ("chunk", str(token))
                        )
            loop.call_soon_threadsafe(queue.put_nowait, ("done", None))
        except Exception as exc:
            loop.call_soon_threadsafe(queue.put_nowait, ("error", exc))

    # Jalankan producer di thread pool
    loop.run_in_executor(None, _producer)

    while True:
        kind, payload = await queue.get()
        if kind == "done":
            return
        if kind == "error":
            raise payload
        yield payload


# ══════════════════════════════════════════════════
#  CORE: FALLBACK CHAIN
# ══════════════════════════════════════════════════

async def _complete_with_fallback(
    model: str,
    messages: list[dict],
    temperature: float | None,
    max_tokens: int | None,
) -> tuple[str, str]:
    """
    Coba setiap provider yang kompatibel secara berurutan.
    Jika semua gagal, coba tanpa spesifikasi provider (g4f auto-select).
    Mengembalikan (response_text, provider_name).
    """
    ranked = manager.get_ranked_providers(model)
    # Tambahkan auto-select sebagai jaring pengaman terakhir
    attempts: list[tuple[str, Any]] = [*ranked, ("auto-select", None)]

    last_err = ""

    for pname, pcls in attempts:
        try:
            log.info(f"⏩  [{model}] mencoba provider: {pname}")

            text = await asyncio.wait_for(
                _call_g4f(model, messages, pcls, temperature, max_tokens),
                timeout=PROVIDER_TIMEOUT,
            )

            if not text or not str(text).strip():
                raise ValueError("Respons kosong dari provider")

            manager.report_success(pname)
            log.info(f"✅  [{model}] berhasil via {pname}")
            return str(text).strip(), pname

        except asyncio.TimeoutError:
            manager.report_failure(pname, "timeout")
            last_err = f"{pname}: timeout ({PROVIDER_TIMEOUT}s)"
            log.warning(f"⏱️  [{model}] {pname} timeout")

        except Exception as exc:
            manager.report_failure(pname, str(exc))
            last_err = f"{pname}: {exc}"
            log.warning(f"❌  [{model}] {pname} gagal — {exc}")

    raise HTTPException(
        status_code=503,
        detail={
            "error": "all_providers_failed",
            "model": model,
            "last_error": last_err,
            "message": f"Semua provider gagal untuk model '{model}'.",
        },
    )


async def _stream_with_fallback(
    model: str,
    messages: list[dict],
    temperature: float | None,
    max_tokens: int | None,
) -> tuple[AsyncGenerator[str, None], str]:
    """
    Sama seperti _complete_with_fallback tapi untuk streaming.
    Menarik chunk pertama untuk memverifikasi provider bekerja
    sebelum mulai streaming ke klien.
    """
    ranked = manager.get_ranked_providers(model)
    if not ranked:
        # Coba auto-select
        ranked = [("auto-select", None)]

    last_err = ""

    for pname, pcls in ranked:
        try:
            log.info(f"⏩  [{model}] stream: mencoba {pname}")
            gen = _stream_g4f(model, messages, pcls, temperature, max_tokens)

            # Tarik chunk pertama sebagai validasi
            first_chunk = await asyncio.wait_for(
                gen.__anext__(), timeout=PROVIDER_TIMEOUT
            )
            manager.report_success(pname)
            log.info(f"✅  [{model}] stream berhasil via {pname}")

            # Gabungkan chunk pertama kembali ke generator
            async def _chain(first: str, rest: AsyncGenerator) -> AsyncGenerator:
                yield first
                async for c in rest:
                    yield c

            return _chain(first_chunk, gen), pname

        except StopAsyncIteration:
            manager.report_failure(pname, "empty stream")
            last_err = f"{pname}: stream kosong"
            log.warning(f"❌  [{model}] {pname} stream kosong")

        except asyncio.TimeoutError:
            manager.report_failure(pname, "timeout")
            last_err = f"{pname}: timeout"
            log.warning(f"⏱️  [{model}] {pname} stream timeout")

        except Exception as exc:
            manager.report_failure(pname, str(exc))
            last_err = f"{pname}: {exc}"
            log.warning(f"❌  [{model}] {pname} stream gagal — {exc}")

    raise HTTPException(
        status_code=503,
        detail={
            "error": "all_stream_providers_failed",
            "model": model,
            "last_error": last_err,
        },
    )


# ══════════════════════════════════════════════════
#  UTILITAS
# ══════════════════════════════════════════════════

def _sse(chunk: ChatCompletionChunk) -> str:
    """Format satu SSE data line."""
    return f"data: {chunk.model_dump_json()}\n\n"


def _estimate_usage(messages: list[dict], completion: str) -> Usage:
    """Estimasi kasar jumlah token (word-based)."""
    pt = sum(len((m.get("content") or "").split()) for m in messages)
    ct = len(completion.split())
    return Usage(prompt_tokens=pt, completion_tokens=ct, total_tokens=pt + ct)


# ══════════════════════════════════════════════════
#  ROUTES
# ══════════════════════════════════════════════════

@app.post("/v1/chat/completions")
@app.post("/chat/completions")
async def chat_completions(req: ChatCompletionRequest):
    """
    Endpoint utama — menerima request format OpenAI,
    meneruskan ke g4f, mengembalikan format OpenAI.
    """
    messages = [
        {"role": m.role, "content": m.content or ""}
        for m in req.messages
    ]

    log.info(
        f"📨  Request masuk — model={req.model}, "
        f"messages={len(messages)}, stream={req.stream}"
    )

    # ── STREAMING ────────────────────────────────
    if req.stream:
        token_gen, pname = await _stream_with_fallback(
            req.model, messages, req.temperature, req.max_tokens
        )
        cid = f"chatcmpl-{uuid.uuid4().hex[:16]}"
        ts = int(time.time())

        async def sse_generator():
            try:
                # Chunk pertama: role
                yield _sse(ChatCompletionChunk(
                    id=cid, created=ts, model=req.model,
                    choices=[StreamChoice(
                        delta=DeltaMessage(role="assistant")
                    )],
                ))

                # Chunk konten
                async for token in token_gen:
                    yield _sse(ChatCompletionChunk(
                        id=cid, created=ts, model=req.model,
                        choices=[StreamChoice(
                            delta=DeltaMessage(content=token)
                        )],
                    ))

                # Chunk penutup
                yield _sse(ChatCompletionChunk(
                    id=cid, created=ts, model=req.model,
                    choices=[StreamChoice(
                        delta=DeltaMessage(),
                        finish_reason="stop",
                    )],
                ))
                yield "data: [DONE]\n\n"

            except Exception as exc:
                log.error(f"Stream error: {exc}")
                err = {"error": {"message": str(exc), "type": "stream_error"}}
                yield f"data: {json.dumps(err)}\n\n"

        return StreamingResponse(
            sse_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-G4F-Provider": pname,
            },
        )

    # ── NON-STREAMING ───────────────────────────
    content, pname = await _complete_with_fallback(
        req.model, messages, req.temperature, req.max_tokens
    )

    response = ChatCompletionResponse(
        model=req.model,
        choices=[
            Choice(
                message=ChatMessage(role="assistant", content=content),
                finish_reason="stop",
            )
        ],
        usage=_estimate_usage(messages, content),
    )

    return JSONResponse(
        content=response.model_dump(),
        headers={"X-G4F-Provider": pname},
    )


# ── MODEL LIST (OpenAI-compatible) ──────────────

@app.get("/v1/models")
@app.get("/models")
async def list_models():
    """Daftar semua model yang tersedia di g4f."""
    return {
        "object": "list",
        "data": [
            {
                "id": m,
                "object": "model",
                "created": 0,
                "owned_by": "g4f",
            }
            for m in manager.list_models()
        ],
    }


# ── HEALTH & STATUS ─────────────────────────────

@app.get("/health")
async def health():
    """Health check sederhana."""
    models = manager.list_models()
    return {
        "status": "healthy",
        "models_available": len(models),
        "timestamp": int(time.time()),
    }


@app.get("/v1/providers/status")
async def provider_status():
    """Laporan detail kesehatan setiap provider."""
    report = manager.status_report()
    healthy = sum(1 for v in report.values() if v["healthy"])
    return {
        "total_providers": len(report),
        "healthy_providers": healthy,
        "providers": report,
    }


# ── ROOT ─────────────────────────────────────────

@app.get("/")
async def root():
    return {
        "service": "G4F API Bridge",
        "version": "1.0.0",
        "description": "OpenAI-compatible bridge: PicoClaw ↔ g4f",
        "endpoints": {
            "chat": "POST /v1/chat/completions",
            "models": "GET  /v1/models",
            "health": "GET  /health",
            "status": "GET  /v1/providers/status",
            "docs": "GET  /docs",
        },
    }


# ══════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn

    log.info("🚀 Memulai G4F API Bridge...")
    log.info(f"📋 Model tersedia: {len(manager.list_models())}")

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8820,
        reload=False,
        log_level="info",
    )