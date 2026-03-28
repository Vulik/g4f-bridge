"""
main.py
~~~~~~~
FastAPI application:
  • Startup: menjalankan ProviderScanner.initialize()
  • POST /v1/chat/completions — bridge utama (non-stream & SSE)
  • GET  /v1/models            — daftar model aktif
  • POST /v1/scan              — trigger rescan manual
  • GET  /v1/providers/status  — status kesehatan provider
  • GET  /health               — health check
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from config import HOST, PORT
from provider_manager import ProviderManager
from provider_scanner import ProviderScanner
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
#  LOGGING
# ══════════════════════════════════════════════════

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-7s │ %(name)s │ %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("g4f-bridge")

# ══════════════════════════════════════════════════
#  GLOBAL STATE
# ══════════════════════════════════════════════════

scanner = ProviderScanner()
manager = ProviderManager()


# ══════════════════════════════════════════════════
#  LIFESPAN (startup / shutdown)
# ══════════════════════════════════════════════════

@asynccontextmanager
async def lifespan(app_: FastAPI):
    """
    Saat startup:
      1. Scanner memindai / memuat cache
      2. Manager dimuat dengan hasil scan
    """
    log.info("🚀 G4F API Bridge memulai...")

    results = await scanner.initialize(force_rescan=False)
    manager.load_scan_results(results)

    s = manager.summary()
    log.info(
        f"🟢 Bridge siap — {s['healthy_providers']} provider, "
        f"{s['total_models']} model"
    )
    log.info(f"🌐 Listening di http://{HOST}:{PORT}")
    log.info(f"📖 Docs di http://{HOST}:{PORT}/docs")

    yield  # ── app berjalan ──

    log.info("👋 Bridge berhenti.")


# ══════════════════════════════════════════════════
#  APP
# ══════════════════════════════════════════════════

app = FastAPI(
    title="G4F API Bridge",
    description=(
        "Menjembatani PicoClaw ↔ g4f "
        "dengan format OpenAI-compatible.\n\n"
        "Mendukung auto-scan, auto-model, "
        "retry, fallback, dan caching."
    ),
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ══════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════

def _sse_line(chunk: ChatCompletionChunk) -> str:
    return f"data: {chunk.model_dump_json()}\n\n"


def _estimate_usage(messages: list[dict], text: str) -> Usage:
    pt = sum(len((m.get("content") or "").split()) for m in messages)
    ct = len(text.split())
    return Usage(
        prompt_tokens=pt, completion_tokens=ct, total_tokens=pt + ct
    )


# ══════════════════════════════════════════════════
#  ROUTES
# ══════════════════════════════════════════════════

# ── CHAT COMPLETIONS ────────────────────────────

@app.post("/v1/chat/completions")
@app.post("/chat/completions")
async def chat_completions(req: ChatCompletionRequest):
    """
    Endpoint utama — menerima format OpenAI,
    meneruskan ke g4f, mengembalikan format OpenAI.
    
    Jika model="auto", bridge memilih model & provider
    terbaik secara otomatis.
    """
    messages = [
        {"role": m.role, "content": m.content or ""}
        for m in req.messages
    ]

    log.info(
        f"📨 Request — model={req.model}, "
        f"msgs={len(messages)}, stream={req.stream}"
    )

    # ── STREAMING ────────────────────────────────

    if req.stream:
        try:
            token_gen, actual_model, pname = (
                await manager.stream_with_fallback(
                    model=req.model,
                    messages=messages,
                    temperature=req.temperature,
                    max_tokens=req.max_tokens,
                )
            )
        except RuntimeError as exc:
            raise HTTPException(
                status_code=503,
                detail={
                    "error": "all_stream_providers_failed",
                    "model": req.model,
                    "message": str(exc),
                },
            )

        cid = f"chatcmpl-{uuid.uuid4().hex[:16]}"
        ts = int(time.time())

        async def sse_gen() -> AsyncGenerator[str, None]:
            try:
                # Chunk awal: role
                yield _sse_line(
                    ChatCompletionChunk(
                        id=cid,
                        created=ts,
                        model=actual_model,
                        choices=[
                            StreamChoice(
                                delta=DeltaMessage(role="assistant")
                            )
                        ],
                    )
                )

                # Chunk konten
                async for token in token_gen:
                    yield _sse_line(
                        ChatCompletionChunk(
                            id=cid,
                            created=ts,
                            model=actual_model,
                            choices=[
                                StreamChoice(
                                    delta=DeltaMessage(content=token)
                                )
                            ],
                        )
                    )

                # Chunk penutup
                yield _sse_line(
                    ChatCompletionChunk(
                        id=cid,
                        created=ts,
                        model=actual_model,
                        choices=[
                            StreamChoice(
                                delta=DeltaMessage(),
                                finish_reason="stop",
                            )
                        ],
                    )
                )
                yield "data: [DONE]\n\n"

            except Exception as exc:
                log.error(f"Stream error: {exc}")
                err = {
                    "error": {
                        "message": str(exc),
                        "type": "stream_error",
                    }
                }
                yield f"data: {json.dumps(err)}\n\n"

        return StreamingResponse(
            sse_gen(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-G4F-Provider": pname,
                "X-G4F-Model": actual_model,
            },
        )

    # ── NON-STREAMING ───────────────────────────

    try:
        content, actual_model, pname = (
            await manager.execute_with_fallback(
                model=req.model,
                messages=messages,
                temperature=req.temperature,
                max_tokens=req.max_tokens,
            )
        )
    except RuntimeError as exc:
        raise HTTPException(
            status_code=503,
            detail={
                "error": "all_providers_failed",
                "model": req.model,
                "message": str(exc),
            },
        )

    response = ChatCompletionResponse(
        model=actual_model,
        choices=[
            Choice(
                message=ChatMessage(
                    role="assistant", content=content
                ),
                finish_reason="stop",
            )
        ],
        usage=_estimate_usage(messages, content),
    )

    return JSONResponse(
        content=response.model_dump(),
        headers={
            "X-G4F-Provider": pname,
            "X-G4F-Model": actual_model,
        },
    )


# ── MODEL LIST ──────────────────────────────────

@app.get("/v1/models")
@app.get("/models")
async def list_models():
    """Daftar semua model aktif (hasil scan terakhir)."""
    models = manager.list_models()
    return {
        "object": "list",
        "data": [
            {
                "id": m,
                "object": "model",
                "created": 0,
                "owned_by": "g4f",
            }
            for m in models
        ]
        + [
            {
                "id": "auto",
                "object": "model",
                "created": 0,
                "owned_by": "g4f-bridge",
                "description": "Otomatis memilih model & provider terbaik",
            }
        ],
    }


# ── MANUAL RESCAN ───────────────────────────────

@app.post("/v1/scan")
async def trigger_rescan():
    """
    Picu pemindaian ulang semua provider.
    Cache lama akan ditimpa.
    """
    log.info("🔄 Rescan manual diminta...")
    t0 = time.time()
    results = await scanner.initialize(force_rescan=True)
    manager.load_scan_results(results)
    elapsed = time.time() - t0

    s = manager.summary()
    active = sum(1 for r in results if r.is_active)
    return {
        "status": "completed",
        "duration_seconds": round(elapsed, 2),
        "total_tested": len(results),
        "active_providers": s["healthy_providers"],
        "active_models": s["total_models"],
        "models": s["models"],
    }


# ── PROVIDER STATUS ─────────────────────────────

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


# ── HEALTH CHECK ────────────────────────────────

@app.get("/health")
async def health():
    s = manager.summary()
    return {
        "status": "healthy" if s["healthy_providers"] > 0 else "degraded",
        "providers": s["healthy_providers"],
        "models": s["total_models"],
        "scan_cached": scanner.scan_duration > 0,
        "timestamp": int(time.time()),
    }


# ── ROOT ────────────────────────────────────────

@app.get("/")
async def root():
    s = manager.summary()
    return {
        "service": "G4F API Bridge",
        "version": "2.0.0",
        "status": "running",
        "providers_active": s["healthy_providers"],
        "models_active": s["total_models"],
        "endpoints": {
            "chat": "POST /v1/chat/completions",
            "models": "GET  /v1/models",
            "scan": "POST /v1/scan",
            "status": "GET  /v1/providers/status",
            "health": "GET  /health",
            "docs": "GET  /docs",
        },
        "notes": {
            "auto_model": (
                'Gunakan model="auto" untuk pemilihan otomatis '
                "model & provider terbaik."
            ),
            "picoclaw": (
                f"Set API Base URL ke http://<host>:{PORT}/v1"
            ),
        },
    }


# ══════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host=HOST,
        port=PORT,
        reload=False,
        log_level="info",
    )