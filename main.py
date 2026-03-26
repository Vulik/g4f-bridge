"""
Main server — OpenAI-compatible API bridge for g4f.
"""

import os
import sys
import uuid
import time
import json
import signal
import asyncio
from typing import Dict, List, Any, Optional, AsyncGenerator
from datetime import datetime
from pathlib import Path

# Ensure project root in path
sys.path.insert(0, str(Path(__file__).parent))

# Initialize core systems
from environment import get_environment, EnvironmentDetector

env = get_environment()
EnvironmentDetector.print_info(env)

from logger_setup import setup_logging, get_logger

logger = setup_logging(log_level="INFO")

from config import get_config, save_config

config = get_config()

from storage import get_storage
from scanner import get_scanner
from router import get_router
from token_manager import get_token_manager
from resilience import get_resilience
from updater import get_updater, get_update_scheduler

# Bridge version
BRIDGE_VERSION = "1.0.0"


# ═══════════════════════════════════════════════════════════════
# OpenAI Response Builders
# ═══════════════════════════════════════════════════════════════

def _gen_id(prefix: str = "chatcmpl") -> str:
    return f"{prefix}-{uuid.uuid4().hex[:12]}"


def build_completion(
    content: str,
    model: str,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
) -> Dict[str, Any]:
    if completion_tokens == 0 and content:
        completion_tokens = max(1, len(content) // 4)
    return {
        "id": _gen_id(),
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": content},
            "finish_reason": "stop",
        }],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


def build_chunk(
    content: str,
    model: str,
    chunk_id: str,
    finish_reason: Optional[str] = None,
) -> str:
    delta = {"content": content} if content else {}
    obj = {
        "id": chunk_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "delta": delta,
            "finish_reason": finish_reason,
        }],
    }
    return f"data: {json.dumps(obj)}\n\n"


def build_error(
    message: str,
    error_type: str = "server_error",
    code: Optional[str] = None,
) -> Dict[str, Any]:
    return {
        "error": {
            "message": message,
            "type": error_type,
            "code": code or error_type,
        }
    }


def build_models_list(models: List[str]) -> Dict[str, Any]:
    ts = int(time.time())
    return {
        "object": "list",
        "data": [
            {"id": m, "object": "model", "created": ts, "owned_by": "g4f-bridge"}
            for m in models
        ],
    }


# ═══════════════════════════════════════════════════════════════
# API key verification
# ═══════════════════════════════════════════════════════════════

def _extract_key(auth_header: Optional[str]) -> Optional[str]:
    if not auth_header:
        return None
    parts = auth_header.split()
    if len(parts) == 2 and parts[0].lower() == "bearer":
        return parts[1]
    return None


def _verify_key(auth_header: Optional[str]) -> bool:
    key = _extract_key(auth_header)
    return key == config.server.api_key


# ═══════════════════════════════════════════════════════════════
# Try FastAPI, fallback Flask
# ═══════════════════════════════════════════════════════════════

USE_FASTAPI = False

try:
    from fastapi import FastAPI, Request, Header
    from fastapi.responses import StreamingResponse, JSONResponse
    from fastapi.middleware.cors import CORSMiddleware
    import uvicorn

    USE_FASTAPI = True
    logger.info("Using FastAPI + Uvicorn")
except ImportError:
    logger.warning("FastAPI/Uvicorn not available, using Flask")


# ═══════════════════════════════════════════════════════════════
# FastAPI Application
# ═══════════════════════════════════════════════════════════════

if USE_FASTAPI:

    app = FastAPI(title="g4f-Bridge", version=BRIDGE_VERSION, docs_url=None, redoc_url=None)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.on_event("startup")
    async def _startup() -> None:
        logger.info("Starting g4f-Bridge (FastAPI)...")
        get_scanner().scan()
        await get_update_scheduler().start()
        logger.info(f"Bridge ready on :{config.server.port}")

    @app.on_event("shutdown")
    async def _shutdown() -> None:
        logger.info("Shutting down...")
        await get_update_scheduler().stop()
        save_config()

    # ── Chat completions ─────────────────────────────────────

    @app.post("/v1/chat/completions")
    async def chat_completions(
        request: Request,
        authorization: Optional[str] = Header(None),
    ) -> Any:
        if not _verify_key(authorization):
            return JSONResponse(
                status_code=401,
                content=build_error("Invalid API key", "authentication_error", "invalid_api_key"),
            )

        try:
            body = await request.json()
        except Exception:
            return JSONResponse(status_code=400, content=build_error("Invalid JSON", "invalid_request_error"))

        model = body.get("model", config.g4f.default_model)
        messages = body.get("messages", [])
        stream = body.get("stream", False)
        temperature = body.get("temperature", 0.7)
        max_tokens = body.get("max_tokens", 4096)

        if not messages:
            return JSONResponse(
                status_code=400,
                content=build_error("messages is required", "invalid_request_error"),
            )

        logger.info(f"Request: model={model} stream={stream} msgs={len(messages)}")

        tm = get_token_manager()
        prepared, session_id = tm.prepare_messages(messages)
        token_count = tm.count_tokens(prepared, model)

        router = get_router()

        if stream:
            return StreamingResponse(
                _stream_gen(router, model, prepared, temperature=temperature, max_tokens=max_tokens),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
            )

        # Non-streaming
        result = await router.route(model=model, messages=prepared, stream=False, temperature=temperature, max_tokens=max_tokens)

        if result.success:
            text = str(result.response)
            comp_tokens = tm.counter.count_text(text)
            return JSONResponse(content=build_completion(text, model, token_count.prompt_tokens, comp_tokens))
        else:
            return JSONResponse(status_code=502, content=build_error(result.error or "All providers failed"))

    async def _stream_gen(
        router: Any,
        model: str,
        messages: List[Dict[str, str]],
        **kwargs: Any,
    ) -> AsyncGenerator[str, None]:
        chunk_id = _gen_id()

        try:
            result = await router.route(model=model, messages=messages, stream=True, **kwargs)

            if not result.success:
                yield build_chunk("", model, chunk_id, finish_reason="stop")
                yield "data: [DONE]\n\n"
                return

            resp = result.response

            if hasattr(resp, "__aiter__"):
                async for chunk in resp:
                    if chunk:
                        yield build_chunk(str(chunk), model, chunk_id)
            elif hasattr(resp, "__iter__") and not isinstance(resp, str):
                for chunk in resp:
                    if chunk:
                        yield build_chunk(str(chunk), model, chunk_id)
            else:
                yield build_chunk(str(resp), model, chunk_id)

            yield build_chunk("", model, chunk_id, finish_reason="stop")
            yield "data: [DONE]\n\n"

        except Exception as e:
            logger.error(f"Stream error: {e}")
            yield build_chunk(f"Error: {e}", model, chunk_id, finish_reason="stop")
            yield "data: [DONE]\n\n"

    # ── Models ───────────────────────────────────────────────

    @app.get("/v1/models")
    async def list_models(authorization: Optional[str] = Header(None)) -> Any:
        if not _verify_key(authorization):
            return JSONResponse(status_code=401, content=build_error("Invalid API key", "authentication_error"))
        return JSONResponse(content=build_models_list(get_scanner().get_all_models()))

    @app.get("/v1/models/{model_id}")
    async def get_model(model_id: str, authorization: Optional[str] = Header(None)) -> Any:
        if not _verify_key(authorization):
            return JSONResponse(status_code=401, content=build_error("Invalid API key", "authentication_error"))

        if model_id in get_scanner().get_all_models():
            return JSONResponse(content={
                "id": model_id, "object": "model",
                "created": int(time.time()), "owned_by": "g4f-bridge",
            })
        return JSONResponse(status_code=404, content=build_error(f"Model '{model_id}' not found", "invalid_request_error"))

    # ── Management (no auth for health, auth for others) ─────

    @app.get("/health")
    async def health() -> Any:
        return {"status": "healthy", "timestamp": datetime.now().isoformat(), "version": BRIDGE_VERSION}

    @app.get("/status")
    async def status(authorization: Optional[str] = Header(None)) -> Any:
        if not _verify_key(authorization):
            return JSONResponse(status_code=401, content=build_error("Unauthorized", "authentication_error"))
        return {
            "bridge_version": BRIDGE_VERSION,
            "environment": env.to_dict(),
            "scanner": get_scanner().get_status(),
            "updater": get_updater().get_status(),
        }

    @app.get("/providers")
    async def providers(authorization: Optional[str] = Header(None)) -> Any:
        if not _verify_key(authorization):
            return JSONResponse(status_code=401, content=build_error("Unauthorized", "authentication_error"))
        return {"providers": {n: p.to_dict() for n, p in get_scanner().providers.items()}}

    @app.get("/compatibility")
    async def compatibility(authorization: Optional[str] = Header(None)) -> Any:
        if not _verify_key(authorization):
            return JSONResponse(status_code=401, content=build_error("Unauthorized", "authentication_error"))
        return {"models": get_scanner().model_to_providers}

    @app.post("/scan")
    async def trigger_scan(authorization: Optional[str] = Header(None)) -> Any:
        if not _verify_key(authorization):
            return JSONResponse(status_code=401, content=build_error("Unauthorized", "authentication_error"))
        get_scanner().scan()
        return {"status": "done", "providers": len(get_scanner().providers)}

    @app.post("/update")
    async def trigger_update(authorization: Optional[str] = Header(None)) -> Any:
        if not _verify_key(authorization):
            return JSONResponse(status_code=401, content=build_error("Unauthorized", "authentication_error"))
        ok, msg = await get_updater().update()
        return {"success": ok, "message": msg}

    @app.get("/version")
    async def version() -> Any:
        return {"bridge": BRIDGE_VERSION, **get_updater().get_status()}

    @app.get("/api-key")
    async def api_key_endpoint(request: Request) -> Any:
        """Only accessible from localhost."""
        client = request.client
        if client and client.host not in ("127.0.0.1", "::1", "localhost"):
            return JSONResponse(status_code=403, content=build_error("Forbidden", "authentication_error"))
        return {"api_key": config.server.api_key}


# ═══════════════════════════════════════════════════════════════
# Flask Fallback
# ═══════════════════════════════════════════════════════════════

else:
    from flask import Flask, request as flask_request, jsonify  # type: ignore

    app = Flask(__name__)

    def _flask_auth() -> bool:
        return _verify_key(flask_request.headers.get("Authorization"))

    @app.before_first_request
    def _flask_startup() -> None:
        get_scanner().scan()

    @app.route("/v1/chat/completions", methods=["POST"])
    def flask_chat() -> Any:
        if not _flask_auth():
            return jsonify(build_error("Invalid API key", "authentication_error")), 401

        body = flask_request.get_json(silent=True) or {}
        model = body.get("model", config.g4f.default_model)
        messages = body.get("messages", [])

        if body.get("stream"):
            return jsonify(build_error("Streaming requires FastAPI mode", "invalid_request_error")), 400

        if not messages:
            return jsonify(build_error("messages required", "invalid_request_error")), 400

        try:
            import g4f  # type: ignore

            # Use scanner to find provider
            scanner = get_scanner()
            providers = scanner.get_providers_for_model(model)

            response = None
            for pinfo in providers[:5]:
                try:
                    actual_model = pinfo.models[0] if pinfo.models else model
                    for m in pinfo.models:
                        if scanner.model_matches(m, model):
                            actual_model = m
                            break

                    response = g4f.ChatCompletion.create(
                        model=actual_model,
                        messages=messages,
                        provider=pinfo.class_ref,
                    )
                    if response:
                        scanner.update_provider_status(pinfo.name, True)
                        break
                except Exception as e:
                    scanner.update_provider_status(pinfo.name, False, error=str(e))
                    continue

            if response:
                return jsonify(build_completion(str(response), model))
            else:
                return jsonify(build_error("All providers failed")), 502

        except Exception as e:
            return jsonify(build_error(str(e))), 500

    @app.route("/v1/models", methods=["GET"])
    def flask_models() -> Any:
        if not _flask_auth():
            return jsonify(build_error("Invalid API key", "authentication_error")), 401
        return jsonify(build_models_list(get_scanner().get_all_models()))

    @app.route("/health", methods=["GET"])
    def flask_health() -> Any:
        return jsonify({"status": "healthy", "version": BRIDGE_VERSION})

    @app.route("/api-key", methods=["GET"])
    def flask_api_key() -> Any:
        remote = flask_request.remote_addr
        if remote not in ("127.0.0.1", "::1"):
            return jsonify(build_error("Forbidden")), 403
        return jsonify({"api_key": config.server.api_key})


# ═══════════════════════════════════════════════════════════════
# Entry Point
# ═══════════════════════════════════════════════════════════════

def handle_signal(signum: int, frame: Any) -> None:
    logger.info(f"Signal {signum} received, shutting down...")
    save_config()
    sys.exit(0)


def print_banner() -> None:
    print()
    print("=" * 60)
    print(f"  g4f-Bridge v{BRIDGE_VERSION}")
    print("=" * 60)
    print(f"  Server:   http://{config.server.host}:{config.server.port}")
    print(f"  API Key:  {config.server.api_key}")
    print(f"  Backend:  {'FastAPI' if USE_FASTAPI else 'Flask'}")
    print("-" * 60)
    print("  PicoClaw settings:")
    print(f"    base_url = http://localhost:{config.server.port}/v1")
    print(f"    api_key  = {config.server.api_key}")
    print("=" * 60)
    print()


def main() -> None:
    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    print_banner()

    if USE_FASTAPI:
        uvicorn.run(
            app,
            host=config.server.host,
            port=config.server.port,
            log_level="warning",
            access_log=False,
        )
    else:
        # Scan manually for Flask
        get_scanner().scan()
        app.run(
            host=config.server.host,
            port=config.server.port,
            debug=False,
            threaded=True,
        )


if __name__ == "__main__":
    main()