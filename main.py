"""
Main server — OpenAI-compatible API bridge with Function Calling support.
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

# Project root
sys.path.insert(0, str(Path(__file__).parent))

# Core init
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
from function_calling import get_emulator

BRIDGE_VERSION = "2.0.0"


# ═══════════════════════════════════════════════════════════════
# Response Builders
# ═══════════════════════════════════════════════════════════════

def _gen_id(prefix: str = "chatcmpl") -> str:
    return f"{prefix}-{uuid.uuid4().hex[:12]}"


def build_completion(
    content: str, model: str,
    prompt_tokens: int = 0, completion_tokens: int = 0,
) -> Dict[str, Any]:
    if completion_tokens == 0 and content:
        completion_tokens = max(1, len(content) // 4)
    return {
        "id": _gen_id(), "object": "chat.completion",
        "created": int(time.time()), "model": model,
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
    content: str, model: str, chunk_id: str,
    finish_reason: Optional[str] = None,
) -> str:
    delta = {"content": content} if content else {}
    obj = {
        "id": chunk_id, "object": "chat.completion.chunk",
        "created": int(time.time()), "model": model,
        "choices": [{
            "index": 0, "delta": delta,
            "finish_reason": finish_reason,
        }],
    }
    return f"data: {json.dumps(obj)}\n\n"


def build_error(
    message: str, error_type: str = "server_error",
    code: Optional[str] = None,
) -> Dict[str, Any]:
    return {
        "error": {
            "message": message, "type": error_type,
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
# Auth helper
# ═══════════════════════════════════════════════════════════════

def _verify_key(auth_header: Optional[str]) -> bool:
    if not auth_header:
        return False
    parts = auth_header.split()
    if len(parts) == 2 and parts[0].lower() == "bearer":
        return parts[1] == config.server.api_key
    return False


# ═══════════════════════════════════════════════════════════════
# Function Calling Handler
# ═══════════════════════════════════════════════════════════════

async def handle_chat_request(body: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main handler — supports full tool calling cycle.

    Cycle:
    1. PicoClaw sends {messages, tools} → Bridge returns {tool_calls}
    2. PicoClaw executes tool
    3. PicoClaw sends {messages with role:"tool"} → Bridge returns {content}
    """
    model = body.get("model", config.g4f.default_model)
    messages = body.get("messages", [])
    tools = body.get("tools", [])
    tool_choice = body.get("tool_choice", "auto")
    stream = body.get("stream", False)
    temperature = body.get("temperature", 0.7)
    max_tokens = body.get("max_tokens", 4096)

    if not messages:
        return build_error("messages is required", "invalid_request_error")

    emulator = get_emulator()
    tm = get_token_manager()
    router = get_router()
    fc_config = config.function_calling

    # ── Determine mode ────────────────────────────────────
    use_fc = fc_config.enabled and emulator.has_tools(body)
    has_results = emulator.has_tool_results(messages)

    if has_results:
        # STEP 3: Tool results → AI processes and gives final answer
        logger.info("📥 Processing tool results")
        prepared, mode = emulator.build_messages(messages, [], "auto")
        mode = "result"
        stream = False  # Need full response

    elif use_fc:
        # STEP 1: Need tool_calls
        logger.info(f"🔧 FC Mode: {len(tools)} tools")
        prepared, mode = emulator.build_messages(messages, tools, tool_choice)
        stream = False  # Need full response for JSON parsing

    else:
        # Normal chat
        logger.info(f"💬 Chat: model={model} msgs={len(messages)}")
        prepared = messages
        mode = "chat"

    # ── Handle streaming (chat mode only) ─────────────────
    if stream and mode == "chat":
        return {
            "__stream__": True, "router": router,
            "model": model, "messages": prepared,
            "temperature": temperature, "max_tokens": max_tokens,
        }

    # ── Prepare & route ───────────────────────────────────
    prepared_final, _ = tm.prepare_messages(prepared)
    token_count = tm.count_tokens(prepared_final, model)

    # Lower temperature for FC mode (more deterministic)
    if mode == "fc":
        temperature = min(temperature, 0.3)

    result = await router.route(
        model=model, messages=prepared_final, stream=False,
        temperature=temperature, max_tokens=max_tokens,
    )

    if not result.success:
        return build_error(result.error or "All providers failed")

    raw = str(result.response)

    # ── Parse based on mode ───────────────────────────────
    if mode in ("fc", "result"):
        parsed = emulator.parse_response(raw, tools, mode)

        # FC mode: retry if no tool_calls
        if (
            mode == "fc"
            and parsed["type"] == "text"
            and fc_config.max_parse_retries > 0
        ):
            logger.warning("🔄 FC retry: no tool_calls, trying harder")

            retry_msgs = prepared_final.copy()
            retry_msgs.append({
                "role": "user",
                "content": "Respond with JSON only. Use the format: "
                           '{"tool_calls":[{"name":"...","arguments":{...}}]}'
            })

            retry_result = await router.route(
                model=model, messages=retry_msgs, stream=False,
                temperature=0.1, max_tokens=max_tokens,
            )

            if retry_result.success:
                parsed = emulator.parse_response(
                    str(retry_result.response), tools, mode
                )

        # Build response
        response = emulator.build_response(
            parsed, model, token_count.prompt_tokens
        )

        if parsed["type"] == "tool_calls":
            logger.info(
                f"✅ {len(parsed['tool_calls'])} tool_call(s) "
                f"via {result.provider_name}"
            )
        elif mode == "result":
            logger.info(f"✅ Final answer via {result.provider_name}")
        else:
            logger.warning("⚠️ FC fallback to text")

        return response

    # ── Normal chat response ──────────────────────────────
    comp_tokens = tm.counter.count_text(raw)
    return build_completion(raw, model, token_count.prompt_tokens, comp_tokens)

# ═══════════════════════════════════════════════════════════════
# Server Setup
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
# FastAPI
# ═══════════════════════════════════════════════════════════════

if USE_FASTAPI:

    app = FastAPI(
        title="g4f-Bridge", version=BRIDGE_VERSION,
        docs_url=None, redoc_url=None,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.on_event("startup")
    async def _startup() -> None:
        logger.info("Starting g4f-Bridge v2 (Function Calling Edition)...")
        get_scanner().scan()
        await get_update_scheduler().start()
        logger.info(f"Bridge ready on :{config.server.port}")

    @app.on_event("shutdown")
    async def _shutdown() -> None:
        logger.info("Shutting down...")
        await get_update_scheduler().stop()
        save_config()

    # ── Chat Completions ──────────────────────────────────

    @app.post("/v1/chat/completions")
    async def chat_completions(
        request: Request,
        authorization: Optional[str] = Header(None),
    ) -> Any:
        if not _verify_key(authorization):
            return JSONResponse(
                status_code=401,
                content=build_error(
                    "Invalid API key", "authentication_error",
                    "invalid_api_key",
                ),
            )

        try:
            body = await request.json()
        except Exception:
            return JSONResponse(
                status_code=400,
                content=build_error("Invalid JSON", "invalid_request_error"),
            )

        # Handle request
        result = await handle_chat_request(body)

        # Check if streaming
        if isinstance(result, dict) and result.get("__stream__"):
            router_obj = result["router"]
            model = result["model"]
            msgs = result["messages"]

            return StreamingResponse(
                _stream_gen(
                    router_obj, model, msgs,
                    temperature=result.get("temperature", 0.7),
                    max_tokens=result.get("max_tokens", 4096),
                ),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                },
            )

        # Check if error
        if "error" in result:
            status = 400 if result["error"].get("type") == "invalid_request_error" else 502
            return JSONResponse(status_code=status, content=result)

        return JSONResponse(content=result)

    async def _stream_gen(
        router: Any, model: str,
        messages: List[Dict[str, str]], **kwargs: Any,
    ) -> AsyncGenerator[str, None]:
        chunk_id = _gen_id()
        try:
            result = await router.route(
                model=model, messages=messages,
                stream=True, **kwargs,
            )

            if not result.success:
                yield build_chunk("", model, chunk_id, "stop")
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

            yield build_chunk("", model, chunk_id, "stop")
            yield "data: [DONE]\n\n"

        except Exception as e:
            logger.error(f"Stream error: {e}")
            yield build_chunk(f"Error: {e}", model, chunk_id, "stop")
            yield "data: [DONE]\n\n"

    # ── Models ────────────────────────────────────────────

    @app.get("/v1/models")
    async def list_models(
        authorization: Optional[str] = Header(None),
    ) -> Any:
        if not _verify_key(authorization):
            return JSONResponse(
                status_code=401,
                content=build_error(
                    "Invalid API key", "authentication_error"
                ),
            )
        return JSONResponse(
            content=build_models_list(get_scanner().get_all_models())
        )

    @app.get("/v1/models/{model_id}")
    async def get_model(
        model_id: str,
        authorization: Optional[str] = Header(None),
    ) -> Any:
        if not _verify_key(authorization):
            return JSONResponse(
                status_code=401,
                content=build_error(
                    "Invalid API key", "authentication_error"
                ),
            )
        if model_id in get_scanner().get_all_models():
            return JSONResponse(content={
                "id": model_id, "object": "model",
                "created": int(time.time()), "owned_by": "g4f-bridge",
            })
        return JSONResponse(
            status_code=404,
            content=build_error(
                f"Model '{model_id}' not found",
                "invalid_request_error",
            ),
        )

    # ── Management ────────────────────────────────────────

    @app.get("/health")
    async def health() -> Any:
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "version": BRIDGE_VERSION,
            "function_calling": config.function_calling.enabled,
            "strict_provider": config.provider_lock.strict_provider_mode,
        }

    @app.get("/status")
    async def status(
        authorization: Optional[str] = Header(None),
    ) -> Any:
        if not _verify_key(authorization):
            return JSONResponse(
                status_code=401,
                content=build_error("Unauthorized", "authentication_error"),
            )
        return {
            "bridge_version": BRIDGE_VERSION,
            "function_calling": {
                "enabled": config.function_calling.enabled,
                "fallback_to_text": config.function_calling.fallback_to_text,
            },
            "provider_lock": {
                "strict_mode": config.provider_lock.strict_provider_mode,
                "locked_provider": config.provider_lock.locked_provider,
                "locked_model": config.provider_lock.locked_model,
            },
            "scanner": get_scanner().get_status(),
            "updater": get_updater().get_status(),
        }

    @app.get("/providers")
    async def providers(
        authorization: Optional[str] = Header(None),
    ) -> Any:
        if not _verify_key(authorization):
            return JSONResponse(
                status_code=401,
                content=build_error("Unauthorized", "authentication_error"),
            )
        return {
            "providers": {
                n: p.to_dict()
                for n, p in get_scanner().providers.items()
            }
        }

    @app.get("/compatibility")
    async def compatibility(
        authorization: Optional[str] = Header(None),
    ) -> Any:
        if not _verify_key(authorization):
            return JSONResponse(
                status_code=401,
                content=build_error("Unauthorized", "authentication_error"),
            )
        return {"models": get_scanner().model_to_providers}

    @app.post("/scan")
    async def trigger_scan(
        authorization: Optional[str] = Header(None),
    ) -> Any:
        if not _verify_key(authorization):
            return JSONResponse(
                status_code=401,
                content=build_error("Unauthorized", "authentication_error"),
            )
        get_scanner().scan()
        return {
            "status": "done",
            "providers": len(get_scanner().providers),
        }

    @app.post("/update")
    async def trigger_update(
        authorization: Optional[str] = Header(None),
    ) -> Any:
        if not _verify_key(authorization):
            return JSONResponse(
                status_code=401,
                content=build_error("Unauthorized", "authentication_error"),
            )
        ok, msg = await get_updater().update()
        return {"success": ok, "message": msg}

    @app.get("/version")
    async def version() -> Any:
        return {"bridge": BRIDGE_VERSION, **get_updater().get_status()}

    @app.get("/api-key")
    async def api_key_endpoint(request: Request) -> Any:
        client = request.client
        if client and client.host not in ("127.0.0.1", "::1", "localhost"):
            return JSONResponse(
                status_code=403,
                content=build_error("Forbidden", "authentication_error"),
            )
        return {"api_key": config.server.api_key}

    @app.get("/config")
    async def get_current_config(
        authorization: Optional[str] = Header(None),
    ) -> Any:
        if not _verify_key(authorization):
            return JSONResponse(
                status_code=401,
                content=build_error("Unauthorized", "authentication_error"),
            )
        return config.to_dict()


# ═══════════════════════════════════════════════════════════════
# Flask Fallback
# ═══════════════════════════════════════════════════════════════

else:
    from flask import Flask, request as flask_request, jsonify  # type: ignore

    app = Flask(__name__)

    def _flask_auth() -> bool:
        return _verify_key(
            flask_request.headers.get("Authorization")
        )

    @app.before_first_request
    def _flask_startup() -> None:
        get_scanner().scan()

    @app.route("/v1/chat/completions", methods=["POST"])
    def flask_chat() -> Any:
        if not _flask_auth():
            return jsonify(
                build_error("Invalid API key", "authentication_error")
            ), 401

        body = flask_request.get_json(silent=True) or {}

        # Sync wrapper for async handler
        import asyncio
        try:
            loop = asyncio.new_event_loop()
            result = loop.run_until_complete(handle_chat_request(body))
        finally:
            loop.close()

        # Handle streaming flag
        if isinstance(result, dict) and result.get("__stream__"):
            return jsonify(
                build_error(
                    "Streaming requires FastAPI mode",
                    "invalid_request_error",
                )
            ), 400

        if "error" in result:
            status = 400 if result["error"].get("type") == "invalid_request_error" else 502
            return jsonify(result), status

        return jsonify(result)

    @app.route("/v1/models", methods=["GET"])
    def flask_models() -> Any:
        if not _flask_auth():
            return jsonify(
                build_error("Invalid API key", "authentication_error")
            ), 401
        return jsonify(
            build_models_list(get_scanner().get_all_models())
        )

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
    logger.info(f"Signal {signum}, shutting down...")
    save_config()
    sys.exit(0)


def print_banner() -> None:
    fc_status = "✅ ON" if config.function_calling.enabled else "❌ OFF"
    lock_status = (
        f"🔒 {config.provider_lock.locked_provider}"
        if config.provider_lock.strict_provider_mode
        else "🔓 OFF"
    )

    print()
    print("=" * 60)
    print(f"  g4f-Bridge v{BRIDGE_VERSION} (Function Calling Edition)")
    print("=" * 60)
    print(f"  Server:           http://{config.server.host}:{config.server.port}")
    print(f"  API Key:          {config.server.api_key}")
    print(f"  Backend:          {'FastAPI' if USE_FASTAPI else 'Flask'}")
    print(f"  Function Calling: {fc_status}")
    print(f"  Provider Lock:    {lock_status}")
    print("-" * 60)
    print("  PicoClaw config:")
    print(f"    api_base = http://localhost:{config.server.port}/v1")
    print(f"    api_key  = {config.server.api_key}")
    print("=" * 60)
    print()


def main() -> None:
    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    print_banner()

    if USE_FASTAPI:
        uvicorn.run(
            app, host=config.server.host, port=config.server.port,
            log_level="warning", access_log=False,
        )
    else:
        get_scanner().scan()
        app.run(
            host=config.server.host, port=config.server.port,
            debug=False, threaded=True,
        )


if __name__ == "__main__":
    main()