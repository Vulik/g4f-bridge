"""
Main server v5 — Premium + g4f, startup testing.
"""

import sys
import uuid
import time
import json
import signal
import asyncio
from typing import Dict, List, Any, Optional, AsyncGenerator
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from environment import get_environment, EnvironmentDetector

env = get_environment()
EnvironmentDetector.print_info(env)

from logger_setup import setup_logging, get_logger

logger = setup_logging(log_level="INFO")

from config import get_config, save_config

config = get_config()

from scanner import get_scanner
from router import get_router
from token_manager import get_token_manager
from resilience import get_resilience
from updater import get_update_scheduler
from function_calling import get_emulator
from test_worker import get_test_worker
from premium_adapter import get_premium_adapter

BRIDGE_VERSION = "5.0.0"


def _gen_request_id() -> str:
    return "req_" + uuid.uuid4().hex[:8]


def _gen_id(prefix: str = "chatcmpl") -> str:
    return f"{prefix}-{uuid.uuid4().hex[:12]}"


def _strip_prefix(model: str) -> str:
    if "/" in model and model != "auto":
        return model.split("/", 1)[-1]
    return model


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
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }
    return f"data: {json.dumps(obj)}\n\n"


def build_error(
    message: str, error_type: str = "server_error",
    code: Optional[str] = None,
) -> Dict[str, Any]:
    return {"error": {"message": message, "type": error_type, "code": code or error_type}}


def build_models_list(models: List[str]) -> Dict[str, Any]:
    ts = int(time.time())
    all_models = set(models)
    for m in models:
        all_models.add(f"openai/{m}")
    return {
        "object": "list",
        "data": [
            {"id": m, "object": "model", "created": ts, "owned_by": "g4f-bridge"}
            for m in sorted(all_models)
        ],
    }


def ensure_format(response: Dict[str, Any]) -> Dict[str, Any]:
    if "error" in response:
        return response
    response.setdefault("id", _gen_id())
    response.setdefault("object", "chat.completion")
    response.setdefault("created", int(time.time()))
    response.setdefault("model", "unknown")
    for choice in response.get("choices", []):
        msg = choice.get("message", {})
        if not msg.get("tool_calls") and msg.get("content") is None:
            msg["content"] = ""
        choice.setdefault("finish_reason", "stop")
        choice.setdefault("index", 0)
    response.setdefault("usage", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})
    return response


def _verify_key(auth_header: Optional[str]) -> bool:
    if not config.server.api_key:
        return True
    if not auth_header:
        return False
    parts = auth_header.split()
    if len(parts) == 2 and parts[0].lower() == "bearer":
        return parts[1] == config.server.api_key
    return False


async def handle_chat_request(body: Dict[str, Any], request_id: str = "") -> Dict[str, Any]:
    model = _strip_prefix(body.get("model", config.g4f.default_model))
    messages = body.get("messages", [])
    tools = body.get("tools", [])
    tool_choice = body.get("tool_choice", "auto")
    stream = body.get("stream", False)
    temperature = body.get("temperature", 0.7)
    max_tokens = body.get("max_tokens", 4096)

    if not messages:
        return build_error("messages is required", "invalid_request_error")

    logger.info(f"[{request_id}] → model={model} msgs={len(messages)} tools={len(tools)} stream={stream}")

    emulator = get_emulator()
    tm = get_token_manager()
    router = get_router()
    fc_config = config.function_calling
    premium = get_premium_adapter()

    has_tools = fc_config.enabled and emulator.has_tools(body)
    has_results = emulator.has_tool_results(messages)

    # Premium API (native FC)
    if premium.is_enabled():
        result = await router._route_premium(
            model, messages, stream, request_id,
            tools if has_tools else None, tool_choice,
            temperature=temperature, max_tokens=max_tokens,
        )
        if result.success:
            if result.is_premium:
                if stream and hasattr(result.response, '__aiter__'):
                    return {
                        "__stream_premium__": True,
                        "generator": result.response,
                        "request_id": request_id,
                    }
                return result.response
        logger.warning(f"[{request_id}] Premium failed, fallback g4f")

    # g4f with FC emulation
    if has_results:
        logger.info(f"[{request_id}] 📥 Tool result")
        prepared, mode = emulator.build_messages(messages, [], "auto")
        mode = "result"
        stream = False
    elif has_tools:
        logger.info(f"[{request_id}] 🔧 FC mode")
        prepared, mode = emulator.build_messages(messages, tools, tool_choice)
        stream = False
    else:
        prepared = messages
        mode = "chat"

    if stream and mode == "chat":
        return {
            "__stream__": True, "router": router, "model": model,
            "messages": prepared, "temperature": temperature,
            "max_tokens": max_tokens, "request_id": request_id,
        }

    if mode == "fc":
        temperature = min(temperature, 0.3)

    result = await router._route_g4f(
        model, prepared, False, request_id,
        temperature=temperature, max_tokens=max_tokens, _has_tools=has_tools,
    )

    if not result.success:
        return build_error(result.error or "All failed")

    raw = str(result.response)

    if mode in ("fc", "result"):
        parsed = emulator.parse_response(raw, tools, mode)

        if mode == "fc" and parsed["type"] == "text" and fc_config.max_parse_retries > 0:
            logger.warning(f"[{request_id}] 🔄 FC retry")
            last_user = ""
            for m in reversed(messages):
                if m.get("role") == "user":
                    last_user = m.get("content", "")
                    break
            retry_msgs = emulator.build_retry_messages(prepared, tools, last_user)
            retry = await router._route_g4f(
                model, retry_msgs, False, request_id,
                temperature=0.1, max_tokens=max_tokens,
            )
            if retry.success:
                parsed = emulator.parse_response(str(retry.response), tools, mode)

        token_count = tm.count_tokens(prepared, model)
        response = emulator.build_response(parsed, model, token_count.prompt_tokens)

        if parsed["type"] == "tool_calls" and parsed["tool_calls"]:
            names = [tc["function"]["name"] for tc in parsed["tool_calls"]]
            logger.info(f"[{request_id}] ← {result.provider_name} fc={','.join(names)}")

        return response

    comp = tm.counter.count_text(raw)
    tc = tm.count_tokens(prepared, model)
    logger.info(f"[{request_id}] ← {result.provider_name} model={result.model_used}")
    return build_completion(raw, model, tc.prompt_tokens, comp)


USE_FASTAPI = False

try:
    from fastapi import FastAPI, Request, Header, Query
    from fastapi.responses import StreamingResponse, JSONResponse
    from fastapi.middleware.cors import CORSMiddleware
    import uvicorn
    USE_FASTAPI = True
    logger.info("Using FastAPI")
except ImportError:
    logger.warning("FastAPI not available, using Flask")


if USE_FASTAPI:
    app = FastAPI(title="g4f-Bridge", version=BRIDGE_VERSION, docs_url=None, redoc_url=None)
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

    @app.on_event("startup")
    async def _startup() -> None:
        logger.info("=" * 60)
        logger.info(f"  g4f-Bridge v{BRIDGE_VERSION}")
        logger.info("=" * 60)

        premium = get_premium_adapter()
        if premium.is_enabled():
            logger.info("✓ Premium API enabled")

        logger.info("Scanning g4f providers...")
        get_scanner().scan()

        test_worker = get_test_worker()
        working = await test_worker.run_startup_test()

        if working == 0 and not premium.is_enabled():
            logger.error("⚠ NO WORKING PROVIDERS!")
        else:
            logger.info(f"✓ {working} working g4f combinations")

        await test_worker.start_scheduler()
        await get_update_scheduler().start()

        logger.info("=" * 60)
        logger.info(f"  Ready on :{config.server.port}")
        logger.info(f"  API Key: {config.server.api_key}")
        logger.info("=" * 60)

    @app.on_event("shutdown")
    async def _shutdown() -> None:
        logger.info("Shutting down...")
        await get_test_worker().stop()
        await get_update_scheduler().stop()
        save_config()

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request, authorization: Optional[str] = Header(None)) -> Any:
        rid = _gen_request_id()
        if not _verify_key(authorization):
            return JSONResponse(status_code=401, content=build_error("Invalid API key", "authentication_error"), headers={"X-Request-ID": rid})
        try:
            body = await request.json()
        except Exception:
            return JSONResponse(status_code=400, content=build_error("Invalid JSON", "invalid_request_error"), headers={"X-Request-ID": rid})

        result = await handle_chat_request(body, rid)

        if isinstance(result, dict) and result.get("__stream_premium__"):
            return StreamingResponse(result["generator"], media_type="text/event-stream", headers={"X-Request-ID": rid})

        if isinstance(result, dict) and result.get("__stream__"):
            async def gen() -> AsyncGenerator[str, None]:
                chunk_id = _gen_id()
                try:
                    r = await result["router"]._route_g4f(result["model"], result["messages"], True, rid, temperature=result.get("temperature", 0.7), max_tokens=result.get("max_tokens", 4096))
                    if not r.success:
                        yield build_chunk("", result["model"], chunk_id, "stop")
                        yield "data: [DONE]\n\n"
                        return
                    resp = r.response
                    if hasattr(resp, "__aiter__"):
                        async for c in resp:
                            if c:
                                yield build_chunk(str(c), result["model"], chunk_id)
                    elif hasattr(resp, "__iter__") and not isinstance(resp, str):
                        for c in resp:
                            if c:
                                yield build_chunk(str(c), result["model"], chunk_id)
                    else:
                        yield build_chunk(str(resp), result["model"], chunk_id)
                    yield build_chunk("", result["model"], chunk_id, "stop")
                    yield "data: [DONE]\n\n"
                except Exception as e:
                    yield build_chunk(f"Error: {e}", result["model"], chunk_id, "stop")
                    yield "data: [DONE]\n\n"
            return StreamingResponse(gen(), media_type="text/event-stream", headers={"X-Request-ID": rid})

        if "error" in result:
            return JSONResponse(status_code=502 if result["error"].get("type") != "invalid_request_error" else 400, content=result, headers={"X-Request-ID": rid})

        return JSONResponse(content=ensure_format(result), headers={"X-Request-ID": rid})

    @app.get("/v1/models")
    async def list_models(authorization: Optional[str] = Header(None)) -> Any:
        if not _verify_key(authorization):
            return JSONResponse(status_code=401, content=build_error("Invalid API key", "authentication_error"))
        return JSONResponse(content=build_models_list(get_test_worker().get_working_models()))

    @app.get("/v1/models/{model_id:path}")
    async def get_model(model_id: str, authorization: Optional[str] = Header(None)) -> Any:
        if not _verify_key(authorization):
            return JSONResponse(status_code=401, content=build_error("Invalid API key", "authentication_error"))
        clean = _strip_prefix(model_id)
        models = get_test_worker().get_working_models()
        if clean in models or model_id in models:
            return JSONResponse(content={"id": model_id, "object": "model", "created": int(time.time()), "owned_by": "g4f-bridge"})
        return JSONResponse(status_code=404, content=build_error(f"Model '{model_id}' not found", "invalid_request_error"))

    @app.get("/health")
    async def health() -> Any:
        return {"status": "healthy", "timestamp": datetime.now().isoformat(), "version": BRIDGE_VERSION, "providers": get_test_worker().get_summary()}

    @app.get("/status")
    async def status(authorization: Optional[str] = Header(None)) -> Any:
        if not _verify_key(authorization):
            return JSONResponse(status_code=401, content=build_error("Unauthorized", "authentication_error"))
        return {"version": BRIDGE_VERSION, "testing": get_test_worker().get_summary(), "working_models": get_test_worker().get_working_models()}

    @app.get("/test-results")
    async def test_results(authorization: Optional[str] = Header(None)) -> Any:
        if not _verify_key(authorization):
            return JSONResponse(status_code=401, content=build_error("Unauthorized", "authentication_error"))
        return JSONResponse(content=get_test_worker().get_results_dict())

    @app.post("/scan")
    async def trigger_scan(authorization: Optional[str] = Header(None)) -> Any:
        if not _verify_key(authorization):
            return JSONResponse(status_code=401, content=build_error("Unauthorized", "authentication_error"))
        get_scanner().scan()
        working = await get_test_worker().run_startup_test()
        return {"status": "done", "working": working}

    @app.get("/providers")
    async def providers(authorization: Optional[str] = Header(None)) -> Any:
        if not _verify_key(authorization):
            return JSONResponse(status_code=401, content=build_error("Unauthorized", "authentication_error"))
        worker = get_test_worker()
        return {"working": [{"provider": r.provider, "model": r.model, "status": r.status, "fc_score": r.fc_score} for key in worker.working_combinations for r in [worker.results.get(key)] if r]}

    @app.get("/api-key")
    async def api_key_ep(request: Request) -> Any:
        if request.client and request.client.host not in ("127.0.0.1", "::1", "localhost"):
            return JSONResponse(status_code=403, content=build_error("Forbidden", "authentication_error"))
        return {"api_key": config.server.api_key}

    @app.get("/config")
    async def get_cfg(authorization: Optional[str] = Header(None)) -> Any:
        if not _verify_key(authorization):
            return JSONResponse(status_code=401, content=build_error("Unauthorized", "authentication_error"))
        return config.to_safe_dict()

else:
    from flask import Flask, request as flask_request, jsonify
    app = Flask(__name__)

    def _flask_auth() -> bool:
        return _verify_key(flask_request.headers.get("Authorization"))

    @app.before_first_request
    def _flask_startup() -> None:
        get_scanner().scan()
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(get_test_worker().run_startup_test())
        finally:
            loop.close()

    @app.route("/v1/chat/completions", methods=["POST"])
    def flask_chat() -> Any:
        rid = _gen_request_id()
        if not _flask_auth():
            return jsonify(build_error("Invalid API key", "authentication_error")), 401
        body = flask_request.get_json(silent=True) or {}
        loop = asyncio.new_event_loop()
        try:
            result = loop.run_until_complete(handle_chat_request(body, rid))
        finally:
            loop.close()
        if isinstance(result, dict) and (result.get("__stream__") or result.get("__stream_premium__")):
            return jsonify(build_error("Streaming requires FastAPI", "invalid_request_error")), 400
        if "error" in result:
            return jsonify(result), 502 if result["error"].get("type") != "invalid_request_error" else 400
        resp = jsonify(ensure_format(result))
        resp.headers["X-Request-ID"] = rid
        return resp

    @app.route("/v1/models", methods=["GET"])
    def flask_models() -> Any:
        if not _flask_auth():
            return jsonify(build_error("Invalid API key", "authentication_error")), 401
        return jsonify(build_models_list(get_test_worker().get_working_models()))

    @app.route("/health", methods=["GET"])
    def flask_health() -> Any:
        return jsonify({"status": "healthy", "version": BRIDGE_VERSION})

    @app.route("/api-key", methods=["GET"])
    def flask_api_key() -> Any:
        if flask_request.remote_addr not in ("127.0.0.1", "::1"):
            return jsonify(build_error("Forbidden")), 403
        return jsonify({"api_key": config.server.api_key})


def handle_signal(signum: int, frame: Any) -> None:
    logger.info(f"Signal {signum}")
    save_config()
    sys.exit(0)


def print_banner() -> None:
    premium = get_premium_adapter()
    p_status = "✅ ON" if premium.is_enabled() else "❌ OFF"
    print()
    print("╔════════════════════════════════════════════════════════════╗")
    print(f"║  g4f-Bridge v{BRIDGE_VERSION}                                      ║")
    print("╠════════════════════════════════════════════════════════════╣")
    print(f"║  Server:  http://{config.server.host}:{config.server.port}                          ║")
    print(f"║  API Key: {config.server.api_key:<42} ║")
    print(f"║  Premium: {p_status:<42} ║")
    print("╠════════════════════════════════════════════════════════════╣")
    print("║  PicoClaw config:                                          ║")
    print(f"║    api_base = http://127.0.0.1:{config.server.port}/v1                   ║")
    print(f"║    api_key  = {config.server.api_key:<42} ║")
    print("╚════════════════════════════════════════════════════════════╝")
    print()


def main() -> None:
    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)
    print_banner()
    if USE_FASTAPI:
        uvicorn.run(app, host=config.server.host, port=config.server.port, log_level="warning", access_log=False)
    else:
        app.run(host=config.server.host, port=config.server.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()