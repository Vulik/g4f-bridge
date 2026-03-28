"""
Smart router v5 — Premium API first, then g4f.
"""

import time
import asyncio
from typing import Dict, List, Any, Optional, AsyncGenerator
from dataclasses import dataclass
from logger_setup import get_logger


@dataclass
class RoutingResult:
    success: bool
    response: Any = None
    provider_name: str = ""
    model_used: str = ""
    tier: str = "g4f"
    error: Optional[str] = None
    response_time: float = 0.0
    attempts: int = 0
    is_premium: bool = False


class ProviderRouter:
    def __init__(self) -> None:
        from config import get_config

        self.logger = get_logger("router")
        cfg = get_config()

        self.timeout = cfg.g4f.timeout_seconds
        self.max_retries = cfg.g4f.max_retries_per_provider
        self.max_fallbacks = cfg.routing.g4f_max_fallbacks

    async def route(
        self,
        model: str,
        messages: List[Dict[str, str]],
        stream: bool = False,
        request_id: str = "",
        tools: Optional[List[Dict]] = None,
        tool_choice: Any = "auto",
        **kwargs: Any,
    ) -> RoutingResult:
        if "/" in model and model != "auto":
            model = model.split("/", 1)[-1]

        from premium_adapter import get_premium_adapter

        adapter = get_premium_adapter()

        if adapter.is_enabled():
            result = await self._route_premium(
                model, messages, stream, request_id,
                tools, tool_choice, **kwargs
            )
            if result.success:
                return result
            self.logger.warning(
                f"[{request_id}] Premium failed: {result.error} | fallback g4f"
            )

        return await self._route_g4f(
            model, messages, stream, request_id, **kwargs
        )

    async def _route_premium(
        self,
        model: str,
        messages: List[Dict[str, str]],
        stream: bool,
        request_id: str,
        tools: Optional[List[Dict]],
        tool_choice: Any,
        **kwargs: Any,
    ) -> RoutingResult:
        from premium_adapter import get_premium_adapter

        adapter = get_premium_adapter()
        provider, actual_model = adapter.find_provider_for_model(model)

        if not provider:
            return RoutingResult(
                success=False, tier="premium",
                error="No premium provider for this model",
            )

        if stream:
            try:
                gen = await adapter.call_stream(
                    provider=provider,
                    model=actual_model,
                    messages=messages,
                    tools=tools,
                    tool_choice=tool_choice,
                    request_id=request_id,
                    **{k: v for k, v in kwargs.items()
                       if k in ("temperature", "max_tokens")},
                )
                return RoutingResult(
                    success=True,
                    response=gen,
                    provider_name=f"premium:{provider.get('name', '')}",
                    model_used=actual_model,
                    tier="premium",
                    is_premium=True,
                )
            except Exception as e:
                return RoutingResult(
                    success=False, tier="premium",
                    error=str(e)[:200],
                )

        result = await adapter.call(
            provider=provider,
            model=actual_model,
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
            request_id=request_id,
            **{k: v for k, v in kwargs.items()
               if k in ("temperature", "max_tokens")},
        )

        if result.success:
            return RoutingResult(
                success=True,
                response=result.response,
                provider_name=f"premium:{result.provider_name}",
                model_used=actual_model,
                tier="premium",
                response_time=result.response_time,
                is_premium=True,
            )

        return RoutingResult(
            success=False, tier="premium",
            provider_name=result.provider_name,
            error=result.error,
            response_time=result.response_time,
        )

    async def _route_g4f(
        self,
        model: str,
        messages: List[Dict[str, str]],
        stream: bool,
        request_id: str,
        **kwargs: Any,
    ) -> RoutingResult:
        from test_worker import get_test_worker
        from scanner import get_scanner
        from resilience import get_resilience

        test_worker = get_test_worker()
        scanner = get_scanner()
        resilience = get_resilience()

        need_fc = bool(kwargs.get("_has_tools", False))

        working = test_worker.get_working_providers(model, need_fc=need_fc)

        if not working and model != "auto":
            self.logger.warning(f"[{request_id}] No providers for '{model}', try auto")
            working = test_worker.get_working_providers("auto", need_fc=need_fc)

        if not working:
            return RoutingResult(
                success=False,
                error=f"No working providers for: {model}",
            )

        self.logger.info(f"[{request_id}] g4f → {len(working)} providers")

        attempts = 0
        last_error = ""

        for pname, pmodel, fc_score in working[:self.max_fallbacks]:
            pinfo = scanner.providers.get(pname)
            if not pinfo:
                continue

            available, reason = resilience.is_provider_available(pname)
            if not available:
                continue

            rl = resilience.get_rate_limiter(pname)
            if not rl.acquire():
                continue

            for retry in range(self.max_retries):
                attempts += 1

                try:
                    t0 = time.time()

                    if stream:
                        gen = await self._call_stream(pinfo, pmodel, messages, **kwargs)
                        elapsed = time.time() - t0
                        scanner.update_provider_status(pname, True, elapsed)
                        resilience.get_circuit_breaker(pname).record_success()
                        return RoutingResult(
                            success=True, response=gen,
                            provider_name=pname, model_used=pmodel,
                            tier="g4f", response_time=elapsed, attempts=attempts,
                        )
                    else:
                        text = await self._call_sync(pinfo, pmodel, messages, **kwargs)
                        elapsed = time.time() - t0

                        if not text or not text.strip():
                            raise ValueError("Empty")

                        scanner.update_provider_status(pname, True, elapsed)
                        resilience.get_circuit_breaker(pname).record_success()

                        self.logger.info(
                            f"[{request_id}] ← g4f:{pname} model={pmodel} "
                            f"fc={fc_score} time={elapsed:.1f}s"
                        )

                        return RoutingResult(
                            success=True, response=text,
                            provider_name=pname, model_used=pmodel,
                            tier="g4f", response_time=elapsed, attempts=attempts,
                        )

                except Exception as e:
                    last_error = str(e)[:100]
                    self.logger.warning(f"[{request_id}] {pname}: {last_error}")
                    scanner.update_provider_status(pname, False, error=last_error)
                    resilience.get_circuit_breaker(pname).record_failure()
                    if retry < self.max_retries - 1:
                        await asyncio.sleep(min(2 ** retry, 5))

        return RoutingResult(
            success=False,
            error=f"All failed ({attempts}): {last_error}",
            attempts=attempts,
        )

    async def _call_sync(
        self, pinfo: Any, model: str,
        messages: List[Dict[str, str]], **kwargs: Any,
    ) -> str:
        import g4f

        g4f_kwargs = {
            k: v for k, v in kwargs.items()
            if k in ("temperature", "max_tokens", "top_p")
        }

        try:
            result = await asyncio.wait_for(
                g4f.ChatCompletion.create_async(
                    model=model,
                    messages=messages,
                    provider=pinfo.class_ref,
                    **g4f_kwargs,
                ),
                timeout=self.timeout,
            )
            return str(result)
        except (AttributeError, TypeError):
            pass

        loop = asyncio.get_event_loop()

        def _sync() -> str:
            return str(g4f.ChatCompletion.create(
                model=model,
                messages=messages,
                provider=pinfo.class_ref,
                **g4f_kwargs,
            ))

        return await asyncio.wait_for(
            loop.run_in_executor(None, _sync),
            timeout=self.timeout,
        )

    async def _call_stream(
        self, pinfo: Any, model: str,
        messages: List[Dict[str, str]], **kwargs: Any,
    ) -> AsyncGenerator[str, None]:
        import g4f

        g4f_kwargs = {
            k: v for k, v in kwargs.items()
            if k in ("temperature", "max_tokens", "top_p")
        }

        try:
            response = g4f.ChatCompletion.create_async(
                model=model, messages=messages,
                provider=pinfo.class_ref, stream=True,
                **g4f_kwargs,
            )

            async def _gen() -> AsyncGenerator[str, None]:
                async for chunk in response:
                    yield str(chunk)
            return _gen()
        except (AttributeError, TypeError):
            pass

        response = g4f.ChatCompletion.create(
            model=model, messages=messages,
            provider=pinfo.class_ref, stream=True,
            **g4f_kwargs,
        )

        async def _wrap() -> AsyncGenerator[str, None]:
            loop = asyncio.get_event_loop()
            it = iter(response)
            while True:
                try:
                    chunk = await loop.run_in_executor(None, next, it)
                    yield str(chunk)
                except StopIteration:
                    break
        return _wrap()


_router: Optional[ProviderRouter] = None


def get_router() -> ProviderRouter:
    global _router
    if _router is None:
        _router = ProviderRouter()
    return _router