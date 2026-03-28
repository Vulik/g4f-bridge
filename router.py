"""
Smart router v4 — Only uses tested & working providers.
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
    error: Optional[str] = None
    response_time: float = 0.0
    attempts: int = 0


class ProviderRouter:
    """Routes requests ONLY to tested & working providers."""

    def __init__(self) -> None:
        from config import get_config

        self.logger = get_logger("router")
        cfg = get_config()

        self.timeout = cfg.g4f.timeout_seconds
        self.max_retries = cfg.g4f.max_retries_per_provider
        self.max_fallbacks = cfg.routing.g4f_max_fallbacks
        self.only_tested = cfg.routing.only_use_tested

    async def route(
        self,
        model: str,
        messages: List[Dict[str, str]],
        stream: bool = False,
        request_id: str = "",
        **kwargs: Any,
    ) -> RoutingResult:
        """Route to working providers only."""

        # Strip prefix: "openai/gpt-4o" → "gpt-4o"
        if "/" in model and model != "auto":
            model = model.split("/", 1)[-1]

        from test_worker import get_test_worker
        from scanner import get_scanner
        from resilience import get_resilience

        test_worker = get_test_worker()
        scanner = get_scanner()
        resilience = get_resilience()

        need_fc = bool(kwargs.get("_has_tools", False))

        # Get ONLY working providers
        working = test_worker.get_working_providers(model, need_fc=need_fc)

        if not working and model != "auto":
            self.logger.warning(
                f"[{request_id}] No working providers for '{model}', "
                f"trying auto"
            )
            working = test_worker.get_working_providers("auto", need_fc=need_fc)

        if not working:
            return RoutingResult(
                success=False,
                error=f"No working providers for model: {model}. Run /scan first.",
            )

        self.logger.info(
            f"[{request_id}] Routing → {len(working)} working providers"
        )

        # Try providers
        attempts = 0
        last_error = ""

        for pname, pmodel, fc_score in working[:self.max_fallbacks]:
            pinfo = scanner.providers.get(pname)
            if not pinfo:
                continue

            available, reason = resilience.is_provider_available(pname)
            if not available:
                self.logger.debug(
                    f"[{request_id}] Skip {pname}: {reason}"
                )
                continue

            rl = resilience.get_rate_limiter(pname)
            if not rl.acquire():
                continue

            for retry in range(self.max_retries):
                attempts += 1

                try:
                    t0 = time.time()

                    if stream:
                        gen = await self._call_stream(
                            pinfo, pmodel, messages, **kwargs
                        )
                        elapsed = time.time() - t0

                        scanner.update_provider_status(pname, True, elapsed)
                        resilience.get_circuit_breaker(pname).record_success()

                        return RoutingResult(
                            success=True, response=gen,
                            provider_name=pname,
                            model_used=pmodel,
                            response_time=elapsed,
                            attempts=attempts,
                        )
                    else:
                        text = await self._call_sync(
                            pinfo, pmodel, messages, **kwargs
                        )
                        elapsed = time.time() - t0

                        if not text or not text.strip():
                            raise ValueError("Empty response")

                        scanner.update_provider_status(pname, True, elapsed)
                        resilience.get_circuit_breaker(pname).record_success()

                        self.logger.info(
                            f"[{request_id}] ← provider={pname} "
                            f"model={pmodel} fc_score={fc_score} "
                            f"time={elapsed:.1f}s"
                        )

                        return RoutingResult(
                            success=True, response=text,
                            provider_name=pname,
                            model_used=pmodel,
                            response_time=elapsed,
                            attempts=attempts,
                        )

                except Exception as e:
                    last_error = str(e)[:100]
                    self.logger.warning(
                        f"[{request_id}] {pname} failed: {last_error}"
                    )
                    scanner.update_provider_status(pname, False, error=last_error)
                    resilience.get_circuit_breaker(pname).record_failure()

                    if retry < self.max_retries - 1:
                        await asyncio.sleep(min(2 ** retry, 5))

        return RoutingResult(
            success=False,
            error=f"All providers failed ({attempts} attempts): {last_error}",
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