"""
Smart router: dispatches to g4f providers with fallback and retry.
"""

import time
import asyncio
import traceback
from typing import Dict, List, Any, Optional, AsyncGenerator, Generator
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
    """Routes requests to g4f providers with fallback."""

    def __init__(self) -> None:
        from config import get_config

        self.logger = get_logger("router")
        cfg = get_config()

        self.timeout = cfg.g4f.timeout_seconds
        self.max_retries = cfg.g4f.max_retries_per_provider
        self.max_fallbacks = cfg.g4f.max_provider_fallbacks

    async def route(
        self,
        model: str,
        messages: List[Dict[str, str]],
        stream: bool = False,
        **kwargs: Any,
    ) -> RoutingResult:
        """Route request to best provider with fallback chain."""
        from scanner import get_scanner
        from resilience import get_resilience

        scanner = get_scanner()
        resilience = get_resilience()

        providers = scanner.get_providers_for_model(model, streaming=stream)

        if not providers:
            # Try "auto" as fallback
            if model != "auto":
                self.logger.warning(f"No providers for '{model}', trying auto")
                providers = scanner.get_providers_for_model("auto", streaming=stream)

            if not providers:
                return RoutingResult(
                    success=False,
                    error=f"No providers available for model: {model}",
                )

        self.logger.info(f"Routing '{model}' → {len(providers)} candidates")

        attempts = 0
        last_error = ""

        for pinfo in providers[:self.max_fallbacks]:
            pname = pinfo.name

            # Resilience checks
            available, reason = resilience.is_provider_available(pname)
            if not available:
                self.logger.debug(f"Skip {pname}: {reason}")
                continue

            # Consume rate-limit slot
            rl = resilience.get_rate_limiter(pname)
            if not rl.acquire():
                self.logger.debug(f"Skip {pname}: rate limited")
                continue

            for retry in range(self.max_retries):
                attempts += 1
                try:
                    actual_model = self._pick_model(pinfo, model, scanner)
                    self.logger.debug(
                        f"Attempt {attempts}: {pname} model={actual_model} "
                        f"(retry {retry+1}/{self.max_retries})"
                    )

                    t0 = time.time()

                    if stream:
                        gen = await self._call_stream(pinfo, actual_model, messages, **kwargs)
                        elapsed = time.time() - t0

                        scanner.update_provider_status(pname, True, elapsed)
                        resilience.get_circuit_breaker(pname).record_success()

                        return RoutingResult(
                            success=True,
                            response=gen,
                            provider_name=pname,
                            model_used=actual_model,
                            response_time=elapsed,
                            attempts=attempts,
                        )
                    else:
                        text = await self._call_sync(pinfo, actual_model, messages, **kwargs)
                        elapsed = time.time() - t0

                        if not text or not text.strip():
                            raise ValueError("Empty response from provider")

                        scanner.update_provider_status(pname, True, elapsed)
                        resilience.get_circuit_breaker(pname).record_success()

                        self.logger.info(f"✓ {pname} ({elapsed:.2f}s, attempt {attempts})")

                        return RoutingResult(
                            success=True,
                            response=text,
                            provider_name=pname,
                            model_used=actual_model,
                            response_time=elapsed,
                            attempts=attempts,
                        )

                except Exception as e:
                    last_error = str(e)
                    self.logger.warning(f"✗ {pname}: {last_error}")

                    scanner.update_provider_status(pname, False, error=last_error)
                    cb = resilience.get_circuit_breaker(pname)
                    cb.record_failure()

                    if not cb.is_available():
                        self.logger.info(f"Circuit opened for {pname}, next provider")
                        break

                    if retry < self.max_retries - 1:
                        wait = min(2 ** retry, 10)
                        await asyncio.sleep(wait)

        return RoutingResult(
            success=False,
            error=f"All providers failed ({attempts} attempts). Last: {last_error}",
            attempts=attempts,
        )

    async def _call_sync(
        self, pinfo: Any, model: str, messages: List[Dict[str, str]], **kwargs: Any
    ) -> str:
        """Call g4f provider (non-streaming)."""
        import g4f  # type: ignore

        loop = asyncio.get_event_loop()

        # Remove unsupported kwargs for g4f
        g4f_kwargs = {
            k: v for k, v in kwargs.items()
            if k in ("temperature", "max_tokens", "top_p")
        }

        # Strategy 1: Try create_async
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

        # Strategy 2: Sync in executor
        def _sync_call() -> str:
            return str(g4f.ChatCompletion.create(
                model=model,
                messages=messages,
                provider=pinfo.class_ref,
                **g4f_kwargs,
            ))

        result = await asyncio.wait_for(
            loop.run_in_executor(None, _sync_call),
            timeout=self.timeout,
        )
        return result

    async def _call_stream(
        self, pinfo: Any, model: str, messages: List[Dict[str, str]], **kwargs: Any
    ) -> AsyncGenerator[str, None]:
        """Call g4f provider (streaming). Returns an async generator."""
        import g4f  # type: ignore

        g4f_kwargs = {
            k: v for k, v in kwargs.items()
            if k in ("temperature", "max_tokens", "top_p")
        }

        # Try async streaming
        try:
            response = g4f.ChatCompletion.create_async(
                model=model,
                messages=messages,
                provider=pinfo.class_ref,
                stream=True,
                **g4f_kwargs,
            )

            async def _async_gen() -> AsyncGenerator[str, None]:
                async for chunk in response:
                    yield str(chunk)

            return _async_gen()
        except (AttributeError, TypeError):
            pass

        # Fallback: sync streaming wrapped
        response = g4f.ChatCompletion.create(
            model=model,
            messages=messages,
            provider=pinfo.class_ref,
            stream=True,
            **g4f_kwargs,
        )

        async def _sync_gen_wrapper() -> AsyncGenerator[str, None]:
            loop = asyncio.get_event_loop()
            it = iter(response)
            while True:
                try:
                    chunk = await loop.run_in_executor(None, next, it)
                    yield str(chunk)
                except StopIteration:
                    break

        return _sync_gen_wrapper()

    @staticmethod
    def _pick_model(pinfo: Any, requested: str, scanner: Any) -> str:
        """Select actual model name."""
        if requested in pinfo.models:
            return requested

        for supported in pinfo.models:
            if scanner.model_matches(supported, requested):
                return supported

        return pinfo.models[0] if pinfo.models else requested


# Global
_router: Optional[ProviderRouter] = None


def get_router() -> ProviderRouter:
    global _router
    if _router is None:
        _router = ProviderRouter()
    return _router