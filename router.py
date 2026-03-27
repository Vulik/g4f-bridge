"""
Smart router with Provider Locking support.
Dispatches requests to g4f providers with fallback and retry.
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
    """Routes requests to g4f providers with fallback and provider locking."""

    def __init__(self) -> None:
        from config import get_config

        self.logger = get_logger("router")
        cfg = get_config()

        self.timeout = cfg.g4f.timeout_seconds
        self.max_retries = cfg.g4f.max_retries_per_provider
        self.max_fallbacks = cfg.g4f.max_provider_fallbacks

        # Provider locking
        self.strict_mode = cfg.provider_lock.strict_provider_mode
        self.locked_provider = cfg.provider_lock.locked_provider
        self.locked_model = cfg.provider_lock.locked_model
        self.fail_on_lock_error = cfg.provider_lock.fail_on_lock_error

        if self.strict_mode:
            self.logger.info(
                f"🔒 STRICT PROVIDER MODE: "
                f"provider={self.locked_provider or 'any'}, "
                f"model={self.locked_model or 'any'}"
            )

    async def route(
        self,
        model: str,
        messages: List[Dict[str, str]],
        stream: bool = False,
        **kwargs: Any,
    ) -> RoutingResult:
        """Route request to best provider."""
        from scanner import get_scanner
        from resilience import get_resilience

        scanner = get_scanner()
        resilience = get_resilience()

        # ── Provider Locking ──────────────────────────────────
        if self.strict_mode:
            return await self._route_strict(
                model, messages, stream, scanner, resilience, **kwargs
            )

        # ── Normal Routing (with fallback) ────────────────────
        return await self._route_normal(
            model, messages, stream, scanner, resilience, **kwargs
        )

    async def _route_strict(
        self,
        model: str,
        messages: List[Dict[str, str]],
        stream: bool,
        scanner: Any,
        resilience: Any,
        **kwargs: Any,
    ) -> RoutingResult:
        """Route with strict provider locking (no fallback)."""

        # Use locked model if specified
        effective_model = self.locked_model or model

        # Find locked provider
        if self.locked_provider:
            providers = [
                p for p in scanner.get_providers_for_model(
                    effective_model, streaming=stream
                )
                if p.name == self.locked_provider
            ]
        else:
            providers = scanner.get_providers_for_model(
                effective_model, streaming=stream
            )[:1]  # Only first (best) provider

        if not providers:
            error_msg = (
                f"🔒 Locked provider '{self.locked_provider}' not available "
                f"for model '{effective_model}'"
            )
            self.logger.error(error_msg)

            if self.fail_on_lock_error:
                return RoutingResult(success=False, error=error_msg)

            # Fallback to normal routing
            self.logger.warning("Falling back to normal routing")
            return await self._route_normal(
                model, messages, stream, scanner, resilience, **kwargs
            )

        pinfo = providers[0]
        self.logger.info(
            f"🔒 Strict mode: using {pinfo.name} "
            f"model={effective_model}"
        )

        # Try locked provider with retries
        for retry in range(self.max_retries):
            try:
                actual_model = self._pick_model(pinfo, effective_model, scanner)

                t0 = time.time()

                if stream:
                    gen = await self._call_stream(
                        pinfo, actual_model, messages, **kwargs
                    )
                    elapsed = time.time() - t0
                    scanner.update_provider_status(pinfo.name, True, elapsed)
                    return RoutingResult(
                        success=True, response=gen,
                        provider_name=pinfo.name,
                        model_used=actual_model,
                        response_time=elapsed, attempts=retry + 1,
                    )
                else:
                    text = await self._call_sync(
                        pinfo, actual_model, messages, **kwargs
                    )
                    elapsed = time.time() - t0

                    if not text or not text.strip():
                        raise ValueError("Empty response")

                    scanner.update_provider_status(pinfo.name, True, elapsed)
                    self.logger.info(
                        f"✓ {pinfo.name} ({elapsed:.2f}s)"
                    )
                    return RoutingResult(
                        success=True, response=text,
                        provider_name=pinfo.name,
                        model_used=actual_model,
                        response_time=elapsed, attempts=retry + 1,
                    )

            except Exception as e:
                self.logger.warning(
                    f"✗ {pinfo.name} attempt {retry+1}: {e}"
                )
                scanner.update_provider_status(
                    pinfo.name, False, error=str(e)
                )
                if retry < self.max_retries - 1:
                    await asyncio.sleep(2 ** retry)

        error_msg = (
            f"🔒 Locked provider '{pinfo.name}' failed "
            f"after {self.max_retries} attempts"
        )

        if self.fail_on_lock_error:
            return RoutingResult(success=False, error=error_msg)

        # Fallback
        self.logger.warning("Lock failed, falling back to normal routing")
        return await self._route_normal(
            model, messages, stream, scanner, resilience, **kwargs
        )

    async def _route_normal(
        self,
        model: str,
        messages: List[Dict[str, str]],
        stream: bool,
        scanner: Any,
        resilience: Any,
        **kwargs: Any,
    ) -> RoutingResult:
        """Normal routing with fallback chain."""
        providers = scanner.get_providers_for_model(model, streaming=stream)

        if not providers:
            if model != "auto":
                self.logger.warning(
                    f"No providers for '{model}', trying auto"
                )
                providers = scanner.get_providers_for_model(
                    "auto", streaming=stream
                )

            if not providers:
                return RoutingResult(
                    success=False,
                    error=f"No providers for model: {model}",
                )

        self.logger.info(
            f"Routing '{model}' → {len(providers)} candidates"
        )

        attempts = 0
        last_error = ""

        for pinfo in providers[:self.max_fallbacks]:
            pname = pinfo.name

            available, reason = resilience.is_provider_available(pname)
            if not available:
                self.logger.debug(f"Skip {pname}: {reason}")
                continue

            rl = resilience.get_rate_limiter(pname)
            if not rl.acquire():
                self.logger.debug(f"Skip {pname}: rate limited")
                continue

            for retry in range(self.max_retries):
                attempts += 1
                try:
                    actual_model = self._pick_model(pinfo, model, scanner)
                    self.logger.debug(
                        f"Attempt {attempts}: {pname} "
                        f"model={actual_model}"
                    )

                    t0 = time.time()

                    if stream:
                        gen = await self._call_stream(
                            pinfo, actual_model, messages, **kwargs
                        )
                        elapsed = time.time() - t0
                        scanner.update_provider_status(
                            pname, True, elapsed
                        )
                        resilience.get_circuit_breaker(pname).record_success()
                        return RoutingResult(
                            success=True, response=gen,
                            provider_name=pname,
                            model_used=actual_model,
                            response_time=elapsed, attempts=attempts,
                        )
                    else:
                        text = await self._call_sync(
                            pinfo, actual_model, messages, **kwargs
                        )
                        elapsed = time.time() - t0

                        if not text or not text.strip():
                            raise ValueError("Empty response")

                        scanner.update_provider_status(
                            pname, True, elapsed
                        )
                        resilience.get_circuit_breaker(pname).record_success()
                        self.logger.info(
                            f"✓ {pname} ({elapsed:.2f}s, #{attempts})"
                        )
                        return RoutingResult(
                            success=True, response=text,
                            provider_name=pname,
                            model_used=actual_model,
                            response_time=elapsed, attempts=attempts,
                        )

                except Exception as e:
                    last_error = str(e)
                    self.logger.warning(f"✗ {pname}: {last_error}")
                    scanner.update_provider_status(
                        pname, False, error=last_error
                    )
                    cb = resilience.get_circuit_breaker(pname)
                    cb.record_failure()

                    if not cb.is_available():
                        break

                    if retry < self.max_retries - 1:
                        await asyncio.sleep(min(2 ** retry, 10))

        return RoutingResult(
            success=False,
            error=f"All providers failed ({attempts} attempts): {last_error}",
            attempts=attempts,
        )

    async def _call_sync(
        self, pinfo: Any, model: str,
        messages: List[Dict[str, str]], **kwargs: Any,
    ) -> str:
        """Call g4f provider (non-streaming)."""
        import g4f  # type: ignore

        loop = asyncio.get_event_loop()
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
        """Call g4f provider (streaming)."""
        import g4f  # type: ignore

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

    @staticmethod
    def _pick_model(pinfo: Any, requested: str, scanner: Any) -> str:
        if requested in pinfo.models:
            return requested
        for supported in pinfo.models:
            if scanner.model_matches(supported, requested):
                return supported
        return pinfo.models[0] if pinfo.models else requested


_router: Optional[ProviderRouter] = None


def get_router() -> ProviderRouter:
    global _router
    if _router is None:
        _router = ProviderRouter()
    return _router