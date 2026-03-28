"""
Smart router v3 — Two-tier routing: Premium API → g4f.
With test-result-based ranking and request ID propagation.
"""

import time
import asyncio
from typing import Dict, List, Any, Optional, AsyncGenerator, Tuple
from dataclasses import dataclass
from logger_setup import get_logger


@dataclass
class RoutingResult:
    success: bool
    response: Any = None
    provider_name: str = ""
    model_used: str = ""
    tier: str = ""  # "premium" or "g4f"
    error: Optional[str] = None
    response_time: float = 0.0
    attempts: int = 0
    is_premium_response: bool = False  # If True, response is full OpenAI dict


class ProviderRouter:
    """Routes requests: Premium API → g4f with fallback."""

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
        """
        Main routing entry point.

        Tier 1: Premium API (native FC, no emulation needed).
        Tier 2: g4f (with FC emulation if needed).
        """
        from config import get_config

        config = get_config()

        # ── Tier 1: Premium API ───────────────────────────
        if config.premium_api.enabled:
            result = await self._route_premium(
                model, messages, stream, request_id,
                tools=tools, tool_choice=tool_choice, **kwargs,
            )
            if result.success:
                return result

            self.logger.warning(
                f"[{request_id}] Premium tier failed: "
                f"{result.error} | fallback to g4f"
            )

        # ── Tier 2: g4f ──────────────────────────────────
        return await self._route_g4f(
            model, messages, stream, request_id, **kwargs
        )

    # ══════════════════════════════════════════════════════
    # Tier 1: Premium
    # ══════════════════════════════════════════════════════

    async def _route_premium(
        self,
        model: str,
        messages: List[Dict[str, str]],
        stream: bool,
        request_id: str,
        tools: Optional[List[Dict]] = None,
        tool_choice: Any = "auto",
        **kwargs: Any,
    ) -> RoutingResult:
        from premium_adapter import get_premium_adapter
        from token_manager import get_token_manager

        adapter = get_premium_adapter()
        provider_entry, actual_model = adapter.find_provider_for_model(model)

        if not provider_entry:
            return RoutingResult(
                success=False, tier="premium",
                error="No premium provider for this model",
            )

        # Apply token limit for premium
        tm = get_token_manager()
        prepared = tm.prepare_messages(messages, "premium", actual_model)

        if stream:
            try:
                gen = await adapter.call_stream(
                    provider=provider_entry,
                    model=actual_model,
                    messages=prepared,
                    tools=tools,
                    tool_choice=tool_choice,
                    request_id=request_id,
                    **{k: v for k, v in kwargs.items()
                       if k in ("temperature", "max_tokens")},
                )
                return RoutingResult(
                    success=True,
                    response=gen,
                    provider_name=f"premium:{provider_entry.name}",
                    model_used=actual_model,
                    tier="premium",
                    is_premium_response=True,
                )
            except Exception as e:
                return RoutingResult(
                    success=False, tier="premium",
                    provider_name=provider_entry.name,
                    error=str(e)[:200],
                )

        # Non-streaming
        result = await adapter.call(
            provider=provider_entry,
            model=actual_model,
            messages=prepared,
            tools=tools,
            tool_choice=tool_choice,
            request_id=request_id,
            **{k: v for k, v in kwargs.items()
               if k in ("temperature", "max_tokens")},
        )

        if result.success:
            return RoutingResult(
                success=True,
                response=result.response,  # Full OpenAI dict
                provider_name=f"premium:{result.provider_name}",
                model_used=actual_model,
                tier="premium",
                response_time=result.response_time,
                is_premium_response=True,
            )
        else:
            return RoutingResult(
                success=False, tier="premium",
                provider_name=result.provider_name,
                error=result.error,
                response_time=result.response_time,
            )

    # ══════════════════════════════════════════════════════
    # Tier 2: g4f
    # ══════════════════════════════════════════════════════

    async def _route_g4f(
        self,
        model: str,
        messages: List[Dict[str, str]],
        stream: bool,
        request_id: str,
        **kwargs: Any,
    ) -> RoutingResult:
        """Route to g4f providers using test-result ranking."""
        from scanner import get_scanner
        from resilience import get_resilience
        from test_worker import get_test_worker

        scanner = get_scanner()
        resilience = get_resilience()
        test_worker = get_test_worker()

        need_fc = bool(kwargs.get("_has_tools", False))

        # Get test-ranked providers first
        ranked = test_worker.get_ranked_providers(model, need_fc=need_fc)

        # Build ordered provider list
        ordered_providers = []

        # Add test-ranked providers
        seen = set()
        for pname, pmodel, fc_score in ranked:
            pinfo = scanner.providers.get(pname)
            if pinfo and pname not in seen:
                ordered_providers.append((pinfo, pmodel))
                seen.add(pname)

        # Add remaining from scanner (not yet ranked)
        scanner_providers = scanner.get_providers_for_model(
            model, streaming=stream
        )
        for pinfo in scanner_providers:
            if pinfo.name not in seen:
                ordered_providers.append((pinfo, model))
                seen.add(pinfo.name)

        # If nothing found, try "auto"
        if not ordered_providers and model != "auto":
            self.logger.warning(
                f"[{request_id}] No providers for '{model}', trying auto"
            )
            auto_ranked = test_worker.get_ranked_providers(
                "auto", need_fc=need_fc
            )
            for pname, pmodel, _ in auto_ranked:
                pinfo = scanner.providers.get(pname)
                if pinfo and pname not in seen:
                    ordered_providers.append((pinfo, pmodel))
                    seen.add(pname)
                    break

            auto_scanner = scanner.get_providers_for_model(
                "auto", streaming=stream
            )
            for pinfo in auto_scanner:
                if pinfo.name not in seen:
                    ordered_providers.append((pinfo, model))
                    seen.add(pinfo.name)

        if not ordered_providers:
            return RoutingResult(
                success=False, tier="g4f",
                error=f"No providers available for model: {model}",
            )

        self.logger.info(
            f"[{request_id}] g4f routing: {len(ordered_providers)} candidates"
        )

        # Try providers with fallback
        attempts = 0
        last_error = ""

        for pinfo, target_model in ordered_providers[:self.max_fallbacks]:
            pname = pinfo.name

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
                actual_model = self._pick_model(
                    pinfo, target_model, scanner
                )

                try:
                    t0 = time.time()

                    if stream:
                        gen = await self._call_stream(
                            pinfo, actual_model, messages, **kwargs
                        )
                        elapsed = time.time() - t0

                        scanner.update_provider_status(
                            pname, True, elapsed
                        )
                        resilience.get_circuit_breaker(
                            pname
                        ).record_success()

                        return RoutingResult(
                            success=True, response=gen,
                            provider_name=pname,
                            model_used=actual_model,
                            tier="g4f",
                            response_time=elapsed,
                            attempts=attempts,
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
                        resilience.get_circuit_breaker(
                            pname
                        ).record_success()

                        self.logger.info(
                            f"[{request_id}] ← tier=g4f "
                            f"provider={pname} "
                            f"model={actual_model} "
                            f"time={elapsed:.1f}s"
                        )

                        return RoutingResult(
                            success=True, response=text,
                            provider_name=pname,
                            model_used=actual_model,
                            tier="g4f",
                            response_time=elapsed,
                            attempts=attempts,
                        )

                except Exception as e:
                    last_error = str(e)
                    self.logger.warning(
                        f"[{request_id}] provider={pname} "
                        f"failed: {last_error[:80]} | "
                        f"fallback=next"
                    )
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
            success=False, tier="g4f",
            error=(
                f"All g4f providers failed "
                f"({attempts} attempts): {last_error}"
            ),
            attempts=attempts,
        )

    # ══════════════════════════════════════════════════════
    # g4f Call Methods (unchanged)
    # ══════════════════════════════════════════════════════

    async def _call_sync(
        self, pinfo: Any, model: str,
        messages: List[Dict[str, str]], **kwargs: Any,
    ) -> str:
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


# ═══════════════════════════════════════════════════════════════
# Global
# ═══════════════════════════════════════════════════════════════

_router: Optional[ProviderRouter] = None


def get_router() -> ProviderRouter:
    global _router
    if _router is None:
        _router = ProviderRouter()
    return _router