"""
Premium API adapter — connects to OpenAI-compatible paid APIs.
Supports native function calling, streaming, and error handling.
"""

import time
import asyncio
from typing import Dict, List, Any, Optional, Tuple, AsyncGenerator
from dataclasses import dataclass
from logger_setup import get_logger


@dataclass
class PremiumResult:
    success: bool
    response: Any = None
    provider_name: str = ""
    model_used: str = ""
    error: Optional[str] = None
    response_time: float = 0.0
    is_stream: bool = False


class PremiumAdapter:
    """
    Handles connections to premium (paid) OpenAI-compatible APIs.
    Uses httpx for async HTTP calls — no extra SDK required.
    """

    def __init__(self) -> None:
        from config import get_config

        self.logger = get_logger("premium")
        self.config = get_config()

    # ── Provider Lookup ───────────────────────────────────

    def find_provider_for_model(
        self, model: str
    ) -> Tuple[Optional[Any], str]:
        """
        Find a premium provider that supports the requested model.

        Returns:
            (PremiumProviderEntry or None, actual_model_name)
        """
        if not self.config.premium_api.enabled:
            return None, model

        for entry in self.config.premium_api.providers:
            if not entry.enabled or not entry.api_key:
                continue

            if model == "auto":
                if entry.models:
                    return entry, entry.models[0]
                continue

            if model in entry.models:
                return entry, model

            # Fuzzy: check if model is a substring
            for m in entry.models:
                if model.lower() in m.lower() or m.lower() in model.lower():
                    return entry, m

        return None, model

    def has_any_provider(self) -> bool:
        """Check if any premium provider is configured and enabled."""
        if not self.config.premium_api.enabled:
            return False
        return any(
            p.enabled and p.api_key
            for p in self.config.premium_api.providers
        )

    # ── API Call (non-streaming) ──────────────────────────

    async def call(
        self,
        provider: Any,
        model: str,
        messages: List[Dict[str, str]],
        tools: Optional[List[Dict]] = None,
        tool_choice: Any = "auto",
        temperature: float = 0.7,
        max_tokens: int = 4096,
        request_id: str = "",
    ) -> PremiumResult:
        """
        Call premium API (non-streaming).
        Returns full OpenAI-format response dict.
        """
        import httpx

        body: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }

        if tools:
            body["tools"] = tools
            body["tool_choice"] = tool_choice

        url = f"{provider.base_url.rstrip('/')}/chat/completions"
        headers = {
            "Authorization": f"Bearer {provider.api_key}",
            "Content-Type": "application/json",
        }

        masked = self._mask_key(provider.api_key)
        self.logger.debug(
            f"[{request_id}] Premium → {provider.name} "
            f"model={model} key={masked}"
        )

        t0 = time.time()

        try:
            async with httpx.AsyncClient(
                timeout=httpx.Timeout(provider.timeout, connect=10)
            ) as client:
                resp = await client.post(url, json=body, headers=headers)

            elapsed = time.time() - t0

            if resp.status_code != 200:
                error_text = resp.text[:300]
                self.logger.warning(
                    f"[{request_id}] Premium {provider.name} "
                    f"HTTP {resp.status_code}: {error_text}"
                )
                return PremiumResult(
                    success=False,
                    provider_name=provider.name,
                    model_used=model,
                    error=f"HTTP {resp.status_code}: {error_text}",
                    response_time=elapsed,
                )

            data = resp.json()

            self.logger.info(
                f"[{request_id}] ← tier=premium provider={provider.name} "
                f"model={model} time={elapsed:.1f}s"
            )

            return PremiumResult(
                success=True,
                response=data,
                provider_name=provider.name,
                model_used=model,
                response_time=elapsed,
            )

        except httpx.TimeoutException:
            elapsed = time.time() - t0
            return PremiumResult(
                success=False,
                provider_name=provider.name,
                error=f"Timeout after {elapsed:.1f}s",
                response_time=elapsed,
            )
        except Exception as e:
            elapsed = time.time() - t0
            return PremiumResult(
                success=False,
                provider_name=provider.name,
                error=str(e)[:200],
                response_time=elapsed,
            )

    # ── API Call (streaming) ──────────────────────────────

    async def call_stream(
        self,
        provider: Any,
        model: str,
        messages: List[Dict[str, str]],
        tools: Optional[List[Dict]] = None,
        tool_choice: Any = "auto",
        temperature: float = 0.7,
        max_tokens: int = 4096,
        request_id: str = "",
    ) -> AsyncGenerator[str, None]:
        """
        Call premium API with streaming.
        Yields raw SSE lines (proxied from upstream).
        """
        import httpx

        body: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
        }

        if tools:
            body["tools"] = tools
            body["tool_choice"] = tool_choice

        url = f"{provider.base_url.rstrip('/')}/chat/completions"
        headers = {
            "Authorization": f"Bearer {provider.api_key}",
            "Content-Type": "application/json",
        }

        self.logger.info(
            f"[{request_id}] Premium stream → {provider.name} model={model}"
        )

        client = httpx.AsyncClient(
            timeout=httpx.Timeout(provider.timeout, connect=10)
        )

        try:
            async with client.stream(
                "POST", url, json=body, headers=headers
            ) as resp:
                if resp.status_code != 200:
                    text = ""
                    async for chunk in resp.aiter_text():
                        text += chunk
                        if len(text) > 300:
                            break
                    raise Exception(f"HTTP {resp.status_code}: {text[:300]}")

                async for line in resp.aiter_lines():
                    stripped = line.strip()
                    if stripped:
                        yield stripped + "\n\n"
        finally:
            await client.aclose()

    # ── Helpers ───────────────────────────────────────────

    @staticmethod
    def _mask_key(key: str) -> str:
        if not key:
            return "****"
        if len(key) > 8:
            return key[:4] + "..." + key[-4:]
        return "****"


# ═══════════════════════════════════════════════════════════════
# Global singleton
# ═══════════════════════════════════════════════════════════════

import threading

_adapter_lock = threading.Lock()
_adapter: Optional[PremiumAdapter] = None


def get_premium_adapter() -> PremiumAdapter:
    global _adapter
    if _adapter is None:
        with _adapter_lock:
            if _adapter is None:
                _adapter = PremiumAdapter()
    return _adapter