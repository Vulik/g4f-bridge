"""
Premium API adapter — connects to OpenAI-compatible paid APIs.
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
    """Handles connections to premium (paid) OpenAI-compatible APIs."""

    def __init__(self) -> None:
        from config import get_config

        self.logger = get_logger("premium")
        self.config = get_config()

    def is_enabled(self) -> bool:
        """Check if premium API is enabled and has providers."""
        if not self.config.premium_api.enabled:
            return False
        return any(
            p.get("enabled", False) and p.get("api_key")
            for p in self.config.premium_api.providers
        )

    def find_provider_for_model(
        self, model: str
    ) -> Tuple[Optional[Dict[str, Any]], str]:
        """Find a premium provider that supports the requested model."""
        if not self.config.premium_api.enabled:
            return None, model

        for entry in self.config.premium_api.providers:
            if not entry.get("enabled", False) or not entry.get("api_key"):
                continue

            models = entry.get("models", [])

            if model == "auto":
                if models:
                    return entry, models[0]
                continue

            if model in models:
                return entry, model

            for m in models:
                if model.lower() in m.lower() or m.lower() in model.lower():
                    return entry, m

        return None, model

    async def call(
        self,
        provider: Dict[str, Any],
        model: str,
        messages: List[Dict[str, str]],
        tools: Optional[List[Dict]] = None,
        tool_choice: Any = "auto",
        temperature: float = 0.7,
        max_tokens: int = 4096,
        request_id: str = "",
    ) -> PremiumResult:
        """Call premium API (non-streaming)."""
        try:
            import httpx
        except ImportError:
            return PremiumResult(
                success=False,
                provider_name=provider.get("name", ""),
                error="httpx not installed. Run: pip install httpx",
            )

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

        base_url = provider.get("base_url", "").rstrip("/")
        url = f"{base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {provider.get('api_key', '')}",
            "Content-Type": "application/json",
        }

        timeout_sec = provider.get("timeout", 60)
        pname = provider.get("name", "premium")

        self.logger.debug(f"[{request_id}] Premium → {pname} model={model}")

        t0 = time.time()

        try:
            async with httpx.AsyncClient(
                timeout=httpx.Timeout(timeout_sec, connect=10)
            ) as client:
                resp = await client.post(url, json=body, headers=headers)

            elapsed = time.time() - t0

            if resp.status_code != 200:
                error_text = resp.text[:300]
                self.logger.warning(
                    f"[{request_id}] Premium {pname} HTTP {resp.status_code}"
                )
                return PremiumResult(
                    success=False,
                    provider_name=pname,
                    model_used=model,
                    error=f"HTTP {resp.status_code}: {error_text}",
                    response_time=elapsed,
                )

            data = resp.json()

            self.logger.info(
                f"[{request_id}] ← tier=premium provider={pname} "
                f"model={model} time={elapsed:.1f}s"
            )

            return PremiumResult(
                success=True,
                response=data,
                provider_name=pname,
                model_used=model,
                response_time=elapsed,
            )

        except Exception as e:
            elapsed = time.time() - t0
            self.logger.warning(f"[{request_id}] Premium {pname} error: {e}")
            return PremiumResult(
                success=False,
                provider_name=pname,
                error=str(e)[:200],
                response_time=elapsed,
            )

    async def call_stream(
        self,
        provider: Dict[str, Any],
        model: str,
        messages: List[Dict[str, str]],
        tools: Optional[List[Dict]] = None,
        tool_choice: Any = "auto",
        temperature: float = 0.7,
        max_tokens: int = 4096,
        request_id: str = "",
    ) -> AsyncGenerator[str, None]:
        """Call premium API with streaming."""
        try:
            import httpx
        except ImportError:
            yield 'data: {"error": "httpx not installed"}\n\n'
            return

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

        base_url = provider.get("base_url", "").rstrip("/")
        url = f"{base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {provider.get('api_key', '')}",
            "Content-Type": "application/json",
        }

        timeout_sec = provider.get("timeout", 60)
        pname = provider.get("name", "premium")

        self.logger.info(f"[{request_id}] Premium stream → {pname} model={model}")

        client = httpx.AsyncClient(
            timeout=httpx.Timeout(timeout_sec, connect=10)
        )

        try:
            async with client.stream("POST", url, json=body, headers=headers) as resp:
                if resp.status_code != 200:
                    text = ""
                    async for chunk in resp.aiter_text():
                        text += chunk
                        if len(text) > 300:
                            break
                    yield f'data: {{"error": "HTTP {resp.status_code}"}}\n\n'
                    return

                async for line in resp.aiter_lines():
                    stripped = line.strip()
                    if stripped:
                        yield stripped + "\n\n"
        finally:
            await client.aclose()


_adapter: Optional[PremiumAdapter] = None


def get_premium_adapter() -> PremiumAdapter:
    global _adapter
    if _adapter is None:
        _adapter = PremiumAdapter()
    return _adapter