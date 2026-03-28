"""
Token management — simplified & stateless.
No sessions, no continuation. Just counting + smart truncation.
"""

import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from logger_setup import get_logger


# ═══════════════════════════════════════════════════════════════
# Model Token Limits (for premium enforcement)
# ═══════════════════════════════════════════════════════════════

MODEL_TOKEN_LIMITS: Dict[str, int] = {
    # OpenAI
    "gpt-4o": 128000,
    "gpt-4o-mini": 128000,
    "gpt-4-turbo": 128000,
    "gpt-4": 8192,
    "gpt-3.5-turbo": 16385,
    "gpt-3.5-turbo-16k": 16385,
    # Anthropic
    "claude-3-opus": 200000,
    "claude-3-sonnet": 200000,
    "claude-3.5-sonnet": 200000,
    "claude-3-haiku": 200000,
    # Google
    "gemini-1.5-pro": 1000000,
    "gemini-1.5-flash": 1000000,
    "gemini-pro": 30720,
    # Meta
    "llama-3.3-70b-versatile": 32768,
    "llama-3.1-70b": 131072,
    # Mistral
    "mixtral-8x7b-32768": 32768,
    "mistral-large": 32768,
}

DEFAULT_TOKEN_LIMIT = 8192


@dataclass
class TokenCount:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    def __post_init__(self) -> None:
        self.total_tokens = self.prompt_tokens + self.completion_tokens


# ═══════════════════════════════════════════════════════════════
# Token Counter (dual-mode: tiktoken or fallback)
# ═══════════════════════════════════════════════════════════════

class TokenCounter:
    def __init__(self) -> None:
        from environment import get_environment

        self.logger = get_logger("token_counter")
        env = get_environment()

        self._encoding = None
        self.use_tiktoken = False

        if env.tiktoken_available:
            try:
                import tiktoken  # type: ignore
                self._encoding = tiktoken.get_encoding("cl100k_base")
                self.use_tiktoken = True
                self.logger.info("Using tiktoken for token counting")
            except Exception as e:
                self.logger.warning(f"tiktoken init failed: {e}")

        if not self.use_tiktoken:
            self.logger.info("Using fallback token estimator")

    def count_text(self, text: str) -> int:
        if not text:
            return 0
        if self.use_tiktoken and self._encoding:
            try:
                return len(self._encoding.encode(text))
            except Exception:
                pass
        return self._fallback_count(text)

    @staticmethod
    def _fallback_count(text: str) -> int:
        if not text:
            return 0
        ascii_chars = sum(1 for c in text if ord(c) < 128)
        ratio = ascii_chars / max(len(text), 1)
        if ratio > 0.8:
            return max(1, len(text) // 4)
        else:
            return max(1, len(text) // 2)

    def count_messages(self, messages: List[Dict[str, str]]) -> int:
        total = 0
        for msg in messages:
            total += 4
            total += self.count_text(msg.get("content", "") or "")
            if "name" in msg:
                total += self.count_text(msg["name"]) + 1
            # Count tool_calls if present
            tc = msg.get("tool_calls")
            if tc:
                import json as _json
                total += self.count_text(_json.dumps(tc))
        total += 2
        return total


# ═══════════════════════════════════════════════════════════════
# Token Manager (stateless)
# ═══════════════════════════════════════════════════════════════

class TokenManager:
    def __init__(self) -> None:
        from config import get_config

        self.logger = get_logger("token_manager")
        self.counter = TokenCounter()
        self.cfg = get_config().token_management

    def prepare_messages(
        self,
        messages: List[Dict[str, str]],
        provider_type: str = "g4f",
        model: str = "gpt-4",
    ) -> List[Dict[str, str]]:
        """
        Prepare messages based on provider type.

        Args:
            messages: Raw messages from PicoClaw.
            provider_type: "premium" or "g4f".
            model: Model name for limit lookup.

        Returns:
            Possibly truncated messages.
        """
        if not messages:
            return messages

        if provider_type == "premium" and self.cfg.premium_enforce_limit:
            limit = MODEL_TOKEN_LIMITS.get(model, DEFAULT_TOKEN_LIMIT)
            # Reserve 20% for completion
            max_prompt = int(limit * 0.8)
            current = self.counter.count_messages(messages)

            if current > max_prompt:
                self.logger.info(
                    f"Premium truncation: {current} > {max_prompt} tokens"
                )
                return self._smart_truncate(messages, max_prompt)
            return messages

        # g4f: pass through (no truncation by default)
        return messages

    def prepare_messages_with_window(
        self,
        messages: List[Dict[str, str]],
        window: int = 0,
    ) -> List[Dict[str, str]]:
        """Apply sliding window (used as fallback on context-too-long error)."""
        if window <= 0:
            window = self.cfg.g4f_sliding_window

        if len(messages) <= window:
            return messages

        system_msgs = [m for m in messages if m.get("role") == "system"]
        other_msgs = [m for m in messages if m.get("role") != "system"]
        return system_msgs + other_msgs[-window:]

    def _smart_truncate(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int,
    ) -> List[Dict[str, str]]:
        """Truncate messages to fit token limit, keeping system + recent."""
        system_msgs = [m for m in messages if m.get("role") == "system"]
        other_msgs = [m for m in messages if m.get("role") != "system"]

        # Always keep system messages
        base_tokens = self.counter.count_messages(system_msgs) if system_msgs else 0
        remaining = max_tokens - base_tokens

        # Add messages from most recent backwards
        kept = []
        for msg in reversed(other_msgs):
            msg_tokens = self.counter.count_messages([msg])
            if remaining - msg_tokens < 0:
                break
            kept.insert(0, msg)
            remaining -= msg_tokens

        return system_msgs + kept

    def count_tokens(
        self,
        messages: List[Dict[str, str]],
        model: str = "gpt-4",
    ) -> TokenCount:
        return TokenCount(prompt_tokens=self.counter.count_messages(messages))


# ═══════════════════════════════════════════════════════════════
# Global singleton
# ═══════════════════════════════════════════════════════════════

import threading

_tm_lock = threading.Lock()
_token_manager: Optional[TokenManager] = None


def get_token_manager() -> TokenManager:
    global _token_manager
    if _token_manager is None:
        with _tm_lock:
            if _token_manager is None:
                _token_manager = TokenManager()
    return _token_manager