"""
Token management — simplified & stateless.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from logger_setup import get_logger


@dataclass
class TokenCount:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    def __post_init__(self) -> None:
        self.total_tokens = self.prompt_tokens + self.completion_tokens


class TokenCounter:
    def __init__(self) -> None:
        from environment import get_environment

        self.logger = get_logger("token_counter")
        env = get_environment()

        self._encoding = None
        self.use_tiktoken = False

        if env.tiktoken_available:
            try:
                import tiktoken
                self._encoding = tiktoken.get_encoding("cl100k_base")
                self.use_tiktoken = True
            except Exception:
                pass

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
            tc = msg.get("tool_calls")
            if tc:
                import json
                total += self.count_text(json.dumps(tc))
        total += 2
        return total


class TokenManager:
    def __init__(self) -> None:
        self.logger = get_logger("token_manager")
        self.counter = TokenCounter()

    def count_tokens(
        self,
        messages: List[Dict[str, str]],
        model: str = "gpt-4",
    ) -> TokenCount:
        return TokenCount(prompt_tokens=self.counter.count_messages(messages))


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