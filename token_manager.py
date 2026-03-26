"""
Token management: dual-mode counting, session rotation, auto-continuation.
"""

import re
import uuid
import threading
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from collections import OrderedDict
from logger_setup import get_logger


@dataclass
class TokenCount:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    def __post_init__(self) -> None:
        self.total_tokens = self.prompt_tokens + self.completion_tokens


class TokenCounter:
    """Dual-mode token counter."""

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
        """Count tokens in text."""
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
        """Estimate token count without tiktoken (~85-90% accuracy)."""
        if not text:
            return 0
        ascii_chars = sum(1 for c in text if ord(c) < 128)
        ratio = ascii_chars / max(len(text), 1)

        if ratio > 0.8:
            # English: ~1 token per 4 characters
            return max(1, len(text) // 4)
        else:
            # CJK/mixed: ~1 token per 2-3 characters
            return max(1, len(text) // 2)

    def count_messages(self, messages: List[Dict[str, str]]) -> int:
        """Count tokens in message list."""
        total = 0
        for msg in messages:
            total += 4  # role + formatting overhead
            total += self.count_text(msg.get("content", ""))
            if "name" in msg:
                total += self.count_text(msg["name"]) + 1
        total += 2  # conversation framing
        return total


class LRUSessionCache(OrderedDict):
    """LRU cache for sessions to prevent memory leaks."""

    def __init__(self, maxsize: int = 200) -> None:
        super().__init__()
        self.maxsize = maxsize

    def __setitem__(self, key: str, value: Any) -> None:
        if key in self:
            self.move_to_end(key)
        super().__setitem__(key, value)
        while len(self) > self.maxsize:
            self.popitem(last=False)


@dataclass
class Session:
    id: str
    conversation_id: str
    messages: List[Dict[str, str]] = field(default_factory=list)
    token_count: int = 0
    created_at: datetime = field(default_factory=datetime.now)
    active: bool = True


class ContinuationHandler:
    """Detects truncated responses and creates continuation prompts."""

    def __init__(self, max_continuations: int = 5) -> None:
        self.max_continuations = max_continuations

    @staticmethod
    def needs_continuation(response: str, finish_reason: str) -> bool:
        if finish_reason == "length":
            return True
        if not response:
            return False

        stripped = response.rstrip()

        # Unclosed code blocks
        if stripped.count("```") % 2 != 0:
            return True

        # Unclosed brackets (only if significant imbalance)
        for o, c in [("{", "}"), ("[", "]"), ("(", ")")]:
            if abs(stripped.count(o) - stripped.count(c)) > 2:
                return True

        return False

    @staticmethod
    def create_continuation_prompt(last_response: str) -> str:
        ctx = last_response[-80:].strip()
        return f"Continue exactly from where you left off. Last text was: \"{ctx}\""

    @staticmethod
    def merge_responses(responses: List[str]) -> str:
        if not responses:
            return ""
        merged = responses[0]
        for i in range(1, len(responses)):
            part = responses[i]
            # Try to remove echoed context
            part = re.sub(
                r'^(Continue|Continuing|Last text was).*?\n',
                '', part, count=1, flags=re.IGNORECASE,
            )
            merged += part
        return merged


class TokenManager:
    """Main token manager."""

    def __init__(self) -> None:
        from config import get_config
        from environment import get_environment

        cfg = get_config().token_manager
        env = get_environment()

        self.logger = get_logger("token_manager")
        self.counter = TokenCounter()
        self.continuation = ContinuationHandler(cfg.max_continuations)

        self.max_tokens = cfg.max_tokens_per_session
        self.enable_continuation = cfg.enable_auto_continuation
        self.sliding_window = cfg.sliding_window_messages

        max_sessions = 100 if env.lightweight_mode else 500
        self._sessions = LRUSessionCache(maxsize=max_sessions)
        self._lock = threading.Lock()

    def prepare_messages(
        self,
        messages: List[Dict[str, str]],
        conversation_id: Optional[str] = None,
    ) -> Tuple[List[Dict[str, str]], str]:
        """Prepare messages with sliding window and session management."""
        if conversation_id is None:
            conversation_id = str(uuid.uuid4())

        # Apply sliding window
        if self.sliding_window > 0:
            messages = self._apply_sliding_window(messages)

        # Get or create session
        session_id = self._get_or_create_session(conversation_id, messages)

        return messages, session_id

    def _apply_sliding_window(
        self, messages: List[Dict[str, str]]
    ) -> List[Dict[str, str]]:
        if len(messages) <= self.sliding_window:
            return messages

        system_msgs = [m for m in messages if m.get("role") == "system"]
        other_msgs = [m for m in messages if m.get("role") != "system"]
        return system_msgs + other_msgs[-self.sliding_window:]

    def _get_or_create_session(
        self, conversation_id: str, messages: List[Dict[str, str]]
    ) -> str:
        with self._lock:
            # Find active session
            for sid, session in self._sessions.items():
                if session.conversation_id == conversation_id and session.active:
                    token_count = self.counter.count_messages(messages)
                    if session.token_count + token_count > self.max_tokens:
                        session.active = False
                        self.logger.info(f"Session {sid} rotated (token limit)")
                        break
                    session.messages = messages
                    session.token_count = token_count
                    return sid

            # Create new session
            sid = str(uuid.uuid4())
            self._sessions[sid] = Session(
                id=sid,
                conversation_id=conversation_id,
                messages=messages,
                token_count=self.counter.count_messages(messages),
            )
            return sid

    def count_tokens(self, messages: List[Dict[str, str]], model: str = "gpt-4") -> TokenCount:
        return TokenCount(prompt_tokens=self.counter.count_messages(messages))

    def handle_response(
        self, response: str, finish_reason: str
    ) -> Tuple[str, bool]:
        needs_cont = False
        if self.enable_continuation:
            needs_cont = self.continuation.needs_continuation(response, finish_reason)
        return response, needs_cont


# Global
_tm_lock = threading.Lock()
_token_manager: Optional[TokenManager] = None


def get_token_manager() -> TokenManager:
    global _token_manager
    if _token_manager is None:
        with _tm_lock:
            if _token_manager is None:
                _token_manager = TokenManager()
    return _token_manager