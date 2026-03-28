"""
schemas.py
~~~~~~~~~~
Pydantic models yang mereplikasi format OpenAI Chat Completions API
agar PicoClaw bisa berkomunikasi tanpa adaptasi tambahan.
"""

from __future__ import annotations

import time
import uuid
from typing import List, Optional, Union

from pydantic import BaseModel, Field


# ───────────────────── REQUEST ─────────────────────

class ChatMessage(BaseModel):
    role: str
    content: Union[str, None] = ""
    name: Optional[str] = None


class ChatCompletionRequest(BaseModel):
    model: str = "gpt-4o-mini"
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 1.0
    max_tokens: Optional[int] = None
    stream: Optional[bool] = False
    presence_penalty: Optional[float] = 0.0
    frequency_penalty: Optional[float] = 0.0
    user: Optional[str] = None


# ───────────────────── RESPONSE (non-stream) ──────

class Choice(BaseModel):
    index: int = 0
    message: ChatMessage
    finish_reason: str = "stop"


class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatCompletionResponse(BaseModel):
    id: str = Field(
        default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex[:16]}"
    )
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[Choice]
    usage: Usage = Field(default_factory=Usage)


# ───────────────────── RESPONSE (stream / SSE) ────

class DeltaMessage(BaseModel):
    role: Optional[str] = None
    content: Optional[str] = None


class StreamChoice(BaseModel):
    index: int = 0
    delta: DeltaMessage
    finish_reason: Optional[str] = None


class ChatCompletionChunk(BaseModel):
    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: List[StreamChoice]