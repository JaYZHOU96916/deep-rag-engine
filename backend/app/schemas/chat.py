from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ModelRoute(BaseModel):
    provider: Literal["local", "openai", "deepseek", "claude"] | None = None
    model: str | None = Field(default=None, min_length=1, max_length=200)


class ChatStreamRequest(BaseModel):
    question: str = Field(min_length=1, max_length=8_000)
    route: ModelRoute | None = None
    use_semantic_cache: bool = True


class TypedSSEEvent(BaseModel):
    event: Literal["thought", "citation", "delta", "error"]
    data: dict
