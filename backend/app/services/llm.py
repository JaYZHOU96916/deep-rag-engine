from __future__ import annotations

import re
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from typing import Literal, Protocol

from anthropic import AsyncAnthropic
from openai import AsyncOpenAI

from app.core.config import Settings, get_settings

ChatMessage = dict[str, str]
ProviderName = Literal["local", "openai", "deepseek", "claude"]


@dataclass(frozen=True, slots=True)
class LLMRoute:
    provider: ProviderName
    model: str


class AsyncLLMClient(Protocol):
    async def complete(self, messages: Sequence[ChatMessage]) -> str: ...

    async def stream(self, messages: Sequence[ChatMessage]) -> AsyncIterator[str]: ...


class OpenAICompatibleLLMClient:
    """OpenAI Chat Completions adapter, also used by DeepSeek-compatible endpoints."""

    def __init__(self, *, api_key: str, base_url: str, model: str, temperature: float) -> None:
        self.client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.temperature = temperature

    async def complete(self, messages: Sequence[ChatMessage]) -> str:
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=list(messages),  # type: ignore[arg-type]
            temperature=self.temperature,
        )
        return response.choices[0].message.content or ""

    async def stream(self, messages: Sequence[ChatMessage]) -> AsyncIterator[str]:
        stream = await self.client.chat.completions.create(
            model=self.model,
            messages=list(messages),  # type: ignore[arg-type]
            temperature=self.temperature,
            stream=True,
        )
        async for chunk in stream:
            content = chunk.choices[0].delta.content if chunk.choices else None
            if content:
                yield content


class ClaudeLLMClient:
    """Native Claude adapter behind the same async client protocol as OpenAI-compatible routes."""

    def __init__(self, *, api_key: str, model: str, temperature: float) -> None:
        self.client = AsyncAnthropic(api_key=api_key)
        self.model = model
        self.temperature = temperature

    async def complete(self, messages: Sequence[ChatMessage]) -> str:
        system, content_messages = _split_anthropic_messages(messages)
        response = await self.client.messages.create(
            model=self.model,
            max_tokens=2_048,
            temperature=self.temperature,
            system=system,
            messages=content_messages,
        )
        return "".join(block.text for block in response.content if block.type == "text")

    async def stream(self, messages: Sequence[ChatMessage]) -> AsyncIterator[str]:
        system, content_messages = _split_anthropic_messages(messages)
        async with self.client.messages.stream(
            model=self.model,
            max_tokens=2_048,
            temperature=self.temperature,
            system=system,
            messages=content_messages,
        ) as stream:
            async for text in stream.text_stream:
                yield text


class LocalGroundedLLMClient:
    """Credential-free development fallback that never introduces evidence beyond supplied references."""

    async def complete(self, messages: Sequence[ChatMessage]) -> str:
        system = messages[0].get("content", "") if messages else ""
        user_content = messages[-1].get("content", "") if messages else ""
        if "hypothetical" in system.lower():
            question = user_content.split("Question:", 1)[-1].split("\n", 1)[0].strip()
            return f"Hypothetical academic passage about {question}."
        if "Self-RAG" in system:
            return user_content.split("Draft:\n", 1)[-1].split("\n\nEvidence:", 1)[0].strip()
        match = re.search(r"(\[Ref: [^\]]+\])\s*\n([^\n]+)", user_content)
        if not match:
            return "检索到的参考资料中未包含关于 当前问题 的确切信息。"
        citation, evidence = match.groups()
        return f"根据检索到的资料，{evidence.strip()} {citation}"

    async def stream(self, messages: Sequence[ChatMessage]) -> AsyncIterator[str]:
        answer = await self.complete(messages)
        for index in range(0, len(answer), 32):
            yield answer[index : index + 32]


def get_llm_client(
    route: LLMRoute | None = None,
    settings: Settings | None = None,
) -> AsyncLLMClient:
    current_settings = settings or get_settings()
    resolved = route or LLMRoute(provider=current_settings.llm_provider, model=current_settings.llm_model)
    if resolved.provider == "local":
        return LocalGroundedLLMClient()
    if resolved.provider == "openai":
        return OpenAICompatibleLLMClient(
            api_key=_secret(current_settings.openai_api_key, "OPENAI_API_KEY"),
            base_url=current_settings.openai_base_url,
            model=resolved.model,
            temperature=current_settings.llm_temperature,
        )
    if resolved.provider == "deepseek":
        return OpenAICompatibleLLMClient(
            api_key=_secret(current_settings.deepseek_api_key, "DEEPSEEK_API_KEY"),
            base_url=current_settings.deepseek_base_url,
            model=resolved.model,
            temperature=current_settings.llm_temperature,
        )
    return ClaudeLLMClient(
        api_key=_secret(current_settings.claude_api_key, "CLAUDE_API_KEY"),
        model=resolved.model,
        temperature=current_settings.llm_temperature,
    )


def resolve_route(
    provider: ProviderName | None,
    model: str | None,
    settings: Settings | None = None,
) -> LLMRoute:
    current_settings = settings or get_settings()
    return LLMRoute(provider=provider or current_settings.llm_provider, model=model or current_settings.llm_model)


def _secret(value: object, name: str) -> str:
    if value is None:
        raise RuntimeError(f"{name} is required for the selected LLM provider.")
    return value.get_secret_value()  # type: ignore[union-attr]


def _split_anthropic_messages(messages: Sequence[ChatMessage]) -> tuple[str, list[dict[str, str]]]:
    system = "\n".join(message["content"] for message in messages if message["role"] == "system")
    converted = [
        {"role": message["role"], "content": message["content"]}
        for message in messages
        if message["role"] != "system"
    ]
    return system, converted
