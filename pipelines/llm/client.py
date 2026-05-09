"""Tiny model factory + provider router.

Pattern routing by name prefix:
    ``claude-cli:<model>`` → ClaudeCLIChatModel
    ``codex:<model>``       → CodexCLIChatModel
    ``claude*``             → ChatAnthropic (cloud, needs ANTHROPIC_API_KEY)
    anything else           → ChatOpenAI    (cloud, needs OPENAI_API_KEY)

Default with no env vars is ``claude-cli:sonnet`` because the user has
the Claude Code CLI authenticated locally.
"""

from __future__ import annotations

import os
from typing import Any

DEFAULT_MODEL = "claude-cli:sonnet"


def detect_provider(model: str) -> str:
    name = (model or "").strip().lower()
    if name.startswith(("claude-cli:", "claude-cli/")):
        return "claude-cli"
    if name.startswith(("codex:", "codex/")):
        return "codex-cli"
    if name.startswith("claude"):
        return "anthropic"
    return "openai"


def build_chat_model(
    model: str | None = None,
    *,
    timeout_seconds: int | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> Any:
    """Construct the right ChatModel for ``model``. Imports are deferred
    so the project doesn't need ``langchain-anthropic`` installed when
    only the CLI path is used."""
    model = model or os.environ.get("UFO_LLM_MODEL", DEFAULT_MODEL)
    provider = detect_provider(model)

    if provider == "claude-cli":
        from .cli_chat import ClaudeCLIChatModel

        clean = model.split(":", 1)[1] if ":" in model else "sonnet"
        kwargs: dict[str, Any] = {"model": clean}
        if timeout_seconds is not None:
            kwargs["timeout_seconds"] = timeout_seconds
        return ClaudeCLIChatModel(**kwargs)

    if provider == "codex-cli":
        from .cli_chat import CodexCLIChatModel

        clean = model.split(":", 1)[1] if ":" in model else "gpt-5"
        kwargs = {"model": clean}
        if timeout_seconds is not None:
            kwargs["timeout_seconds"] = timeout_seconds
        return CodexCLIChatModel(**kwargs)

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        kwargs = {"model_name": model}
        if temperature is not None:
            kwargs["temperature"] = temperature
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        return ChatAnthropic(**kwargs)

    from langchain_openai import ChatOpenAI

    kwargs = {"model": model}
    if temperature is not None:
        kwargs["temperature"] = temperature
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    return ChatOpenAI(**kwargs)
