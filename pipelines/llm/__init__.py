"""LLM backend for the UFO pipeline.

Wraps the local `claude` and `codex` CLIs as LangChain ChatModels so the
pipeline runs without any cloud API key — the CLI's existing OAuth
session does the talking. Pattern lifted (and trimmed) from
~/Code/decksmith-remote-work/python/voice_clone/lcgraph/llm/.
"""

from .cli_chat import ClaudeCLIChatModel, CodexCLIChatModel
from .client import build_chat_model, detect_provider

__all__ = [
    "ClaudeCLIChatModel",
    "CodexCLIChatModel",
    "build_chat_model",
    "detect_provider",
]
