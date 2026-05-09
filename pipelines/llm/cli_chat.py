"""LangChain ChatModel wrappers for the local `claude` and `codex` CLIs.

Why subprocess and not the Anthropic/OpenAI SDKs? The user's deployment
is "Claude Code CLI and Codex CLI talking to LangGraph" — the CLIs are
already authenticated (OAuth, keychain) and runnable. Spawning them is
the lowest-friction path; no API keys to manage. Trade-off: ~1s of CLI
startup latency per call, no streaming yet.

Two wrappers ship here:

- ClaudeCLIChatModel — ``claude -p ... --output-format json``
- CodexCLIChatModel  — ``codex exec ...``

Both override ``with_structured_output`` because the CLIs don't speak
function-calling. The override appends the Pydantic JSON schema to the
prompt with a "respond as JSON" preamble, then parses the first JSON
object out of the response and validates against the schema.

(Adapted from
``decksmith-remote-work/python/voice_clone/lcgraph/llm/_cli_chat.py``.)
"""

from __future__ import annotations

import json
import re
import subprocess
from typing import Any

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable, RunnableLambda
from pydantic import BaseModel


def _build_structured_user_prompt(user_text: str, schema: type[BaseModel]) -> str:
    schema_json = json.dumps(schema.model_json_schema(), indent=2)
    return (
        f"{user_text}\n\n"
        "Respond with a single JSON object that matches this JSON schema "
        "EXACTLY — no prose before or after, no markdown fences:\n"
        f"{schema_json}"
    )


def _extract_first_json_object(text: str) -> dict | None:
    """Walk every `{` and try `JSONDecoder().raw_decode` on the suffix.

    More robust than `re.search(r"\\{.*\\}", ..., DOTALL)` which is greedy
    and silently mashes two valid JSON objects into one when the model
    emits both.
    """
    decoder = json.JSONDecoder()
    start = text.find("{")
    while start != -1:
        try:
            candidate, _end = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            start = text.find("{", start + 1)
            continue
        if isinstance(candidate, dict):
            return candidate
        start = text.find("{", start + 1)
    return None


def _parse_structured_response(raw: str, schema: type[BaseModel]) -> BaseModel:
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        payload = _extract_first_json_object(text)
        if payload is None:
            raise ValueError(
                f"CLI response did not contain a JSON object (got {text[:200]!r})"
            )
    return schema.model_validate(payload)


def _make_structured_runnable(
    cli_model: BaseChatModel,
    schema: type[BaseModel],
    *,
    include_raw: bool,
) -> Runnable:
    def _invoke_structured(prompt_value: Any) -> Any:
        messages = (
            prompt_value.to_messages()
            if hasattr(prompt_value, "to_messages")
            else list(prompt_value)
        )
        last_human_idx = -1
        for i, msg in enumerate(messages):
            if isinstance(msg, HumanMessage):
                last_human_idx = i
        augmented: list[BaseMessage] = []
        for i, msg in enumerate(messages):
            if i == last_human_idx:
                augmented.append(
                    HumanMessage(
                        content=_build_structured_user_prompt(
                            str(msg.content), schema
                        )
                    )
                )
            else:
                augmented.append(msg)
        result = cli_model._generate(augmented)
        raw = result.generations[0].message.content
        parsed = _parse_structured_response(str(raw), schema)
        if include_raw:
            return {"raw": result.generations[0].message, "parsed": parsed}
        return parsed

    return RunnableLambda(_invoke_structured)


def _flatten_messages(messages: list[BaseMessage]) -> tuple[str, str]:
    """Reduce a LangChain message list to (system_prompt, user_prompt).

    The CLIs accept one system + one user prompt. Earlier human turns
    get folded into the system prompt as "Previous conversation:" so
    the CLI sees the chat context.
    """
    system_chunks: list[str] = []
    earlier_human: list[str] = []
    last_human = ""
    for msg in messages:
        if isinstance(msg, SystemMessage):
            system_chunks.append(str(msg.content))
        elif isinstance(msg, HumanMessage):
            if last_human:
                earlier_human.append(last_human)
            last_human = str(msg.content)
        else:
            earlier_human.append(f"[assistant said] {msg.content}")
    if earlier_human:
        system_chunks.append("Previous conversation:\n" + "\n".join(earlier_human))
    return "\n\n".join(system_chunks), last_human


class ClaudeCLIChatModel(BaseChatModel):
    """LangChain ChatModel that invokes ``claude -p ...``.

    Defaults to the ``sonnet`` alias (latest Sonnet the CLI is logged
    into). Set ``model=`` to pin a specific id (e.g.
    ``claude-sonnet-4-6``, ``claude-opus-4-7``).
    """

    model: str = "sonnet"
    timeout_seconds: int = 180
    extra_cli_args: tuple[str, ...] = ()

    @property
    def _llm_type(self) -> str:
        return "claude-cli"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        system, user = _flatten_messages(messages)

        cmd: list[str] = [
            "claude", "-p",
            "--model", self.model,
            "--output-format", "json",
        ]
        if system:
            cmd += ["--append-system-prompt", system]
        cmd.extend(self.extra_cli_args)
        cmd.append(user)

        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
                timeout=self.timeout_seconds,
            )
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(
                f"claude CLI failed (exit {exc.returncode}): "
                f"stderr={(exc.stderr or '').strip()!r}, "
                f"stdout={(exc.stdout or '')[:200].strip()!r}"
            ) from exc
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                f"claude CLI timed out after {self.timeout_seconds}s"
            ) from exc

        try:
            payload = json.loads(proc.stdout.strip())
            content = (
                payload.get("result")
                or payload.get("response")
                or payload.get("text")
                or json.dumps(payload)
            ) if isinstance(payload, dict) else str(payload)
        except json.JSONDecodeError:
            content = proc.stdout.strip()

        return ChatResult(
            generations=[ChatGeneration(message=AIMessage(content=str(content)))]
        )

    def with_structured_output(
        self,
        schema: type[BaseModel],
        *,
        include_raw: bool = False,
        **_kwargs: Any,
    ) -> Runnable:
        return _make_structured_runnable(self, schema, include_raw=include_raw)


class CodexCLIChatModel(BaseChatModel):
    """LangChain ChatModel that invokes ``codex exec ...``.

    Codex's exec mode prints raw text to stdout — no JSON-output flag.
    Structured output uses the same JSON-schema-in-prompt approach as
    the Claude wrapper.
    """

    model: str = "gpt-5"
    timeout_seconds: int = 180
    extra_cli_args: tuple[str, ...] = ()

    @property
    def _llm_type(self) -> str:
        return "codex-cli"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        system, user = _flatten_messages(messages)
        full_prompt = f"<system>\n{system}\n</system>\n\n{user}" if system else user

        cmd: list[str] = [
            "codex", "exec",
            "--model", self.model,
            "--skip-git-repo-check",
        ]
        cmd.extend(self.extra_cli_args)
        cmd.append(full_prompt)

        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
                timeout=self.timeout_seconds,
            )
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(
                f"codex CLI failed (exit {exc.returncode}): "
                f"stderr={(exc.stderr or '').strip()!r}, "
                f"stdout={(exc.stdout or '')[:200].strip()!r}"
            ) from exc
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                f"codex CLI timed out after {self.timeout_seconds}s"
            ) from exc

        return ChatResult(
            generations=[
                ChatGeneration(message=AIMessage(content=proc.stdout.strip()))
            ]
        )

    def with_structured_output(
        self,
        schema: type[BaseModel],
        *,
        include_raw: bool = False,
        **_kwargs: Any,
    ) -> Runnable:
        return _make_structured_runnable(self, schema, include_raw=include_raw)


__all__ = ["ClaudeCLIChatModel", "CodexCLIChatModel"]
