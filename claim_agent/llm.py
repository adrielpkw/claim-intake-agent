"""LLM clients. The agent only knows the `complete()` interface.

- AnthropicClient: real model with tool use (needs ANTHROPIC_API_KEY).
- ScriptedClient:  returns pre-written responses. Lets us test the agent loop and
                   guardrails offline, including how the agent reacts to a BAD model.
"""
import json
import os
import urllib.error
import urllib.request
import uuid
from typing import Protocol

from .models import LLMResponse, ToolCall


class LLMClient(Protocol):
    def complete(self, system: str, messages: list[dict], tools: list[dict]) -> LLMResponse: ...


class ScriptedClient:
    name = "scripted"

    def __init__(self, responses: list[LLMResponse | Exception]):
        self._responses = list(responses)
        self.seen_messages: list[list[dict]] = []  # what the "model" was shown, for PII tests

    def complete(self, system, messages, tools):
        self.seen_messages.append([dict(m) for m in messages])
        if not self._responses:
            return LLMResponse(text="(script ended)")
        item = self._responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


class AnthropicClient:
    def __init__(self, model: str | None = None, max_tokens: int = 1024):
        import anthropic  # imported here so offline tests do not need the package

        self._client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY
        self._model = model or os.getenv("CLAIM_AGENT_MODEL", "claude-sonnet-5-5")
        self._max_tokens = max_tokens
        self.name = f"anthropic:{self._model}"

    def complete(self, system, messages, tools):
        resp = self._client.messages.create(
            model=self._model,
            max_tokens=self._max_tokens,
            system=system,
            tools=tools,
            messages=self._to_api(messages),
        )
        text = "".join(b.text for b in resp.content if b.type == "text")
        calls = [ToolCall(b.id, b.name, dict(b.input)) for b in resp.content if b.type == "tool_use"]
        return LLMResponse(text=text, tool_calls=calls)

    @staticmethod
    def _to_api(messages: list[dict]) -> list[dict]:
        out: list[dict] = []
        for m in messages:
            if m["role"] == "user":
                out.append({"role": "user", "content": m["content"]})
            elif m["role"] == "assistant":
                blocks = []
                if m.get("text"):
                    blocks.append({"type": "text", "text": m["text"]})
                for c in m.get("tool_calls", []):
                    blocks.append({"type": "tool_use", "id": c.id, "name": c.name, "input": c.args})
                out.append({"role": "assistant", "content": blocks})
            elif m["role"] == "tool":
                block = {"type": "tool_result", "tool_use_id": m["tool_call_id"], "content": m["content"]}
                if out and out[-1]["role"] == "user" and isinstance(out[-1]["content"], list):
                    out[-1]["content"].append(block)  # merge parallel tool results
                else:
                    out.append({"role": "user", "content": [block]})
        return out


class OllamaClient:
    """Local model through Ollama (https://ollama.com). No API key, no per-call cost.

    Needs a model that supports tool calling, e.g. qwen2.5:7b or llama3.1:8b.
    Small local models follow tool rules less reliably, so report results WITH the model name.
    """

    def __init__(self, model: str | None = None, host: str | None = None, timeout: int = 300):
        self._model = model or os.getenv("OLLAMA_MODEL", "qwen2.5:7b")
        self._host = (host or os.getenv("OLLAMA_HOST", "http://localhost:11434")).rstrip("/")
        self._timeout = timeout
        self.name = f"ollama:{self._model}"

    def complete(self, system, messages, tools):
        payload = {
            "model": self._model,
            "stream": False,
            "options": {"temperature": 0},  # repeatable runs
            "messages": [{"role": "system", "content": system}] + self._to_api(messages),
            "tools": [
                {"type": "function",
                 "function": {"name": t["name"], "description": t["description"], "parameters": t["input_schema"]}}
                for t in tools
            ],
        }
        req = urllib.request.Request(
            f"{self._host}/api/chat", data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=self._timeout) as r:
                data = json.loads(r.read())
        except urllib.error.URLError as e:
            raise RuntimeError(f"Cannot reach Ollama at {self._host}. Is it running? (ollama serve) {e}") from e

        msg = data.get("message", {})
        calls = []
        for tc in msg.get("tool_calls") or []:
            fn = tc.get("function", {})
            args = fn.get("arguments", {})
            if isinstance(args, str):  # some versions return a JSON string
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {}
            calls.append(ToolCall(f"call_{uuid.uuid4().hex[:8]}", fn.get("name", ""), args))
        return LLMResponse(text=msg.get("content") or "", tool_calls=calls)

    @staticmethod
    def _to_api(messages: list[dict]) -> list[dict]:
        out: list[dict] = []
        names: dict[str, str] = {}  # tool_call_id -> tool name (Ollama wants the name on results)
        for m in messages:
            if m["role"] == "user":
                out.append({"role": "user", "content": m["content"]})
            elif m["role"] == "assistant":
                for c in m.get("tool_calls", []):
                    names[c.id] = c.name
                entry = {"role": "assistant", "content": m.get("text", "")}
                if m.get("tool_calls"):
                    entry["tool_calls"] = [{"function": {"name": c.name, "arguments": c.args}} for c in m["tool_calls"]]
                out.append(entry)
            elif m["role"] == "tool":
                out.append({"role": "tool", "tool_name": names.get(m["tool_call_id"], ""), "content": m["content"]})
        return out


def make_client() -> LLMClient:
    """LLM_PROVIDER=ollama for a local model; default is the Anthropic API."""
    if os.getenv("LLM_PROVIDER", "anthropic").lower() == "ollama":
        return OllamaClient()
    return AnthropicClient()
