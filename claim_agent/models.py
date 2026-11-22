"""Neutral message types so the agent loop does not depend on any one LLM provider."""
from dataclasses import dataclass, field


@dataclass
class ToolCall:
    id: str
    name: str
    args: dict


@dataclass
class LLMResponse:
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
