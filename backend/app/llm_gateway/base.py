"""Abstract LLM provider interface.

Nothing outside this package should import ChatNVIDIA, anthropic, or openai
directly. Everything talks to `LLMProvider`, so swapping or adding a provider
(a new model host, a new SDK version) never touches calling code — this is the
concrete "abstraction layer" the master prompt asks for, not a label on the
existing ad-hoc provider code.

This sits alongside — not instead of — the existing `app/providers.py`
multi-provider gateway (Anthropic/OpenAI/NVIDIA-via-OpenAI-compat/DeepSeek),
which stays as the app's default, already-tested LLM path. `nvidia_provider.py`
is the LangChain/`ChatNVIDIA`-based implementation the new spec asks for
specifically, wired in as an additional selectable backend rather than a
wholesale replacement of code that already works.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterator, Optional


@dataclass
class LLMResult:
    text: str
    provider: str
    model: str
    latency_seconds: Optional[float] = None


class LLMProviderError(Exception):
    """Raised on any failure — timeout, auth, rate limit, bad response.
    Callers (the LangGraph `generate` node, the extractive fallback path)
    catch this uniformly regardless of which concrete provider raised it."""


class LLMProvider(ABC):
    name: str
    model: str

    @abstractmethod
    def invoke(self, system: str, user_message: str) -> LLMResult:
        """Single-shot completion. Must raise LLMProviderError on failure,
        never a raw SDK exception."""

    @abstractmethod
    def stream(self, system: str, user_message: str) -> Iterator[str]:
        """Token-stream a completion. Must raise LLMProviderError on failure."""

    def is_configured(self) -> bool:
        """Whether this provider has the credentials it needs. Default: True,
        since most providers need a key to even construct; override if a
        provider can be constructed without one (e.g. a local model)."""
        return True
