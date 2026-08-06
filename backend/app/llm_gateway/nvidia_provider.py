"""NVIDIA AI Endpoints provider via LangChain's `ChatNVIDIA` — no self-hosted
model. This is the concrete implementation of `LLMProvider` for
`LLM_PROVIDER=nvidia_endpoints`. It's distinct from the existing NVIDIA-via-
OpenAI-compatible-endpoint path in `app/providers.py` (which still works and
needs no extra dependency); this one uses the idiomatic LangChain integration
so it composes with LCEL chains, LangSmith tracing, and the rest of the
LangChain retrieval layer without a translation shim.
"""
import time
from typing import Iterator

from .. import config
from .base import LLMProvider, LLMProviderError, LLMResult


class NvidiaEndpointsProvider(LLMProvider):
    name = "nvidia_endpoints"

    def __init__(self):
        self.model = config.LLM_MODEL or "nvidia/nemotron-3-ultra-550b-a55b"
        self._client = None  # lazy: don't import/construct without a key present

    def is_configured(self) -> bool:
        return bool(config.NVIDIA_NIM_API_KEY)

    def _get_client(self):
        if self._client is not None:
            return self._client
        if not config.NVIDIA_NIM_API_KEY:
            raise LLMProviderError("NVIDIA_NIM_API_KEY is not set")
        try:
            from langchain_nvidia_ai_endpoints import ChatNVIDIA
        except ImportError as e:
            raise LLMProviderError(
                f"langchain-nvidia-ai-endpoints is not installed: {e}"
            )
        kwargs = dict(
            model=self.model,
            api_key=config.NVIDIA_NIM_API_KEY,
            temperature=config.LLM_TEMPERATURE,
            max_tokens=config.LLM_MAX_TOKENS,
            timeout=config.LLM_TIMEOUT,
        )
        self._client = ChatNVIDIA(**kwargs)
        return self._client

    def _call_with_retry(self, fn, *args, **kwargs):
        last_err = None
        for attempt in range(config.LLM_MAX_RETRIES + 1):
            try:
                return fn(*args, **kwargs)
            except Exception as e:
                last_err = e
                if attempt < config.LLM_MAX_RETRIES:
                    time.sleep(min(2 ** attempt, 8))  # exponential backoff, capped
        raise LLMProviderError(f"{self.name} failed after "
                                f"{config.LLM_MAX_RETRIES + 1} attempts: {last_err}")

    def invoke(self, system: str, user_message: str) -> LLMResult:
        client = self._get_client()
        start = time.time()

        def _do():
            from langchain_core.messages import HumanMessage, SystemMessage
            return client.invoke([SystemMessage(content=system), HumanMessage(content=user_message)])

        response = self._call_with_retry(_do)
        return LLMResult(
            text=response.content, provider=self.name, model=self.model,
            latency_seconds=round(time.time() - start, 3),
        )

    def stream(self, system: str, user_message: str) -> Iterator[str]:
        client = self._get_client()
        from langchain_core.messages import HumanMessage, SystemMessage
        try:
            for chunk in client.stream([SystemMessage(content=system), HumanMessage(content=user_message)]):
                if chunk.content:
                    yield chunk.content
        except Exception as e:
            raise LLMProviderError(f"{self.name} streaming failed: {e}")
