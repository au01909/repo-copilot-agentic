"""Provider-agnostic LLM client.

LLM_PROVIDER env var picks the backend: anthropic | openai | nvidia | deepseek | none.
NVIDIA NIM and DeepSeek both expose OpenAI-compatible chat completion endpoints, so
they're implemented as OpenAI client instances pointed at a different base_url —
that's the real, documented way both providers recommend integrating them.
"""
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

from . import config


@dataclass
class LLMDebugInfo:
    provider: str
    model: str
    api_key_present: bool
    api_key_prefix: Optional[str]
    latency_seconds: Optional[float] = None
    status: str = "not_called"       # not_called | success | error
    error: Optional[str] = None


def _key_prefix(key: Optional[str]) -> Optional[str]:
    if not key:
        return None
    return key[:7] + "…" if len(key) > 7 else key


class ProviderUnavailable(Exception):
    pass


def get_active_provider_debug() -> LLMDebugInfo:
    """Report which provider is configured and whether credentials are present,
    without making a network call. Used for the /api/debug/llm endpoint."""
    provider = config.LLM_PROVIDER
    mapping = {
        "anthropic": (config.ANTHROPIC_API_KEY, config.ANTHROPIC_MODEL),
        "openai": (config.OPENAI_API_KEY, config.OPENAI_MODEL),
        "nvidia": (config.NVIDIA_NIM_API_KEY, config.NVIDIA_MODEL),
        "deepseek": (config.DEEPSEEK_API_KEY, config.DEEPSEEK_MODEL),
        "none": (None, "n/a"),
    }
    key, model = mapping.get(provider, (None, "unknown"))
    return LLMDebugInfo(
        provider=provider, model=model,
        api_key_present=bool(key), api_key_prefix=_key_prefix(key),
    )


def _anthropic_complete(system: str, user_msg: str) -> str:
    import anthropic
    if not config.ANTHROPIC_API_KEY:
        raise ProviderUnavailable("ANTHROPIC_API_KEY is not set")
    client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
    response = client.messages.create(
        model=config.ANTHROPIC_MODEL, max_tokens=1000, system=system,
        messages=[{"role": "user", "content": user_msg}],
    )
    return "\n".join(b.text for b in response.content if getattr(b, "type", None) == "text").strip()


def _openai_compatible_client(base_url: Optional[str], api_key: Optional[str]):
    import openai
    if not api_key:
        raise ProviderUnavailable("API key not set for this provider")
    kwargs = {"api_key": api_key}
    if base_url:
        kwargs["base_url"] = base_url
    return openai.OpenAI(**kwargs)


def _openai_family_complete(base_url: Optional[str], api_key: Optional[str], model: str,
                             system: str, user_msg: str) -> str:
    client = _openai_compatible_client(base_url, api_key)
    resp = client.chat.completions.create(
        model=model, max_tokens=1000,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user_msg}],
    )
    return (resp.choices[0].message.content or "").strip()


_PROVIDER_TABLE = {
    "openai": lambda: (config.OPENAI_BASE_URL, config.OPENAI_API_KEY, config.OPENAI_MODEL),
    "nvidia": lambda: (config.NVIDIA_BASE_URL, config.NVIDIA_NIM_API_KEY, config.NVIDIA_MODEL),
    "deepseek": lambda: (config.DEEPSEEK_BASE_URL, config.DEEPSEEK_API_KEY, config.DEEPSEEK_MODEL),
}


def complete(system: str, user_msg: str) -> Tuple[str, LLMDebugInfo]:
    """Call the configured provider. Raises ProviderUnavailable if not configured;
    callers should catch this and fall back to extractive mode."""
    provider = config.LLM_PROVIDER
    debug = get_active_provider_debug()
    start = time.time()
    try:
        if provider == "anthropic":
            text = _anthropic_complete(system, user_msg)
        elif provider in _PROVIDER_TABLE:
            base_url, api_key, model = _PROVIDER_TABLE[provider]()
            text = _openai_family_complete(base_url, api_key, model, system, user_msg)
        elif provider == "none":
            raise ProviderUnavailable("LLM_PROVIDER=none")
        else:
            raise ProviderUnavailable(f"Unknown LLM_PROVIDER '{provider}'")
        debug.latency_seconds = round(time.time() - start, 3)
        debug.status = "success"
        return text, debug
    except ProviderUnavailable as e:
        debug.status = "error"
        debug.error = str(e)
        raise
    except Exception as e:
        debug.status = "error"
        debug.error = str(e)
        raise ProviderUnavailable(str(e)) from e


def is_configured() -> bool:
    return get_active_provider_debug().api_key_present
