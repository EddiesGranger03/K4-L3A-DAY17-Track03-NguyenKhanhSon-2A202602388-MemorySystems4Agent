from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ProviderConfig:
    """Provider configuration shared by the agents."""

    provider: str
    model_name: str
    temperature: float
    api_key: str | None = None
    base_url: str | None = None


_ALIASES = {
    "anthorpic": "anthropic",
    "anthropic": "anthropic",
    "claude": "anthropic",
    "openai": "openai",
    "gpt": "openai",
    "custom": "custom",
    "openai-compatible": "custom",
    "compatible": "custom",
    "gemini": "gemini",
    "google": "gemini",
    "google-genai": "gemini",
    "bard": "gemini",
    "ollama": "ollama",
    "openrouter": "openrouter",
}


def normalize_provider(value: str) -> str:
    """Map aliases like `anthorpic` -> `anthropic`, case-insensitive."""
    key = (value or "").strip().lower()
    return _ALIASES.get(key, key)


def build_chat_model(config: ProviderConfig):
    """Instantiate the real chat model for the selected provider.

    Only called in live mode. Offline benchmark/tests never reach here.
    Raises informative errors when optional deps are missing.
    """
    provider = normalize_provider(config.provider)

    if provider == "openai":
        try:
            from langchain_openai import ChatOpenAI
        except ImportError as exc:
            raise ImportError("langchain-openai is required for provider 'openai'") from exc
        kwargs: dict = {"model": config.model_name, "temperature": config.temperature}
        if config.api_key:
            kwargs["api_key"] = config.api_key
        if config.base_url:
            kwargs["base_url"] = config.base_url
        return ChatOpenAI(**kwargs)

    if provider == "custom":
        try:
            from langchain_openai import ChatOpenAI
        except ImportError as exc:
            raise ImportError("langchain-openai is required for provider 'custom'") from exc
        if not config.base_url:
            raise ValueError("provider 'custom' requires a base_url (CUSTOM_BASE_URL)")
        kwargs = {
            "model": config.model_name,
            "temperature": config.temperature,
            "base_url": config.base_url,
        }
        if config.api_key:
            kwargs["api_key"] = config.api_key
        return ChatOpenAI(**kwargs)

    if provider == "gemini":
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
        except ImportError as exc:
            raise ImportError("langchain-google-genai is required for provider 'gemini'") from exc
        kwargs = {"model": config.model_name, "temperature": config.temperature}
        if config.api_key:
            kwargs["google_api_key"] = config.api_key
        return ChatGoogleGenerativeAI(**kwargs)

    if provider == "anthropic":
        try:
            from langchain_anthropic import ChatAnthropic
        except ImportError as exc:
            raise ImportError("langchain-anthropic is required for provider 'anthropic'") from exc
        kwargs = {"model": config.model_name, "temperature": config.temperature}
        if config.api_key:
            kwargs["api_key"] = config.api_key
        if config.base_url:
            kwargs["base_url"] = config.base_url
        return ChatAnthropic(**kwargs)

    if provider == "ollama":
        try:
            from langchain_ollama import ChatOllama
        except ImportError as exc:
            raise ImportError("langchain-ollama is required for provider 'ollama'") from exc
        kwargs = {"model": config.model_name, "temperature": config.temperature}
        if config.base_url:
            kwargs["base_url"] = config.base_url
        return ChatOllama(**kwargs)

    if provider == "openrouter":
        try:
            from langchain_openai import ChatOpenAI
        except ImportError as exc:
            raise ImportError("langchain-openai is required for provider 'openrouter'") from exc
        try:
            from langchain_openrouter import ChatOpenRouter  # type: ignore
        except ImportError:
            ChatOpenRouter = None  # type: ignore
        base_url = config.base_url or "https://openrouter.ai/api/v1"
        if ChatOpenRouter is not None:
            kwargs = {"model": config.model_name, "temperature": config.temperature}
            if config.api_key:
                kwargs["api_key"] = config.api_key
            return ChatOpenRouter(**kwargs)
        kwargs = {
            "model": config.model_name,
            "temperature": config.temperature,
            "base_url": base_url,
        }
        if config.api_key:
            kwargs["api_key"] = config.api_key
        return ChatOpenAI(**kwargs)

    raise ValueError(f"Unsupported provider: {config.provider!r}")
