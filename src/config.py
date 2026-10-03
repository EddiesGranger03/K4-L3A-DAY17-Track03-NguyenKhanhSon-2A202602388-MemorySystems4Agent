from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

try:
    from model_provider import ProviderConfig, normalize_provider
except ImportError:  # pragma: no cover - supports `python -m src.*`
    from src.model_provider import ProviderConfig, normalize_provider


@dataclass
class LabConfig:
    base_dir: Path
    data_dir: Path
    state_dir: Path
    compact_threshold_tokens: int
    compact_keep_messages: int
    model: ProviderConfig
    judge_model: ProviderConfig


def _getenv(name: str, default: str = "") -> str:
    value = os.getenv(name, default)
    return value.strip() if isinstance(value, str) else default


def load_config(base_dir: Path | None = None) -> LabConfig:
    # 1. Resolve repo root.
    root = (base_dir or Path(__file__).resolve().parent.parent).resolve()

    # 2. Optionally load `.env` (never crash if dotenv is missing).
    try:
        from dotenv import load_dotenv

        load_dotenv(root / ".env")
    except Exception:
        pass

    # 3. Create `state/` if missing.
    state_dir = root / "state"
    state_dir.mkdir(parents=True, exist_ok=True)

    data_dir = root / "data"

    # Compact-memory defaults: tuned so standard 10-turn convs rarely
    # compact, while the 16-turn stress conv (very long turns) compacts
    # several times.
    threshold = int(_getenv("COMPACT_THRESHOLD_TOKENS", "1200") or "1200")
    keep = int(_getenv("COMPACT_KEEP_MESSAGES", "6") or "6")

    provider = normalize_provider(_getenv("LLM_PROVIDER", "openai") or "openai")
    model_name = _getenv("LLM_MODEL", "gpt-4o-mini") or "gpt-4o-mini"
    try:
        temperature = float(_getenv("LLM_TEMPERATURE", "0") or "0")
    except ValueError:
        temperature = 0.0

    api_key = (
        _getenv("OPENAI_API_KEY")
        or _getenv("GEMINI_API_KEY")
        or _getenv("GOOGLE_API_KEY")
        or _getenv("ANTHROPIC_API_KEY")
        or _getenv("OPENROUTER_API_KEY")
        or _getenv("CUSTOM_API_KEY")
        or _getenv("LLM_API_KEY")
        or None
    )
    base_url = (
        _getenv("CUSTOM_BASE_URL")
        or _getenv("OLLAMA_BASE_URL")
        or _getenv("OPENROUTER_BASE_URL")
        or _getenv("LLM_BASE_URL")
        or None
    )
    if provider == "ollama" and not base_url:
        base_url = "http://localhost:11434"

    model = ProviderConfig(
        provider=provider,
        model_name=model_name,
        temperature=temperature,
        api_key=api_key,
        base_url=base_url,
    )
    judge_provider = normalize_provider(_getenv("JUDGE_PROVIDER", provider) or provider)
    judge_model_name = _getenv("JUDGE_MODEL", model_name) or model_name
    judge_model = ProviderConfig(
        provider=judge_provider,
        model_name=judge_model_name,
        temperature=0.0,
        api_key=api_key,
        base_url=base_url,
    )

    return LabConfig(
        base_dir=root,
        data_dir=data_dir,
        state_dir=state_dir,
        compact_threshold_tokens=threshold,
        compact_keep_messages=keep,
        model=model,
        judge_model=judge_model,
    )
