from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

try:
    from config import LabConfig, load_config
except ImportError:  # pragma: no cover
    from src.config import LabConfig, load_config

try:
    from memory_store import estimate_tokens
except ImportError:  # pragma: no cover
    from src.memory_store import estimate_tokens

try:
    from model_provider import build_chat_model
except ImportError:  # pragma: no cover
    from src.model_provider import build_chat_model


@dataclass
class SessionState:
    messages: list[dict[str, str]] = field(default_factory=list)
    token_usage: int = 0
    prompt_tokens_processed: int = 0


class BaselineAgent:
    """Agent A: within-thread memory only. New thread => total amnesia."""

    def __init__(self, config: LabConfig | None = None, force_offline: bool = False) -> None:
        self.config = config or load_config()
        self.force_offline = force_offline
        self.sessions: dict[str, SessionState] = {}
        self.langchain_agent = None
        self._live_checkpointer = None
        if not force_offline:
            try:
                self._maybe_build_langchain_agent()
            except Exception:
                self.langchain_agent = None

    def _session(self, thread_id: str) -> SessionState:
        st = self.sessions.get(thread_id)
        if st is None:
            st = SessionState()
            self.sessions[thread_id] = st
        return st

    def reply(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        if self.langchain_agent is not None and not self.force_offline:
            try:
                return self._reply_live(user_id, thread_id, message)
            except Exception:
                pass
        return self._reply_offline(thread_id, message)

    def token_usage(self, thread_id: str) -> int:
        st = self.sessions.get(thread_id)
        return st.token_usage if st else 0

    def prompt_token_usage(self, thread_id: str) -> int:
        st = self.sessions.get(thread_id)
        return st.prompt_tokens_processed if st else 0

    def compaction_count(self, thread_id: str) -> int:
        return 0

    def _reply_offline(self, thread_id: str, message: str) -> dict[str, Any]:
        st = self._session(thread_id)
        history_tokens = sum(estimate_tokens(m["content"]) for m in st.messages)
        prompt_tokens = history_tokens + estimate_tokens(message)
        st.prompt_tokens_processed += prompt_tokens

        st.messages.append({"role": "user", "content": message})
        text = self._offline_response(message)
        st.messages.append({"role": "assistant", "content": text})
        reply_tokens = estimate_tokens(text)
        st.token_usage += reply_tokens
        return {"text": text, "agent_tokens": reply_tokens, "prompt_tokens": prompt_tokens}

    def _offline_response(self, message: str) -> str:
        low = message.lower()
        recall_cues = ["tên gì", "tên mình", "mình là ai", "ở đâu", "nơi ở",
                       "nghề", "làm nghề", "style", "phong cách", "đồ uống",
                       "món ăn", "nuôi", "con gì", "tóm tắt", "nhắc lại",
                       "nhớ", "sở thích", "quan tâm"]
        if any(c in low for c in recall_cues):
            return ("Mình chưa lưu thông tin dài hạn trong phiên này nên chưa thể "
                    "nhắc lại. Bạn chia sẻ lại giúp mình nhé.")
        return ("Mình đã ghi nhận ý của bạn trong phiên này. "
                "Bạn nói rõ thêm một chút để mình hỗ trợ tiếp nhé.")

    def _reply_live(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        """Live path: same thread-scoped semantics, real model, no User.md."""
        st = self._session(thread_id)
        history_tokens = sum(estimate_tokens(m["content"]) for m in st.messages)
        prompt_tokens = history_tokens + estimate_tokens(message)
        st.prompt_tokens_processed += prompt_tokens
        st.messages.append({"role": "user", "content": message})
        assert self.langchain_agent is not None
        try:
            result = self.langchain_agent.invoke(
                {"messages": [{"role": "user", "content": message}]},
                config={"configurable": {"thread_id": f"{user_id}::{thread_id}"}},
            )
            msgs = result.get("messages", []) if isinstance(result, dict) else []
            text = str(msgs[-1].content) if msgs and hasattr(msgs[-1], "content") else str(result)
        except Exception:
            text = self._offline_response(message)
        st.messages.append({"role": "assistant", "content": text})
        reply_tokens = estimate_tokens(text)
        st.token_usage += reply_tokens
        return {"text": text, "agent_tokens": reply_tokens, "prompt_tokens": prompt_tokens}

    def _maybe_build_langchain_agent(self):
        """Wire a live thread-scoped agent (no persistent memory by design)."""
        model = build_chat_model(self.config.model)
        try:
            from langgraph.checkpoint.memory import InMemorySaver
            self._live_checkpointer = InMemorySaver()
        except Exception:
            self._live_checkpointer = None
        try:
            try:
                from langchain.agents import create_agent  # type: ignore

                kwargs: dict[str, Any] = {"model": model, "tools": []}
                if self._live_checkpointer is not None:
                    kwargs["checkpointer"] = self._live_checkpointer
                self.langchain_agent = create_agent(**kwargs)
            except Exception:
                from langgraph.prebuilt import create_react_agent  # type: ignore

                kwargs = {"model": model, "tools": []}
                if self._live_checkpointer is not None:
                    kwargs["checkpointer"] = self._live_checkpointer
                self.langchain_agent = create_react_agent(**kwargs)
        except Exception:
            self.langchain_agent = None
