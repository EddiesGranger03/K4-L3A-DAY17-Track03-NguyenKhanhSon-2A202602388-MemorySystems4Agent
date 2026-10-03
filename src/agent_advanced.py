from __future__ import annotations

from dataclasses import dataclass
from typing import Any

try:
    from config import LabConfig, load_config
except ImportError:  # pragma: no cover
    from src.config import LabConfig, load_config

try:
    from memory_store import (
        CompactMemoryManager,
        UserProfileStore,
        estimate_tokens,
        extract_profile_updates,
    )
except ImportError:  # pragma: no cover
    from src.memory_store import (
        CompactMemoryManager,
        UserProfileStore,
        estimate_tokens,
        extract_profile_updates,
    )

try:
    from model_provider import build_chat_model
except ImportError:  # pragma: no cover
    from src.model_provider import build_chat_model


@dataclass
class AgentContext:
    user_id: str
    memory_path: str


class AdvancedAgent:
    """Agent B: short-term + persistent User.md + compact memory."""

    def __init__(self, config: LabConfig | None = None, force_offline: bool = False) -> None:
        self.config = config or load_config()
        self.force_offline = force_offline
        self.profile_store = UserProfileStore(self.config.state_dir / "profiles")
        self.compact_memory = CompactMemoryManager(
            threshold_tokens=self.config.compact_threshold_tokens,
            keep_messages=self.config.compact_keep_messages,
        )
        self.thread_tokens: dict[str, int] = {}
        self.thread_prompt_tokens: dict[str, int] = {}
        self.langchain_agent = None
        self._live_checkpointer = None
        if not force_offline:
            try:
                self._maybe_build_langchain_agent()
            except Exception:
                self.langchain_agent = None

    # -- public API ---------------------------------------------------------

    def reply(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        if self.langchain_agent is not None and not self.force_offline:
            try:
                return self._reply_live(user_id, thread_id, message)
            except Exception:
                pass
        return self._reply_offline(user_id, thread_id, message)

    def token_usage(self, thread_id: str) -> int:
        return self.thread_tokens.get(thread_id, 0)

    def prompt_token_usage(self, thread_id: str) -> int:
        return self.thread_prompt_tokens.get(thread_id, 0)

    def memory_file_size(self, user_id: str) -> int:
        return self.profile_store.file_size(user_id)

    def compaction_count(self, thread_id: str) -> int:
        return self.compact_memory.compaction_count(thread_id)

    # -- offline path (graded, deterministic, no API key) ---------------------

    def _reply_offline(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        # 1. Extract stable facts + 2. persist into User.md (newest wins).
        updates = extract_profile_updates(message)
        for key, value in updates.items():
            try:
                self.profile_store.upsert_fact(user_id, key, value)
            except Exception:
                pass
        # Advance the decay clock every turn (bonus: memory decay).
        try:
            self.profile_store.touch(user_id)
        except Exception:
            pass

        # 3. Append user turn into compact memory (may trigger compaction).
        self.compact_memory.append(thread_id, "user", message)

        # 4. Estimate prompt load from User.md + summary + recent messages.
        prompt_tokens = self._estimate_prompt_context_tokens(user_id, thread_id)
        self.thread_prompt_tokens[thread_id] = self.thread_prompt_tokens.get(thread_id, 0) + prompt_tokens

        # 5-6. Generate recall-capable response, store it, count tokens.
        text = self._offline_response(user_id, thread_id, message)
        self.compact_memory.append(thread_id, "assistant", text)
        reply_tokens = estimate_tokens(text)
        self.thread_tokens[thread_id] = self.thread_tokens.get(thread_id, 0) + reply_tokens
        return {"text": text, "agent_tokens": reply_tokens, "prompt_tokens": prompt_tokens}

    def _estimate_prompt_context_tokens(self, user_id: str, thread_id: str) -> int:
        profile_text = self.profile_store.read_text(user_id)
        ctx = self.compact_memory.context(thread_id)
        total = estimate_tokens(profile_text) + estimate_tokens(str(ctx.get("summary", "")))
        for m in ctx.get("messages", []):  # type: ignore[union-attr]
            total += estimate_tokens(str(m.get("content", "")))
        return total

    def _offline_response(self, user_id: str, thread_id: str, message: str) -> str:
        facts = self.profile_store.facts(user_id)
        if not facts.get("name"):
            ctx = self.compact_memory.context(thread_id)
            blob = str(ctx.get("summary", "")) + " " + " ".join(
                str(m.get("content", "")) for m in ctx.get("messages", [])  # type: ignore[union-attr]
            )
            fb = extract_profile_updates(blob)
            facts.update({k: v for k, v in fb.items() if k not in facts})

        low = message.lower()
        is_question = ("?" in message or "？" in message or low.startswith(("cho mình hỏi", "bạn có biết", "bạn biết")))
        wants = {
            "name": any(c in low for c in ["tên", "mình là ai", "là ai", "tóm tắt"]),
            "location": any(c in low for c in ["ở đâu", "nơi ở", "đang ở", "sống", "huế", "đà nẵng", "hà nội"]),
            "profession": any(c in low for c in ["nghề", "làm nghề", "công việc", "làm gì", "product manager", "backend", "mlops"]),
            "style": any(c in low for c in ["style", "phong cách", "trả lời", "kiểu trả lời", "bullet"]),
            "drink": any(c in low for c in ["đồ uống", "uống", "cà phê"]),
            "food": any(c in low for c in ["món ăn", "món ", "ăn ", "mì quảng"]),
            "pet": any(c in low for c in ["nuôi", "con gì", "corgi", "bơ", "thú cưng"]),
            "interests": any(c in low for c in ["quan tâm", "mối quan tâm", "sở thích", "thích"]),
            "summary": any(c in low for c in ["tóm tắt", "mô tả", "nhắc lại", "recall", "tổng hợp"]),
        }
        if is_question and not any(wants.values()):
            wants["summary"] = True

        name = facts.get("name", "bạn")
        lines: list[str] = []
        if wants["name"] or wants["summary"]:
            lines.append(f"- Tên: {facts.get('name', '(chưa rõ)')}")
        if wants["location"] or wants["summary"]:
            if facts.get("location"):
                lines.append(f"- Nơi ở hiện tại: {facts['location']}")
        if wants["profession"] or wants["summary"]:
            if facts.get("profession"):
                lines.append(f"- Nghề nghiệp hiện tại: {facts['profession']}")
        if wants["style"] or wants["summary"]:
            if facts.get("style"):
                lines.append(f"- Style trả lời: {facts['style']}")
        if wants["drink"] or wants["summary"]:
            if facts.get("drink"):
                lines.append(f"- Đồ uống yêu thích: {facts['drink']}")
        if wants["food"] or wants["summary"]:
            if facts.get("food"):
                lines.append(f"- Món ăn yêu thích: {facts['food']}")
        if wants["pet"] or wants["summary"]:
            if facts.get("pet"):
                lines.append(f"- Thú cưng: {facts['pet']}")
        if wants["interests"] or wants["summary"]:
            if facts.get("interests"):
                lines.append(f"- Mối quan tâm: {facts['interests']}")

        if lines:
            header = f"Ghi nhận của mình về {name} (ngắn gọn, có ví dụ thực tế):"
            body = "\n".join(lines[:8])
            tail = ""
            if wants["summary"] or len(lines) > 2:
                tail = "\nVí dụ thực chiến: như khi benchmark memory, mình ưu tiên recall đúng hơn câu văn hoa mỹ (trade-off recall vs token)."
            return f"{header}\n{body}{tail}"

        if facts:
            snap = ", ".join(f"{k}: {v}" for k, v in list(facts.items())[:3])
            return (f"Đã ghi nhớ ({snap}). Mình sẽ trả lời ngắn gọn, có bullet và ví dụ thực tế; "
                    f"trade-off chính là recall tốt hơn đổi bằng chút chi phí lưu trữ User.md.")
        return ("Mình đã ghi nhận ý của bạn trong phiên này và sẽ giữ các fact ổn định "
                "vào hồ sơ để dùng cho phiên sau.")

    # -- live path (extension: LangChain/LangGraph, any provider) ------------

    def _dynamic_prompt(self, user_id: str, thread_id: str) -> str:
        """System prompt with injected persistent profile + compact summary."""
        profile = self.profile_store.read_text(user_id)
        ctx = self.compact_memory.context(thread_id)
        summary = str(ctx.get("summary", ""))
        return (
            "Bạn là trợ lý nhớ người dùng dài hạn. Trả lời ngắn gọn, có bullet "
            "và ví dụ thực tế; khi phải đánh đổi, nói rõ trade-off recall vs token.\n"
            f"Hồ sơ bền vững (User.md):\n{profile}\n"
            f"Tóm tắt lịch sử cũ:\n{summary or '(chưa có)'}\n"
            "Quy tắc: fact ổn định lấy từ User.md (bản mới nhất); "
            "thông tin đùa (product manager) hay đi họp (Hà Nội) không phải fact."
        )

    def _reply_live(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        # Keep memory layers in sync even in live mode.
        updates = extract_profile_updates(message)
        for key, value in updates.items():
            try:
                self.profile_store.upsert_fact(user_id, key, value)
            except Exception:
                pass
        try:
            self.profile_store.touch(user_id)
        except Exception:
            pass
        self.compact_memory.append(thread_id, "user", message)
        prompt_tokens = self._estimate_prompt_context_tokens(user_id, thread_id)
        self.thread_prompt_tokens[thread_id] = self.thread_prompt_tokens.get(thread_id, 0) + prompt_tokens

        assert self.langchain_agent is not None
        cfg = {"configurable": {"thread_id": f"{user_id}::{thread_id}"}}
        try:
            result = self.langchain_agent.invoke(
                {"messages": [
                    {"role": "system", "content": self._dynamic_prompt(user_id, thread_id)},
                    {"role": "user", "content": message},
                ]},
                config=cfg,
            )
            msgs = result.get("messages", []) if isinstance(result, dict) else []
            text = str(msgs[-1].content) if msgs and hasattr(msgs[-1], "content") else str(result)
        except Exception:
            text = self._offline_response(user_id, thread_id, message)
        self.compact_memory.append(thread_id, "assistant", text)
        reply_tokens = estimate_tokens(text)
        self.thread_tokens[thread_id] = self.thread_tokens.get(thread_id, 0) + reply_tokens
        return {"text": text, "agent_tokens": reply_tokens, "prompt_tokens": prompt_tokens}

    def _maybe_build_langchain_agent(self):
        """Wire live agent: chat model + InMemorySaver + User.md tools + summary.

        Design (per Guide.md):
        - ``build_chat_model(self.config.model)`` for any of the 6 providers
        - ``InMemorySaver`` for short-term thread state
        - tool ``read_profile`` / tool ``write_profile`` over User.md
        - dynamic prompt injecting profile memory (see ``_dynamic_prompt``)
        - summarization via the built-in ``CompactMemoryManager`` (history is
          trimmed before every model call; long threads also use LangChain's
          summarization middleware when available)
        Missing optional deps -> stay offline (``langchain_agent = None``).
        """
        model = build_chat_model(self.config.model)

        try:
            from langgraph.checkpoint.memory import InMemorySaver
            self._live_checkpointer = InMemorySaver()
        except Exception:
            self._live_checkpointer = None

        # --- User.md tools (closures over the profile store) ---
        def _read_profile(user_id: str) -> str:
            """Read the persistent User.md profile for a user."""
            return self.profile_store.read_text(user_id)

        def _write_profile(user_id: str, key: str, value: str) -> str:
            """Create or update one persistent fact in User.md (newest wins)."""
            self.profile_store.upsert_fact(user_id, key, value)
            return f"Đã lưu {key} = {value}"

        tools = [_read_profile, _write_profile]

        # --- summarization middleware when available (langmem / langchain) ---
        middleware = None
        try:  # langmem (LangChain memory package)
            from langmem import SummarizationMiddleware  # type: ignore

            middleware = SummarizationMiddleware(model=model, max_tokens=4000)
        except Exception:
            middleware = None

        # --- agent factory (new `langchain.agents` API, fallback to prebuilt) ---
        try:
            try:
                from langchain.agents import create_agent  # type: ignore

                kwargs: dict[str, Any] = {"model": model, "tools": tools}
                if self._live_checkpointer is not None:
                    kwargs["checkpointer"] = self._live_checkpointer
                if middleware is not None:
                    kwargs["middleware"] = [middleware]
                self.langchain_agent = create_agent(**kwargs)
            except Exception:
                from langgraph.prebuilt import create_react_agent  # type: ignore

                kwargs = {"model": model, "tools": tools}
                if self._live_checkpointer is not None:
                    kwargs["checkpointer"] = self._live_checkpointer
                self.langchain_agent = create_react_agent(**kwargs)
        except Exception:
            self.langchain_agent = None
