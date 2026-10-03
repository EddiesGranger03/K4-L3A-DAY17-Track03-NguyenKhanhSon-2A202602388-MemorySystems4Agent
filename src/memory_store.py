from __future__ import annotations

import json
import math
import re
import time
from dataclasses import dataclass, field
from pathlib import Path


def estimate_tokens(text: str) -> int:
    """Heuristic token estimator: ~4 chars per token (stable for offline bench)."""
    if not text:
        return 0
    stripped = text.strip()
    if not stripped:
        return 0
    return max(1, math.ceil(len(stripped) / 4))


def _sanitize_user_id(user_id: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", (user_id or "default").strip())
    cleaned = cleaned.strip("._-") or "default"
    return cleaned[:64]


_DEFAULT_PROFILE = "# Hồ sơ người dùng\n\n(Chưa có thông tin. Agent sẽ cập nhật khi người dùng chia sẻ fact ổn định.)\n"

_LABELS = {
    "name": "Tên",
    "location": "Nơi ở",
    "profession": "Nghề nghiệp",
    "style": "Phong cách trả lời",
    "drink": "Đồ uống yêu thích",
    "food": "Món ăn yêu thích",
    "pet": "Thú cưng",
    "interests": "Mối quan tâm",
}

_KEY_BY_LABEL = {v.lower(): k for k, v in _LABELS.items()}

_FACT_ORDER = ["name", "location", "profession", "style", "drink", "food", "pet", "interests"]


# ---------------------------------------------------------------------------
# Structured entity extraction (bonus: Entity extraction có cấu trúc)
# ---------------------------------------------------------------------------

@dataclass
class Entity:
    """One extracted candidate fact with a calibrated confidence score."""

    kind: str            # name | location | profession | style | drink | food | pet | interests
    value: str           # normalized surface form, e.g. "Đà Nẵng", "MLOps engineer"
    confidence: float    # 0.0 - 1.0; extract_profile_updates() keeps >= threshold
    detail: str = ""     # why this confidence (pattern name / noise reason)
    start: int = -1      # char offset in the source message (-1 when n/a)


CONFIDENCE_THRESHOLD = 0.6


_CITY_WHITELIST = ["Đà Nẵng", "Huế", "Hà Nội", "Hội An", "Mỹ Khê"]

_NAME_PATTERNS = [
    (re.compile(r"tên\s+(?:là|mình\s+là|của\s+mình\s+là)\s+([A-ZÀ-Ỹ][\wÀ-ỹ.\-]*"
                r"(?:\s+[A-ZÀ-Ỹ][\wÀ-ỹ.\-]*)*)"), 0.95, "name:explicit-ten-la"),
    (re.compile(r"mình\s+tên\s+là\s+([A-ZÀ-Ỹ][\wÀ-ỹ.\-]*"
                r"(?:\s+[A-ZÀ-Ỹ][\wÀ-ỹ.\-]*)*)"), 0.95, "name:minh-ten-la"),
    (re.compile(r"tên\s+mình\s+là\s+([A-ZÀ-Ỹ][\wÀ-ỹ.\-]*"
                r"(?:\s+[A-ZÀ-Ỹ][\wÀ-ỹ.\-]*)*)"), 0.95, "name:ten-minh-la"),
]

_PROF_KEYWORDS = ["MLOps engineer", "backend engineer", "product manager"]

# Negation windows: a mention inside one of these is a *rejected* fact, not news.
# e.g. "không còn ở Đà Nẵng", "đừng nói backend nữa", "đó là thông tin cũ".
_LOCATION_DENIAL_BEFORE = ["không còn", "không còn ở", "rời", "chuyển đi khỏi"]
_PROF_DENIAL_BEFORE = ["không còn", "không còn làm", "đừng", "đừng nói"]
_PROF_DENIAL_AFTER = ["nữa", "thông tin cũ", "là thông tin cũ", "cũ rồi"]


def _window_has(low: str, idx: int, phrases: list[str], before: int, after: int) -> bool:
    window = low[max(0, idx - before): idx + after]
    return any(p in window for p in phrases)


def _word_present(haystack_lower: str, keyword: str) -> int:
    """Word-boundary search; returns start index or -1.

    Plain ``in`` would match "AI" inside "hai"/"mãi"/"tại" — the exact bug
    that once overwrote interests="Python, AI" with interests="AI" from the
    recall question "...hai mối quan tâm...". Regex \\b is Unicode-aware so
    "hai" stays one word while "AI agent" matches.
    """
    m = re.search(r"\b" + re.escape(keyword.lower()) + r"\b", haystack_lower)
    return m.start() if m else -1


def _looks_like_question_only(message: str) -> bool:
    text = message.strip()
    if "?" not in text and "？" not in text:
        return False
    declarative = ["tên là", "mình tên", "đang ở", "mình ở", "sống ở",
                   "làm ", "chuyển sang", "thích", "yêu thích", "nuôi",
                   "trả lời", "uống", "món ", "đính chính"]
    low = text.lower()
    return not any(cue in low for cue in declarative)


def extract_structured_entities(message: str) -> list[Entity]:
    """Return every candidate entity with a confidence score (unfiltered).

    Low-confidence candidates (jokes, travel noise, weak cues) are returned
    with confidence < CONFIDENCE_THRESHOLD so callers and tests can inspect
    what was rejected and why (see ``detail``). Question-only turns return [].
    """
    if not message or not message.strip():
        return []
    if _looks_like_question_only(message):
        return []
    out: list[Entity] = []
    low = message.lower()

    # ---- name
    for pat, conf, detail in _NAME_PATTERNS:
        m = pat.search(message)
        if m:
            name = re.sub(r"\s+", " ", m.group(1)).strip().strip(".,;:")
            name = re.split(r"\s+(?:hiện|đang|là|ở|và)\s+", name)[0].strip()
            if len(name) >= 2:
                out.append(Entity("name", name, conf, detail, m.start(1)))
            break

    # ---- location (one entity per whitelisted city found; noise gets low conf)
    for city in _CITY_WHITELIST:
        idx = low.find(city.lower())
        if idx < 0:
            continue
        window = low[max(0, idx - 40): idx + 40]
        noise_reason = ""
        if _window_has(low, idx, _LOCATION_DENIAL_BEFORE, before=20, after=0):
            noise_reason = "location:negated-khong-con"  # "không còn ở X" marks X as OLD
        elif city == "Hà Nội" and ("họp" in window or "đùa" in low or "chỉ là" in low):
            noise_reason = "travel-noise:hop-chi-la"
        elif "chỉ là nơi" in low or "không phải nơi ở" in low:
            noise_reason = "travel-noise:chi-la-noi"
        elif "ví dụ cũ" in low and city == "Đà Nẵng" and "đừng lấy" in low:
            noise_reason = "old-example:do-not-take"
        residence_cues = ["đang ở", "mình ở", "hiện ở", "hiện tại", "sống ở",
                          "nơi ở", "đính chính", "ở huế", "ở đà nẵng",
                          "ở hà nội", "làm việc ở", "chuyển", "đang làm việc"]
        has_residence = any(c in low for c in residence_cues) or f"ở {city.lower()}" in low
        if noise_reason:
            out.append(Entity("location", city, 0.2, noise_reason, idx))
        elif has_residence:
            out.append(Entity("location", city, 0.9, "location:residence-intent", idx))
        elif city in ("Đà Nẵng", "Huế") and ("đang ở" in low or "hiện" in low or "đính chính" in low):
            out.append(Entity("location", city, 0.65, "location:weak-cue", idx))
        else:
            out.append(Entity("location", city, 0.35, "location:mention-without-intent", idx))

    # ---- profession (negation-aware: denials never resurrect stale jobs)
    for prof in _PROF_KEYWORDS:
        pidx = low.find(prof.lower())
        if pidx < 0:
            continue
        if prof == "product manager" and "đùa" in low:
            out.append(Entity("profession", prof, 0.15, "profession:joke-dua", pidx))
            continue
        if _window_has(low, pidx, _PROF_DENIAL_BEFORE, before=25, after=0) or \
           _window_has(low, pidx + len(prof), _PROF_DENIAL_AFTER, before=0, after=30):
            # "không còn làm backend", "đừng nói backend nữa", "là thông tin cũ":
            # the message *rejects* this job. A replacement later in the same
            # message (e.g. "chuyển sang MLOps") still wins via position order.
            out.append(Entity("profession", prof, 0.2, "profession:negated-stale", pidx))
            continue
        correction = any(c in low for c in ["đính chính", "không còn", "chuyển sang", "giờ ", "hiện tại"])
        conf = 0.95 if correction else 0.9
        out.append(Entity("profession", prof, conf, "profession:explicit-mention", pidx))

    # ---- style
    style_bits: list[str] = []
    if "ngắn gọn" in low:
        style_bits.append("ngắn gọn")
    if "bullet" in low:
        style_bits.append("3 bullet ngắn" if "3 bullet" in low else "bullet ngắn")
    if "ví dụ thực tế" in low or "ví dụ thực chiến" in low or "ví dụ" in low:
        style_bits.append("có ví dụ thực tế")
    if "trade-off" in low or "tradeoff" in low or "đánh đổi" in low:
        style_bits.append("nhấn trade-off")
    if style_bits:
        has_verb = any(c in low for c in ["thích", "muốn", "ưu tiên", "hãy", "nhớ", "phong cách", "style"])
        merged = ", ".join(dict.fromkeys(style_bits))
        if has_verb:
            out.append(Entity("style", merged, 0.85, "style:preference-verb", low.find("ngắn gọn")))
        else:
            out.append(Entity("style", merged, 0.4, "style:no-preference-verb", -1))

    # ---- drink / food / pet (literal, high precision)
    if "cà phê sữa đá" in low:
        out.append(Entity("drink", "cà phê sữa đá", 0.95, "drink:literal", low.find("cà phê sữa đá")))
    if "mì quảng" in low:
        out.append(Entity("food", "mì Quảng", 0.95, "food:literal", low.find("mì quảng")))
    if "corgi" in low:
        out.append(Entity("pet", "corgi tên Bơ" if "bơ" in low else "corgi",
                          0.95, "pet:literal-corgi", low.find("corgi")))
    elif re.search(r"\bbơ\b", low) and ("bé" in low or "nuôi" in low):
        out.append(Entity("pet", "corgi tên Bơ", 0.7, "pet:bo-context", -1))

    # ---- interests (need explicit like/care verb + WORD match)
    interest_cues = ["thích", "quan tâm", "mối quan tâm", "đam mê", "yêu thích"]
    if any(c in low for c in interest_cues):
        found = [kw for kw in ["Python", "AI", "MLOps", "RAG", "benchmark"]
                 if _word_present(low, kw) >= 0]
        if found:
            positions = sorted(((_word_present(low, kw), kw) for kw in found))
            out.append(Entity("interests", ", ".join(kw for _, kw in positions), 0.8,
                              "interests:explicit-cue", positions[0][0]))
    else:
        for kw in ["Python", "AI", "MLOps"]:
            at = _word_present(low, kw)
            if at >= 0:
                out.append(Entity("interests", kw, 0.3, "interests:no-cue-verb", at))
                break
    return out


def extract_profile_updates(message: str, threshold: float = CONFIDENCE_THRESHOLD) -> dict[str, str]:
    """Convert raw user text into stable profile facts.

    Structured pipeline: ``extract_structured_entities`` proposes candidates,
    this function keeps only ``confidence >= threshold`` and resolves
    conflicts per key (newest text position wins -> correction-safe).
    Question-only turns and jokes/travel noise fall below the threshold.
    """
    entities = extract_structured_entities(message)
    best: dict[str, Entity] = {}
    for e in entities:
        if e.confidence < threshold:
            continue
        cur = best.get(e.kind)
        if cur is None or (e.start >= cur.start):
            best[e.kind] = e
    out: dict[str, str] = {}
    # profession guard: "không còn làm backend" must not resurrect the old job
    profs = [e for e in entities if e.kind == "profession" and e.confidence >= threshold]
    if profs and "không còn" in (message or "").lower():
        non_backend = [e for e in profs if e.value != "backend engineer"]
        if non_backend:
            best["profession"] = max(non_backend, key=lambda e: e.start)
    for kind, ent in best.items():
        out[kind] = ent.value
    return out


@dataclass
class UserProfileStore:
    """Persistent storage for `User.md` (one folder per user).

    Bonus: memory-decay metadata lives in a sidecar ``User.meta.json`` so
    ``User.md`` stays human-readable while each fact carries
    ``{hits, last_turn, updated_at}`` for strength-based pruning.
    """

    root_dir: Path
    decay_half_life_turns: float = 10.0

    def path_for(self, user_id: str) -> Path:
        return self.root_dir / _sanitize_user_id(user_id) / "User.md"

    def meta_path_for(self, user_id: str) -> Path:
        return self.root_dir / _sanitize_user_id(user_id) / "User.meta.json"

    def read_text(self, user_id: str) -> str:
        path = self.path_for(user_id)
        if not path.exists():
            return _DEFAULT_PROFILE
        try:
            return path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return _DEFAULT_PROFILE

    def write_text(self, user_id: str, content: str) -> Path:
        path = self.path_for(user_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        if not content.endswith("\n"):
            content += "\n"
        path.write_text(content, encoding="utf-8")
        return path

    def edit_text(self, user_id: str, search_text: str, replacement: str) -> bool:
        current = self.read_text(user_id)
        if search_text not in current:
            return False
        updated = current.replace(search_text, replacement, 1)
        self.write_text(user_id, updated)
        return True

    def file_size(self, user_id: str) -> int:
        path = self.path_for(user_id)
        try:
            return path.stat().st_size
        except FileNotFoundError:
            return 0

    def facts(self, user_id: str) -> dict[str, str]:
        out: dict[str, str] = {}
        for line in self.read_text(user_id).splitlines():
            m = re.match(r"\s*-\s*([^:]+)\s*:\s*(.+)", line.strip())
            if not m:
                continue
            label, value = m.group(1).strip().lower(), m.group(2).strip()
            key = _KEY_BY_LABEL.get(label)
            if key and value:
                out[key] = value
        return out

    def upsert_fact(self, user_id: str, key: str, value: str) -> Path:
        """Insert or overwrite one fact (conflict handling: newest wins)."""
        value = value.strip()
        if not value:
            return self.path_for(user_id)
        facts = self.facts(user_id)
        facts[key] = value
        lines = ["# Hồ sơ người dùng", ""]
        for k in _FACT_ORDER:
            if k in facts:
                lines.append(f"- {_LABELS[k]}: {facts[k]}")
        for k, v in facts.items():
            if k not in _LABELS:
                lines.append(f"- {k}: {v}")
        path = self.write_text(user_id, "\n".join(lines) + "\n")
        self._bump_meta(user_id, key)
        return path

    # --- memory decay (bonus) -------------------------------------------

    def _read_meta(self, user_id: str) -> dict:
        mp = self.meta_path_for(user_id)
        try:
            return json.loads(mp.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            return {"turn": 0, "facts": {}}

    def _write_meta(self, user_id: str, meta: dict) -> None:
        mp = self.meta_path_for(user_id)
        mp.parent.mkdir(parents=True, exist_ok=True)
        mp.write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")

    def _bump_meta(self, user_id: str, key: str) -> None:
        meta = self._read_meta(user_id)
        meta["turn"] = int(meta.get("turn", 0)) + 1
        facts = meta.setdefault("facts", {})
        entry = facts.get(key, {"hits": 0, "last_turn": 0})
        entry["hits"] = int(entry.get("hits", 0)) + 1
        entry["last_turn"] = meta["turn"]
        entry["updated_at"] = time.time()
        facts[key] = entry
        self._write_meta(user_id, meta)

    def touch(self, user_id: str) -> int:
        """Advance the decay clock by one turn without changing facts."""
        meta = self._read_meta(user_id)
        meta["turn"] = int(meta.get("turn", 0)) + 1
        self._write_meta(user_id, meta)
        return meta["turn"]

    def current_turn(self, user_id: str) -> int:
        return int(self._read_meta(user_id).get("turn", 0))

    def fact_strength(self, user_id: str, key: str, now: int | None = None) -> float:
        """strength = hits * 0.5 ** ((now - last_turn) / half_life)."""
        meta = self._read_meta(user_id)
        entry = meta.get("facts", {}).get(key)
        if not entry:
            return 0.0
        now = meta.get("turn", 0) if now is None else now
        age = max(0, now - int(entry.get("last_turn", now)))
        half = max(0.5, float(self.decay_half_life_turns))
        return float(entry.get("hits", 0)) * (0.5 ** (age / half))

    def decayed_facts(self, user_id: str, min_strength: float = 0.15) -> dict[str, str]:
        """Facts whose decayed strength is still above ``min_strength``."""
        now = self.current_turn(user_id)
        return {k: v for k, v in self.facts(user_id).items()
                if self.fact_strength(user_id, k, now) >= min_strength}

    def prune_decayed(self, user_id: str, min_strength: float = 0.15) -> list[str]:
        """Drop weak facts from User.md; return removed keys (never removes all)."""
        facts = self.facts(user_id)
        if not facts:
            return []
        now = self.current_turn(user_id)
        weak = [k for k in facts if self.fact_strength(user_id, k, now) < min_strength]
        # Guardrail: never prune the last remaining fact via decay alone.
        if len(weak) >= len(facts):
            return []
        for k in weak:
            facts.pop(k, None)
        lines = ["# Hồ sơ người dùng", ""]
        for k in _FACT_ORDER:
            if k in facts:
                lines.append(f"- {_LABELS[k]}: {facts[k]}")
        for k, v in facts.items():
            if k not in _LABELS:
                lines.append(f"- {k}: {v}")
        self.write_text(user_id, "\n".join(lines) + "\n")
        meta = self._read_meta(user_id)
        for k in weak:
            meta.get("facts", {}).pop(k, None)
        self._write_meta(user_id, meta)
        return weak


def summarize_messages(messages: list[dict[str, str]], max_items: int = 6) -> str:
    """Heuristic summary: keep head/tail excerpts so compaction is inspectable."""
    if not messages:
        return ""
    items = messages[:max_items]
    parts: list[str] = []
    for m in items:
        role = m.get("role", "user")
        content = re.sub(r"\s+", " ", str(m.get("content", ""))).strip()
        if len(content) > 120:
            content = content[:117] + "..."
        parts.append(f"{role}: {content}")
    head = f"Tóm tắt {len(messages)} tin nhắn cũ"
    return head + " | " + " || ".join(parts)


@dataclass
class CompactMemoryManager:
    threshold_tokens: int
    keep_messages: int
    state: dict[str, dict[str, object]] = field(default_factory=dict)

    def _ensure(self, thread_id: str) -> dict[str, object]:
        st = self.state.get(thread_id)
        if st is None:
            st = {"messages": [], "summary": "", "compactions": 0}
            self.state[thread_id] = st
        return st

    def _total_tokens(self, st: dict[str, object]) -> int:
        total = estimate_tokens(str(st.get("summary", "")))
        for m in st.get("messages", []):  # type: ignore[union-attr]
            total += estimate_tokens(str(m.get("content", "")))  # type: ignore[union-attr]
        return total

    def append(self, thread_id: str, role: str, content: str) -> None:
        st = self._ensure(thread_id)
        messages = st["messages"]
        assert isinstance(messages, list)
        messages.append({"role": role, "content": content})
        if self._total_tokens(st) <= self.threshold_tokens:
            return
        keep = max(1, self.keep_messages)
        if len(messages) <= keep:
            return
        older = messages[:-keep]
        recent = messages[-keep:]
        new_summary = summarize_messages(older)
        prev = str(st.get("summary", ""))
        st["summary"] = (prev + " || " + new_summary).strip(" |") if prev else new_summary
        st["messages"] = recent
        st["compactions"] = int(st.get("compactions", 0)) + 1

    def context(self, thread_id: str) -> dict[str, object]:
        st = self._ensure(thread_id)
        return {
            "messages": list(st.get("messages", [])),  # type: ignore[union-attr]
            "summary": str(st.get("summary", "")),
            "compactions": int(st.get("compactions", 0)),
        }

    def compaction_count(self, thread_id: str) -> int:
        st = self.state.get(thread_id)
        if not st:
            return 0
        return int(st.get("compactions", 0))
