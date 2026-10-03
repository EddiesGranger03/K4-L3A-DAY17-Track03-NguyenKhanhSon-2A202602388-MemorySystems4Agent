from __future__ import annotations

from pathlib import Path

try:
    from agent_advanced import AdvancedAgent
    from agent_baseline import BaselineAgent
    from config import load_config
    from memory_store import (
        CONFIDENCE_THRESHOLD,
        CompactMemoryManager,
        Entity,
        UserProfileStore,
        extract_profile_updates,
        extract_structured_entities,
    )
    from model_provider import ProviderConfig
except ImportError:  # pragma: no cover
    from src.agent_advanced import AdvancedAgent
    from src.agent_baseline import BaselineAgent
    from src.config import load_config
    from src.memory_store import (
        CONFIDENCE_THRESHOLD,
        CompactMemoryManager,
        Entity,
        UserProfileStore,
        extract_profile_updates,
        extract_structured_entities,
    )
    from src.model_provider import ProviderConfig


def make_config(tmp_path: Path):
    """Build an isolated config for tests (small compact threshold)."""
    base = load_config(Path(__file__).resolve().parent.parent)
    state_dir = tmp_path / "state"
    state_dir.mkdir(parents=True, exist_ok=True)
    base.state_dir = state_dir
    base.data_dir = Path(__file__).resolve().parent.parent / "data"
    base.compact_threshold_tokens = 200
    base.compact_keep_messages = 2
    base.model = ProviderConfig(provider="openai", model_name="offline-test", temperature=0.0)
    base.judge_model = ProviderConfig(provider="openai", model_name="offline-test", temperature=0.0)
    return base


def test_user_markdown_read_write_edit(tmp_path: Path) -> None:
    cfg = make_config(tmp_path)
    store = UserProfileStore(cfg.state_dir / "profiles")
    store.write_text("dungct", "# Hồ sơ người dùng\n\n- Tên: DũngCT\n")
    assert "DũngCT" in store.read_text("dungct")
    assert store.file_size("dungct") > 0
    assert store.edit_text("dungct", "DũngCT", "DũngCT Stress") is True
    assert "DũngCT Stress" in store.read_text("dungct")
    assert store.edit_text("dungct", "không-tồn-tại-xyz", "x") is False
    store.upsert_fact("dungct", "location", "Huế")
    assert store.facts("dungct")["location"] == "Huế"
    # conflict handling: newest wins, no duplicates
    store.upsert_fact("dungct", "location", "Đà Nẵng")
    text = store.read_text("dungct")
    assert "Đà Nẵng" in text and text.count("Nơi ở") == 1


def test_compact_trigger(tmp_path: Path) -> None:
    cfg = make_config(tmp_path)
    mgr = CompactMemoryManager(threshold_tokens=cfg.compact_threshold_tokens, keep_messages=2)
    for i in range(12):
        mgr.append("t1", "user", f"tin nhắn dài số {i} " + "nội dung benchmark memory " * 8)
    assert mgr.compaction_count("t1") >= 1
    ctx = mgr.context("t1")
    assert str(ctx.get("summary", "")).strip() != ""
    assert len(ctx["messages"]) <= 2  # type: ignore[index]


def test_cross_session_recall(tmp_path: Path) -> None:
    cfg = make_config(tmp_path)
    adv = AdvancedAgent(config=cfg, force_offline=True)
    base = BaselineAgent(config=cfg, force_offline=True)

    adv.reply("dungct", "thread-A", "Mình tên là DũngCT, đang ở Huế và làm MLOps engineer.")
    adv.reply("dungct", "thread-A", "Đồ uống yêu thích là cà phê sữa đá, trả lời ngắn gọn nhé.")
    res_adv = adv.reply("dungct", "thread-B-moi", "Nhắc lại giúp mình tên, nơi ở và nghề nghiệp?")
    assert "DũngCT" in res_adv["text"]
    assert "Huế" in res_adv["text"]
    assert "MLOps engineer" in res_adv["text"]

    base.reply("dungct", "thread-A", "Mình tên là DũngCT, đang ở Huế và làm MLOps engineer.")
    res_base = base.reply("dungct", "thread-B-moi", "Nhắc lại giúp mình tên, nơi ở và nghề nghiệp?")
    assert "DũngCT" not in res_base["text"] or "chưa lưu" in res_base["text"].lower() or "chưa thể" in res_base["text"]


def test_compact_reduces_prompt_load_on_long_thread(tmp_path: Path) -> None:
    cfg = make_config(tmp_path)
    adv = AdvancedAgent(config=cfg, force_offline=True)
    base = BaselineAgent(config=cfg, force_offline=True)
    long_turn = ("Tin benchmark dài về Artemis, X-59, WMO và kế hoạch điện BC " * 12)
    for i in range(14):
        adv.reply("u1", "long", f"{long_turn} lượt {i}")
        base.reply("u1", "long", f"{long_turn} lượt {i}")
    assert adv.compaction_count("long") >= 1
    assert base.compaction_count("long") == 0
    assert adv.prompt_token_usage("long") < base.prompt_token_usage("long")


def test_structured_entity_extraction_confidence(tmp_path: Path) -> None:
    # Structured candidates carry calibrated confidence; noise is visible but weak.
    ents = extract_structured_entities("Mình tên là DũngCT, đang ở Huế và làm MLOps engineer.")
    by_kind = {e.kind: e for e in ents}
    assert by_kind["name"].value == "DũngCT" and by_kind["name"].confidence >= CONFIDENCE_THRESHOLD
    assert by_kind["location"].value == "Huế" and by_kind["location"].confidence >= 0.9
    assert isinstance(ents[0], Entity) and all(hasattr(e, "detail") for e in ents)

    # Joke / travel noise is proposed with LOW confidence and filtered out.
    noise = extract_structured_entities(
        "Hay là chuyển sang product manager cho đỡ canh pipeline, nhưng đó chỉ là câu đùa. "
        "Hà Nội chỉ là nơi mình bay ra họp hai ngày."
    )
    assert any(e.kind == "profession" and e.confidence < CONFIDENCE_THRESHOLD for e in noise)
    assert any(e.kind == "location" and e.value == "Hà Nội" and e.confidence < CONFIDENCE_THRESHOLD
               for e in noise)
    filtered = extract_profile_updates(
        "Hay là chuyển sang product manager cho đỡ canh pipeline, nhưng đó chỉ là câu đùa."
    )
    assert "profession" not in filtered
    assert extract_profile_updates("Mình tên gì?") == {}


def test_memory_decay(tmp_path: Path) -> None:
    cfg = make_config(tmp_path)
    store = UserProfileStore(cfg.state_dir / "profiles")
    store.decay_half_life_turns = 2.0
    store.upsert_fact("u1", "location", "Huế")
    store.upsert_fact("u1", "drink", "cà phê sữa đá")
    fresh_drink = store.fact_strength("u1", "drink")
    assert fresh_drink >= 1.0
    # Simulate time passing without re-mentioning the drink: strength decays.
    for _ in range(12):
        store.touch("u1")
    assert store.fact_strength("u1", "drink") < fresh_drink
    # Re-mentioning refreshes one fact; the stale one can be pruned.
    store.upsert_fact("u1", "location", "Đà Nẵng")
    removed = store.prune_decayed("u1", min_strength=0.15)
    assert "drink" in removed
    assert store.facts("u1").get("location") == "Đà Nẵng"
    # Guardrail: decay never wipes the last remaining fact.
    assert store.prune_decayed("u1", min_strength=999.0) == []


def test_negation_and_word_boundary_regressions(tmp_path: Path) -> None:
    # "không còn ở X" marks X as OLD: Huế (affirmed) must beat Đà Nẵng (negated).
    upd = extract_profile_updates(
        "Giờ mình đang ở Huế chứ không còn ở Đà Nẵng mỗi ngày nữa."
    )
    assert upd.get("location") == "Huế"

    # "đừng nói backend nữa / thông tin cũ" must not resurrect the stale job.
    upd = extract_profile_updates(
        "Nếu nhắc lại nghề nghiệp, đừng nói backend engineer nữa nhé, "
        "vì đó là thông tin cũ."
    )
    assert "profession" not in upd

    # Replacement in the same message still wins: negated old + affirmed new.
    upd = extract_profile_updates(
        "Mình không còn làm backend engineer nữa, giờ chuyển sang MLOps engineer."
    )
    assert upd.get("profession") == "MLOps engineer"

    # "AI" must be a standalone word: "hai mối quan tâm" must not set interests.
    assert extract_profile_updates(
        "Tóm tắt ngắn về mình: tên, nghề nghiệp hiện tại và hai mối quan tâm kỹ thuật chính."
    ).get("interests") is None
    # ...while a real mention still works.
    assert extract_profile_updates("Mình vẫn thích Python, AI ứng dụng nhé.").get("interests") == "Python, AI"
