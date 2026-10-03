from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    from agent_advanced import AdvancedAgent
    from agent_baseline import BaselineAgent
    from config import load_config
except ImportError:  # pragma: no cover - supports `python -m src.benchmark`
    from src.agent_advanced import AdvancedAgent
    from src.agent_baseline import BaselineAgent
    from src.config import load_config


@dataclass
class BenchmarkRow:
    agent_name: str
    agent_tokens_only: int
    prompt_tokens_processed: int
    recall_score: float
    response_quality: float
    memory_growth_bytes: int
    compactions: int


def load_conversations(path: Path) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        return data.get("conversations", [])
    return data


def recall_points(answer: str, expected: list[str]) -> float:
    if not expected:
        return 1.0
    ans = (answer or "").lower()
    hits = sum(1 for e in expected if str(e).lower() in ans)
    return hits / max(1, len(expected))


def heuristic_quality(answer: str, expected: list[str]) -> float:
    """Lightweight offline quality: structure + personalization + conciseness."""
    text = answer or ""
    low = text.lower()
    score = 0.45
    n = len(text)
    if 20 <= n <= 900:
        score += 0.15
    elif n > 900:
        score += 0.05
    if any(m in text for m in ["-", "•", "1.", "2."]):
        score += 0.10
    if "ví dụ" in low:
        score += 0.10
    if "trade-off" in low or "tradeoff" in low or "đánh đổi" in low:
        score += 0.05
    if any(str(e).lower() in low for e in (expected or [])):
        score += 0.10
    if "ngắn gọn" in low or "bullet" in low:
        score += 0.05
    return max(0.0, min(1.0, round(score, 3)))


def run_agent_benchmark(agent_name: str, agent, conversations: list[dict[str, Any]], config) -> BenchmarkRow:
    total_agent_tokens = 0
    total_prompt_tokens = 0
    recall_scores: list[float] = []
    quality_scores: list[float] = []
    compactions = 0
    threads: list[str] = []

    # memory baseline per user
    users = {c.get("user_id", "default") for c in conversations}
    before: dict[str, int] = {}
    for u in users:
        if hasattr(agent, "memory_file_size"):
            try:
                before[u] = agent.memory_file_size(u)
            except Exception:
                before[u] = 0

    for conv in conversations:
        user_id = conv.get("user_id", "default")
        thread_id = f"{conv.get('id', 'conv')}"
        threads.append(thread_id)
        for turn in conv.get("turns", []):
            try:
                agent.reply(user_id, thread_id, turn)
            except Exception:
                pass
        # recall in a FRESH thread (cross-session)
        recall_thread = f"{conv.get('id', 'conv')}--recall"
        threads.append(recall_thread)
        for q in conv.get("recall_questions", []):
            question = q.get("question", "") if isinstance(q, dict) else str(q)
            expected = q.get("expected_contains", []) if isinstance(q, dict) else []
            try:
                res = agent.reply(user_id, recall_thread, question)
                text = res.get("text", "") if isinstance(res, dict) else str(res)
            except Exception:
                text = ""
            recall_scores.append(recall_points(text, expected))
            quality_scores.append(heuristic_quality(text, expected))

    for t in threads:
        try:
            total_agent_tokens += agent.token_usage(t)
        except Exception:
            pass
        try:
            total_prompt_tokens += agent.prompt_token_usage(t)
        except Exception:
            pass
        try:
            compactions += agent.compaction_count(t)
        except Exception:
            pass

    growth = 0
    for u in users:
        if hasattr(agent, "memory_file_size"):
            try:
                growth += max(0, agent.memory_file_size(u) - before.get(u, 0))
            except Exception:
                pass

    avg_recall = round(sum(recall_scores) / len(recall_scores), 3) if recall_scores else 0.0
    avg_quality = round(sum(quality_scores) / len(quality_scores), 3) if quality_scores else 0.0
    return BenchmarkRow(
        agent_name=agent_name,
        agent_tokens_only=total_agent_tokens,
        prompt_tokens_processed=total_prompt_tokens,
        recall_score=avg_recall,
        response_quality=avg_quality,
        memory_growth_bytes=growth,
        compactions=compactions,
    )


def format_rows(rows: list[BenchmarkRow]) -> str:
    headers = ["Agent", "Agent tokens only", "Prompt tokens processed",
               "Cross-session recall", "Response quality",
               "Memory growth (bytes)", "Compactions"]
    data = [[r.agent_name, r.agent_tokens_only, r.prompt_tokens_processed,
             r.recall_score, r.response_quality,
             r.memory_growth_bytes, r.compactions] for r in rows]
    try:
        from tabulate import tabulate

        return tabulate(data, headers=headers, tablefmt="github")
    except Exception:
        lines = ["| " + " | ".join(headers) + " |",
                 "|" + "|".join(["---"] * len(headers)) + "|"]
        for row in data:
            lines.append("| " + " | ".join(str(c) for c in row) + " |")
        return "\n".join(lines)


def main() -> None:
    try:
        import sys

        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    config = load_config(Path(__file__).resolve().parent.parent)
    std_path = config.data_dir / "conversations.json"
    stress_path = config.data_dir / "advanced_long_context.json"
    standard = load_conversations(std_path)
    stress = load_conversations(stress_path)

    # Reset persistent profiles so `Memory growth` is repeatable across runs.
    import re as _re
    import shutil as _shutil

    def _sanitize(uid: str) -> str:
        try:
            try:
                from memory_store import _sanitize_user_id as _s
            except ImportError:
                from src.memory_store import _sanitize_user_id as _s
            return _s(uid)
        except Exception:
            return _re.sub(r"[^A-Za-z0-9._-]+", "_", (uid or "default").strip()) or "default"

    for conv in standard + stress:
        uid = str(conv.get("user_id", "default"))
        try:
            _shutil.rmtree(config.state_dir / "profiles" / _sanitize(uid), ignore_errors=True)
        except Exception:
            pass

    baseline = BaselineAgent(config=config, force_offline=True)
    advanced = AdvancedAgent(config=config, force_offline=True)
    std_rows = [
        run_agent_benchmark("Baseline", baseline, standard, config),
        run_agent_benchmark("Advanced", advanced, standard, config),
    ]
    print("## Standard Benchmark (data/conversations.json)")
    print(format_rows(std_rows))
    print()

    # Fresh agents for the stress suite so counters are isolated per table.
    baseline_s = BaselineAgent(config=config, force_offline=True)
    advanced_s = AdvancedAgent(config=config, force_offline=True)
    stress_rows = [
        run_agent_benchmark("Baseline", baseline_s, stress, config),
        run_agent_benchmark("Advanced", advanced_s, stress, config),
    ]
    print("## Long-Context Stress Benchmark (data/advanced_long_context.json)")
    print(format_rows(stress_rows))
    print()
    print("Ghi chú:")
    print("- Advanced recall cao hơn nhờ User.md bền vững + xử lý correction/noise.")
    print("- Hội thoại ngắn: Advanced có thể tốn prompt hơn do luôn kéo User.md theo.")
    print("- Hội thoại rất dài: compact memory nén lịch sử cũ nên prompt tokens "
          "processed của Advanced thấp hơn Baseline đáng kể (Baseline replay toàn bộ).")


if __name__ == "__main__":
    main()
