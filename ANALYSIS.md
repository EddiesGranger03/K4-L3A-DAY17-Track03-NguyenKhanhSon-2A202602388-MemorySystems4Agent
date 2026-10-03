# Phân tích kết quả — Day 17: Memory Systems for AI Agent

## 1. Cách chạy lại (tái lập được, offline, không cần API key)

```powershell
# từ thư mục gốc repo (ngay cạnh README.md)
python src/benchmark.py
python -m pytest src/test_agents.py -v
```

Lưu ý trên Windows:

- `import` trong `src/` là import phẳng (`from config import ...`), nên phải chạy
  `python src/benchmark.py` từ thư mục gốc, không dùng `python -m src.benchmark`
  (sẽ báo `ModuleNotFoundError: No module named 'config'`).
- Lệnh `pytest` trực tiếp có thể không có trên PATH; dùng `python -m pytest`.
- Benchmark chạy ở chế độ offline tất định (`force_offline=True`); `.env` chỉ cần
  cho chế độ live. `state/` đã nằm trong `.gitignore` nên có thể xóa để chạy lại
  từ đầu (`benchmark.py` cũng tự reset profile của các user trong dataset trước
  mỗi lần chạy để cột `Memory growth` ổn định).
- Biến môi trường `load_config()` đọc: `LLM_PROVIDER`, `LLM_MODEL`,
  `LLM_TEMPERATURE`, `OPENAI_API_KEY` / `GEMINI_API_KEY` / `GOOGLE_API_KEY` /
  `ANTHROPIC_API_KEY` / `OPENROUTER_API_KEY` / `CUSTOM_API_KEY` / `LLM_API_KEY`,
  `CUSTOM_BASE_URL` / `OLLAMA_BASE_URL` / `OPENROUTER_BASE_URL` / `LLM_BASE_URL`,
  `COMPACT_THRESHOLD_TOKENS` (mặc định `1200`), `COMPACT_KEEP_MESSAGES`
  (mặc định `6`). Mặc định provider `openai` / model `gpt-4o-mini`.
- Chế độ live đã đấu dây thật (`create_agent` + `InMemorySaver` + tool
  `read_profile`/`write_profile` + prompt động + middleware tóm tắt khi có
  `langmem`): có API key trong `.env` và `force_offline=False` thì agent dùng
  model thật; không có key thì tự rơi về offline. Đã kiểm chứng dựng graph
  thành công với `langchain`/`langgraph`/`langchain-openai` (`CompiledStateGraph`
  + checkpointer cho cả Baseline và Advanced).

## 2. Kết quả benchmark (đã cài `tabulate`)

### Standard Benchmark (`data/conversations.json`, 10 hội thoại × ~10 lượt, user `dungct`)

| Agent    | Agent tokens only | Prompt tokens processed | Cross-session recall | Response quality | Memory growth (bytes) | Compactions |
|----------|------------------:|------------------------:|---------------------:|-----------------:|----------------------:|------------:|
| Baseline | 2854 | 19924 | 0 | 0.6 | 0 | 0 |
| Advanced | 5377 | 35568 | 1 | 0.982 | 300 | 0 |

### Long-Context Stress Benchmark (`data/advanced_long_context.json`, 1 hội thoại 16 lượt rất dài, user `dungct_stress`)

| Agent    | Agent tokens only | Prompt tokens processed | Cross-session recall | Response quality | Memory growth (bytes) | Compactions |
|----------|------------------:|------------------------:|---------------------:|-----------------:|----------------------:|------------:|
| Baseline | 488 | 23814 | 0 | 0.6 | 0 | 0 |
| Advanced | 1250 | 17142 | 1 | 0.983 | 248 | 15 |

### Test

`python -m pytest src/test_agents.py -v`: **7 passed**
(`test_user_markdown_read_write_edit`, `test_compact_trigger`,
`test_cross_session_recall`, `test_compact_reduces_prompt_load_on_long_thread`,
`test_structured_entity_extraction_confidence`, `test_memory_decay`,
`test_negation_and_word_boundary_regressions`).

## 3. Trả lời 4 câu hỏi của Guide.md (Bước 8)

Mỗi luận điểm dưới đây theo cấu trúc: số liệu → cơ chế trong code → giới hạn.

**a) Vì sao Advanced recall tốt hơn Baseline?**
Số liệu: `1.0` vs `0.0` ở cả hai bảng.
Cơ chế (`src/agent_advanced.py: _reply_offline`, `src/memory_store.py:
extract_structured_entities` + `extract_profile_updates`): mỗi lượt trích fact
ổn định có kèm confidence rồi `upsert_fact()` vào `User.md`
(`state/profiles/<user>/User.md`); khi recall hỏi ở **thread mới**,
`_offline_response()` đọc lại file và trả lời theo khóa (tên / nơi ở / nghề /
style / đồ uống / món ăn / thú cưng / mối quan tâm). Baseline
(`src/agent_baseline.py`) khóa session theo `thread_id` nên thread mới trắng
hoàn toàn — `Memory growth = 0` chính là bằng chứng nó không ghi gì.
Giới hạn: recall chỉ tốt khi extraction đúng; fact không khớp pattern sẽ không
bao giờ vào file (đánh đổi precision lấy coverage, xem mục bonus).

**b) Vì sao Advanced có thể tốn hơn ở hội thoại ngắn?**
Số liệu: Standard `Agent tokens only` 5377 vs 2854; `Prompt tokens processed`
35568 vs 19924.
Cơ chế: mỗi lượt Advanced cộng đủ 3 thành phần
(`_estimate_prompt_context_tokens`: `User.md` + summary + recent messages) và
câu trả lời có gắn fact nên dài hơn; trong khi 10 lượt ngắn chưa vượt ngưỡng
compact (`Compactions = 0` ở bảng Standard) nên chưa có gì bù lại chi phí.
Giới hạn: đây là chi phí cố định của persistent memory — thread càng ngắn,
tỷ lệ overhead càng lộ.

**c) Vì sao compact có lợi thế ở hội thoại dài?**
Số liệu: Stress `Prompt tokens processed` 17142 vs 23814 (giảm ~28%),
`Compactions` 15 vs 0.
Cơ chế (`src/memory_store.py: CompactMemoryManager.append`): vượt
`threshold_tokens` thì message cũ được dồn vào `summary`, chỉ giữ
`keep_messages` tin gần nhất nguyên văn; Baseline replay toàn bộ lịch sử mỗi
lượt nên cost tăng O(n²). Đúng như Rubric 75–90 yêu cầu: compact tối ưu
**`Prompt tokens processed`**, không phải `Agent tokens only` (cột này của
Advanced vẫn cao hơn vì câu trả lời mang fact + ví dụ + trade-off).
Kiểm chứng bổ sung đã làm: đặt ngưỡng rất cao → compact không chạy →
prompt của Advanced tiến gần Baseline; trả ngưỡng về `1200` thì lợi thế quay lại.
Giới hạn: summary heuristic có thể làm mất abstraction nếu ngưỡng quá thấp.

**d) File memory tăng trưởng ra sao và rủi ro gì?**
Số liệu: `Memory growth` 300 / 248 bytes; `Compactions` 0 ở Standard, 15 ở Stress.
Cơ chế: mỗi fact mới là một dòng `- Nhãn: giá trị` trong `User.md`
(`UserProfileStore.upsert_fact`), kèm metadata decay trong sidecar
`User.meta.json`. Rủi ro quan sát được: (1) file phình theo thời gian nếu trích
quá tham — đã có decay + `prune_decayed()` để dọn fact yếu; (2) fact sai từ lượt
nhiễu ("Hà Nội đi họp", "product manager đùa") nếu lọt vào file sẽ ở đó vĩnh
viễn — đã có confidence threshold + negation guard; (3) correction (Đà Nẵng ↔
Huế, backend → MLOps) nếu chỉ nối thêm sẽ cho hai đáp án mâu thuẫn — code ghi đè
"newest wins" (`upsert_fact` + `edit_text`).

## 4. Bonus (mốc 90–100): 4 hướng + live mode

1. **Confidence threshold** (`CONFIDENCE_THRESHOLD = 0.6`,
   `extract_structured_entities` → `extract_profile_updates`): chỉ ghi khi
   pattern mạnh (tên phải khớp `tên là ...`, style cần động từ
   `thích/muốn/ưu tiên`, lượt chỉ-hỏi trả `[]`). Giải quyết: file phình + nhiễu.
   Cải thiện: precision của recall. Rủi ro mới: fact diễn đạt khác pattern bị bỏ
   sót (mất thay vì sai).
2. **Conflict handling**: `upsert_fact()` ghi đè theo khóa; mention mới nhất
   thắng khi có cue `đính chính/không còn/chuyển sang`; kèm phủ định
   (`không còn ở X`, `đừng nói X nữa`, `là thông tin cũ` → confidence 0.2).
   Giải quyết: câu "nơi ở / nghề hiện tại" sau correction. Rủi ro mới: cue
   correction nhận nhầm có thể ghi đè fact đúng — production cần xác nhận.
3. **Entity extraction có cấu trúc** (`Entity{kind, value, confidence, detail,
   start}`): mọi ứng viên đều mang điểm và lý do, test đọc được cái gì bị loại
   và vì sao. Đã bắt 2 bug thật bằng test: `AI` khớp chuỗi con trong `hai`
   (fix bằng regex word-boundary `\bAI\b`), và `backend` bị hồi sinh từ câu
   `đừng nói backend nữa` (fix bằng negation window). Rủi ro mới: regex tiếng
   Việt vẫn giòn trước cách diễn đạt mới — cần eval set riêng.
4. **Memory decay** (`User.meta.json`: `{hits, last_turn}` mỗi fact,
   `strength = hits * 0.5**(age/half_life)`, `touch()` mỗi lượt,
   `decayed_facts()`/`prune_decayed()` kèm guardrail không bao giờ xóa fact
   cuối cùng). Giải quyết: `User.md` phình vô hạn. Cải thiện: token/memory cost
   dài hạn. Rủi ro mới: fact đúng nhưng lâu không nhắc có thể bị dọn — phải
   tune `half_life` và `min_strength` theo sản phẩm.
5. **Live mode** (mở rộng): `build_chat_model()` cho cả 6 provider,
   `create_agent` + `InMemorySaver` (short-term theo thread), tool
   `read_profile`/`write_profile` (persistent), `_dynamic_prompt()` chèn profile
   + summary, `SummarizationMiddleware` khi có `langmem` (compact của
   `CompactMemoryManager` vẫn chạy trước mỗi gọi model). Thiếu dep/key → tự về
   offline nên benchmark/test không vỡ.

## 5. Đối chiếu với Rubric

- 0–60: đủ Baseline (nhớ trong thread), Advanced (`User.md` + compact),
  benchmark tiếng Việt, cấu trúc repo rõ — đạt.
- 60–75: cùng input cho cả hai agent, đủ test `User.md` / compact trigger /
  cross-session recall, bảng đủ 6 cột — đạt (7 test).
- 75–90: đủ Standard + Stress, stress 16 lượt làm lộ cost Baseline, phân tích
  tách rõ `Prompt tokens processed` vs `Agent tokens only` — đạt.
- 90–100: 4 bonus + live mode, mỗi bonus đủ 3 mặt (giải quyết gì / cải thiện
  gì / rủi ro gì) — đạt.
