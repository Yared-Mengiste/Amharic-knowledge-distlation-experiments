# Experiment log

One entry per run, newest first. Each entry should let someone rerun the experiment exactly.

## Template

### YYYY-MM-DD — <short name>
- **Goal:**
- **Command(s):**
- **Inputs:** (dataset file and version or tag, model and version, provider)
- **Settings:** (seed, temperature, caps, LoRA rank, learning rate, epochs, …)
- **Cost and time:**
- **Results:** (numbers, plus a link to the output files)
- **Decision and next step:**
- **Open questions:**

---

### 2026-11-24 — Separate completed answers from distillation implementation

- **Result:** answer-generation work is committed independently from `main`; existing distillation commits are preserved on `students/distillation`.
- **Dataset:** 4,952 training conversations and 299 references retained unchanged. All 300 eval prompts and audit inputs now live in `data/sft/`.
- **Cleanup:** removed intermediate prompt, teacher-response and review folders. Retained original preparation inputs/scripts; regenerated folders are ignored.
- **Validation:** replay the existing filter/merge commands from `data/sft/README.md` into scratch and compare accepted files byte for byte; verify unique IDs and held-out separation.
- **Approval state:** native-speaker spot checks remain pending; no training or paid generation was run.

### 2026-11-21 — Teacher answer review, recovery and artifact cleanup

- **Result:** reviewed all 307 queue rows (102 corrected, 197 retained, 8 excluded), then all 109 hard rejections (107 recovered, 2 excluded), at the user's request. Accepted totals: **4,952 train + 299 eval**. All 416 review decisions and 117 rejection/no-row dispositions are complete; 209 answers changed and 10 excluded overall.
- **Retained evidence:** completed review decisions, original rejection records, dispositions, native-speaker sheet and teacher provenance are now retained in `data/sft/`. Redundant change/exclusion ledgers and the review folder were removed on 2026-11-24.
- **Cleanup:** removed redundant manifests, audit/validation snapshots, queue/disposition backups, duplicate action ledgers, the large per-row tokenizer report, superseded reports and two one-off review scripts. Current corrected answers are reproducible with the existing filter and merger; the original teacher output file remains append-only.
- **Rebuild commands, run during cleanup:**
  ```bash
  python3 teacher/filter_teacher.py --teacher data/sft/teacher_outputs.jsonl --math data/sft/math_answers.csv --review_ids data/sft/teacher_answer_review_ids.txt --outdir /tmp/amharic-reviewed-rebuild
  python3 teacher/merge_review.py --review data/sft/review_queue.csv --teacher data/sft/teacher_outputs.jsonl --outdir /tmp/amharic-reviewed-rebuild
  cmp data/sft/sft_train.jsonl /tmp/amharic-reviewed-rebuild/sft_train.jsonl
  cmp data/sft/eval_teacher.jsonl /tmp/amharic-reviewed-rebuild/eval_teacher.jsonl
  git diff --check
  ```
- **Rebuild validation:** initial scratch filter accepted 4,845 rows; completed review merged 375 train + 31 eval, leaving 10 exclusions. Both accepted files reproduced **byte for byte**. Exact approved answers, exclusions, source metadata/prompts, previous accepted bytes, unique IDs and train/eval separation were verified. All 5,261 teacher IDs are accounted for. Protected dataset/review files retain their pre-cleanup hashes.
- **Prompt validation:** `PYTHONPATH=/home/yared/.cache/uv/archive-v0/DqZbTpunb28f7XhN /home/yared/workspace/cora_gnn/.venv/bin/python -B data_prep/validate.py` exited 0 with ALL PASS. Current output: `data/work/validation.json`. Existing Python 3.12.13 environment: pandas 3.0.5, numpy 2.5.2, scipy 1.18.0, cached datasketch 2.0.0.
- **Settings/cost:** direct assistant corrections, no new teacher generation or sampling seed. Raw teacher remains `google/gemma-4-26b-a4b-it` on DeepInfra. Native-review sampling used seed 42/category stratification. Existing cached packages/tokenizers only; $0, no paid API, GPU, commit or upload.
- **Next:** student audit reports only 100 pending native-speaker spot checks; no unresolved rejection dispositions or merge errors. Review the new corrected responses too, then freeze and repeat reviewed tokenizer preflight before GPU smoke/training.


### 2026-11-20 — Full pilot teacher filtering
- **Goal:** check the completed teacher outputs and separate accepted answers, human-review rows and rejected answers.
- **Command(s):**
  ```bash
  python3 teacher/filter_teacher.py --teacher data/sft/teacher_outputs.jsonl --math data/sft/math_answers.csv --review_ids data/sft/teacher_answer_review_ids.txt --outdir data/sft
  ```
- **Inputs:** [Full teacher outputs](../data/sft/teacher_outputs.jsonl), pilot math answers and safety/sensitive review IDs.
- **Settings:** existing default filters; minimum Ge'ez-letter share 0.70, or 0.45 for the categories with relaxed language checks.
- **Cost and time:** local filtering only; no API calls.
- **Results:** 5,261 unique outputs: 4,845 accepted, 307 queued for human review, 109 rejected. Accepted outputs are 4,577 training rows and 268 eval references. Review contains 279 train and 28 eval rows; rejected contains 105 train and 4 eval rows. All IDs are accounted for, the three groups are disjoint, and accepted train/eval IDs remain separate.
  - Flags include 58 calendar-label checks, 59 language failures (57 expected Amharic, 2 expected English), 29 math failures, 19 truncated answers, 7 repetition flags and 1 HTML flag. Flags can overlap.
  - All 170 safety answers are queued for human review. No review decisions have been entered.
  - [Filter report](../data/sft/filter_report.md), [review queue](../data/sft/review_queue.csv), [rejected answers](../data/sft/rejected.jsonl), [accepted training data](../data/sft/sft_train.jsonl), [eval references](../data/sft/eval_teacher.jsonl).
- **Decision and next step:** retain current filters; have native Amharic speakers review the queue, triage rejected answers, merge approved corrections with `teacher/merge_review.py`, then spot-check about 100 accepted training answers. Human review and rejected-answer triage remain pending.


### 2026-11-19 — Full OpenRouter pilot teacher generation
- **Goal:** generate teacher answers for all 4,961 train and 300 eval prompts.
- **Command(s):** user-reported full-run command:
  ```bash
  python3 teacher/gen_teacher.py \
    --train data/prompts/pilot_train.jsonl \
    --eval data/prompts/pilot_eval.jsonl \
    --out data/sft/teacher_outputs.jsonl \
    --openrouter --workers 8 \
    --timeout 180 --retries 2 \
    --provider DeepInfra --quantizations bf16,fp8 \
    --log_every 15 \
    2>> data/teacher/gen_errors.log
  ```
- **Inputs:** pilot prompt set: `data/prompts/pilot_train.jsonl` and `data/prompts/pilot_eval.jsonl`; teacher `google/gemma-4-26b-a4b-it`, provider DeepInfra.
- **Settings:** temperature 0.6, top_p 0.95, default teacher system prompt, reasoning disabled, existing per-category token caps. Every output records seed 42; OpenRouter requests do not send a sampling seed. The generator uses its fixed ordering seed of 13.
- **Cost and time:** $0.629123 recorded in output usage ($0.629 reported by the CLI), 1.17 h reported by the user. Saved timestamps range from 16:25:21 to 17:35:28 on 2026-11-19.
- **Results:** [Teacher outputs](../data/sft/teacher_outputs.jsonl): 5,261 valid unique rows, all expected IDs present, no missing IDs or invalid JSON lines. All outputs record DeepInfra and the selected teacher model. Usage: 1,220,124 output tokens and 3,061,161 prompt tokens. Finish reasons: 5,242 `stop`, 19 `length`.
- **Decision and next step:** generation coverage is complete; filter and review outputs before training.

### 2026-11-19 — System-prompt fix + train smoke v2 (2 × 40 prompts, DeepInfra)
- **Goal:** check the tightened system prompt (no bracketed English glosses; ዓ.ም never on foreign-event years; translations output only the translation).
- **Command(s):** same as the train smoke with `--out data/teacher/smoke_train_v2_same.jsonl` (the same 40 prompts), then again with `--out data/teacher/smoke_train_v2_next.jsonl` pre-filled with the first 40 rows (so the next 40 prompts ran).
- **Inputs / settings:** as before; system prompt changed in `teacher/gen_teacher.py`.
- **Cost and time:** $0.009, about 1.5 min.
- **Results:**
  - Glosses on the same 40 prompts: 46 in 18 answers before, 12 in 8 answers after. Fresh 40 prompts: 2 in 2 answers. Most of what remains is technical terms.
  - Translation am→en: fixed. All 5 answers are English only, with no Amharic preface and no restated source. Average answer length dropped from 250 to about 200–230 tokens.
  - Calendar: still "ከ1841 እስከ 1846 ዓ.ም" for the British prime-minister question (the filter flags it for review).
  - New: 1 of 80 answers looped ("passed passed …") until the 900 cap (ethiopia_culture, Irreecha). The filter rejects it as looping/truncated.
  - Teacher factual errors the filter can't catch: names Magellan instead of Balboa for the 1513 Pacific sighting; says ልዑል አለማየሁ was taken "በ1932 ዓ.ም" (it was 1860 ዓ.ም / እ.ኤ.አ. 1868); dates a news story to 2024 instead of 2020.
- **Decision and next step:** keep the new system prompt. The full run needs the project lead's go-ahead. Factual accuracy will need the native-speaker spot check; regenerate filter-rejected rows once after the run.
- **Open questions:** a few train prompts write Gregorian years with ዓ.ም themselves (e.g. pilot_8c3a08f3701998f66b12f112, a passage about ልዑል አለማየሁ), which may push the teacher toward the same mistake.

### 2026-11-19 — OpenRouter train smoke test (40 prompts, DeepInfra pinned)
- **Goal:** cover every category on train prompts with one pinned provider.
- **Command(s):** `python teacher/gen_teacher.py --train data/prompts/pilot_train.jsonl --eval data/prompts/pilot_eval.jsonl --out data/teacher/smoke_train.jsonl --openrouter --only train --limit 40 --workers 8 --provider DeepInfra --quantizations bf16,fp8 --log_every 15`, then `filter_teacher.py` on it (output in a scratch dir).
- **Inputs:** first 40 train rows in run order (all 17 categories); `google/gemma-4-26b-a4b-it` on DeepInfra.
- **Settings:** seed 42, temperature 0.6, top_p 0.95, default system prompt, thinking off.
- **Cost and time:** $0.005, about 45 s. That puts the full 5,261-prompt run at roughly $0.65 and 1–1.5 h.
- **Results:** 40/40 `stop`, all on DeepInfra, no thinking, no HTML. Filter: 36 accepted, 3 sent to review (2 safety, 1 calendar), 1 rejected.
  - English glosses in brackets in 18/40 answers, including basic words (e.g. መደመር (Addition), ማባዛት (Multiplication)).
  - Gregorian years labelled ዓ.ም: "ከ1841 እስከ 1846 ዓ.ም" for a British prime-minister question. The calendar flag caught it.
  - Amharic→English translation: in 1 of 2 answers the teacher first rewrote the passage in Amharic, then translated it (428/450 tokens). The filter rejected it as wrong language. The other answer opened with an Amharic preface before the English.
- **Decision and next step:** proposed a system-prompt fix for glosses, Gregorian ዓ.ም and English-only translation output, then another train smoke on different prompts before the full run. Waiting on the project lead.

### 2026-11-19 — OpenRouter smoke test (20 prompts)
- **Goal:** check the OpenRouter path end to end before the full pilot run.
- **Command(s):** `python teacher/gen_teacher.py --train data/prompts/pilot_train.jsonl --eval data/prompts/pilot_eval.jsonl --out data/teacher/smoke.jsonl --openrouter --workers 8 --limit 20 --log_every 10`
- **Inputs:** first 20 rows in run order (all eval `native_chat`); `google/gemma-4-26b-a4b-it`; no provider pinned.
- **Settings:** seed 42, temperature 0.6, top_p 0.95, default system prompt, thinking off.
- **Cost and time:** $0.002, about 40 s.
- **Results:** 20/20 `stop`, none cut off (longest 673/900 tokens), no thinking, no HTML, no ዓ.ም on Gregorian years. `cost_usd` and `provider` recorded. Requests went to 6 different providers. English glosses in brackets in 13/20 answers (26 in total). Some are reasonable (names, exercise names, IAEA), but many follow ordinary Amharic words, against the system prompt. One answer had Ge'ez share 0.67, below the 0.70 filter threshold, because it named English websites.
- **Decision and next step:** pin one provider before the full run. Do a train-only smoke across categories, because this one covered only eval `native_chat`.
- **Open questions:** should the system prompt be tightened further on glosses, or should the filter flag them? Should the Ge'ez threshold allow answers that name English sites or products?

### 2026-11-19 — Repo restructure; pilot prompt set rebuilt
- **Goal:** move the pilot-fix pipeline into `data_prep/` and its outputs into `data/prompts/`, and check the set still reproduces.
- **Command(s):** `bash data_prep/run_all.sh` (Python from `.venv`)
- **Inputs:** `data/raw/pilot_train_5000.csv`, `data/raw/pilot_eval_300.csv`, `data_prep/manual/`
- **Results:** all validation checks pass. `pilot_train.jsonl` (4,961), `pilot_eval.jsonl` (300), `removed.csv`, `changed.csv`, `review_sample.csv` and `teacher_answer_review_ids.txt` are byte-identical to the earlier build; only path strings in `REPORT.md` changed. The OpenRouter teacher scripts ran end to end (generate → filter) against a local mock server.
- **Decision and next step:** this build of `data/prompts/` is the pilot v1 prompt set. Next: OpenRouter smoke test.

### 2026-11-11 — Local teacher smoke test (llama.cpp Vulkan, laptop)
- **Goal:** check the teacher's speed and answer quality on local hardware.
- **Inputs:** 20 eval prompts (all `native_chat`); `unsloth/gemma-4-26B-A4B-it-GGUF:UD-Q4_K_M`; 2 parallel slots.
- **Results:** 15.8 tok/s total. 8 of 20 answers were cut off at the 600-token cap. Gregorian years were labelled ዓ.ም. One wrong multiple-choice answer, one made-up answer to a question whose passage was missing, and one stray HTML tag.
- **Decision:** tightened the system prompt, raised the caps, and added HTML and calendar checks to the filter. The full run moves to OpenRouter.
