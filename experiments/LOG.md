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

### 2026-11-22 — Small Amharic model research and tokenizer comparison

- **Goal:** justify the student shortlist with public adaptation/evaluation evidence and measured tokenization costs before settling on Amharic Llama 1B.
- **Deliverables:** [selection report](../docs/RESEARCH_student_selection.md) and [results/provenance](student_tokenizer_research_2026-11-22.json). Public downloaded cards/configs/tokenizers remain under ignored `outputs/model_research/public/`.
- **Discovery commands:**
  ```bash
  curl -fsS --max-time 4 'https://huggingface.co/api/models?search=amharic&limit=80&sort=downloads&direction=-1&full=true' -o /tmp/amharic-model-search.json
  curl -fsS --max-time 4 'https://huggingface.co/api/models?search=amharic&pipeline_tag=text-generation&limit=200&sort=downloads&direction=-1' -o /tmp/amharic-generative-search.json
  curl -fsS --max-time 4 'https://huggingface.co/api/models?search=walia&limit=30' -o /tmp/amharic-walia-search.json
  curl -fsS --max-time 4 'https://huggingface.co/api/models?search=gemma&filter=am&limit=40&sort=downloads&direction=-1' -o /tmp/amharic-gemma-search.json
  ```
- **Measurement provenance:** public tokenizer revisions and SHA-256 hashes are retained in the results. Measurements used tokenizers 0.22.2 / Python 3.12.13. The one-off research script and its invocation instructions were removed at the user's request on 2026-11-24.
- **Input:** unchanged `data/sft/sft_train.jsonl`, 4,952 conversations / 9,904 messages, SHA-256 `94f38a1348df4c5be554f0ccec58cea0093ee77223f2f84f4c31ba4edc72763e`. No held-out eval text used. Native-speaker approval remains pending; this is a diagnostic, not a dataset freeze or training gate.
- **Settings:** deterministic content-only encoding with tokenizers 0.22.2 / Python 3.12.13; `add_special_tokens=False`, no truncation/padding/chat template or extra normalization. The tokenizer's native normalization remains active. Amharic-dominant subset: >=70% Ethiopic letters, 8,805 messages / 485,366 whitespace words. No sampling seed or provider inference.
- **Results:** 26 public checkpoint records checked; 16 tokenizers successfully measured. Amharic Llama 1B: **1.81** tokens/word and **929,695** total content tokens; Gemma 4 E2B: **3.21 / 1,612,464**; Qwen 2.5 1.5B and Qwen 3 1.7B: **6.45 / 3,197,799**; Qwen 3.5 2B, 4B and tested Amharic 4B: **7.09 / 3,508,841**. Llama uses 42.3% fewer complete-corpus content tokens than Gemma and 70.9% fewer than Qwen 2.5/3. Vocabulary comparison found 15,122 new Llama entries with zero changes to original token IDs. All measured shortlist tokenizers round-trip every input exactly; the separate 51.8M GEC/research tokenizer changes 40 messages through its normalization.
- **Additional findings:** Amharic Gemma 4 E2B/E4B adapters exist but lack reported task results. Amharic-trimmed Gemma 3 models reduce weight memory without a substantial measured tokenization advantage; the 1B/32K version lacks tokenizer files. Tiny Aya explicitly supports Amharic but carries noncommercial terms; its official tokenizer was gated, so its health derivative was measured separately. Sparse cards, task-specific models and misleading parameter names prevent ranking by model name/downloads alone. Gemma E2B has about 5.1B total parameters; E4B about 8B.
- **Cost/time:** $0; local CPU tokenization and public metadata/tokenizer downloads only. Intermittent DNS and transfer timeouts required resumable retries. No credentials sent, private data uploaded, model weights downloaded, paid inference, GPU run or model generation.
- **Decision:** retain the existing Llama/Gemma comparison. Llama 1B is justified as the cost-sensitive starting candidate; no answer-quality winner established. Optional smaller Gemma and research-only Tiny Aya leads are documented without changing the locked training protocol.
- **Validation:** script syntax, input SHA-256, 4,952 unique IDs, identical message/subset coverage across all 16 measurements, every measured tokenizer file hash, absence of downloaded model weights, and `git diff --check` all pass.
- **Open work:** native-speaker review/freeze, exact chat-template preflight, GPU smoke/resume, held-out quality evaluation and local speed/memory measurements.

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
  python3 -m students.data audit > /tmp/amharic-cleanup-student-audit.json
  git diff --check
  ```
- **Rebuild validation:** initial scratch filter accepted 4,845 rows; completed review merged 375 train + 31 eval, leaving 10 exclusions. Both accepted files reproduced **byte for byte**. Exact approved answers, exclusions, source metadata/prompts, previous accepted bytes, unique IDs and train/eval separation were verified. All 5,261 teacher IDs are accounted for. Protected dataset/review files retain their pre-cleanup hashes.
- **Prompt validation:** `PYTHONPATH=/home/yared/.cache/uv/archive-v0/DqZbTpunb28f7XhN /home/yared/workspace/cora_gnn/.venv/bin/python -B data_prep/validate.py` exited 0 with ALL PASS. Current output: `data/work/validation.json`. Existing Python 3.12.13 environment: pandas 3.0.5, numpy 2.5.2, scipy 1.18.0, cached datasketch 2.0.0.
- **Tokenizer validation:** both pinned tokenizers passed 5,251 accepted conversations and 300 eval generation prompts, with zero overlength/masking failures; maxima 1,065 Llama / 1,965 Gemma tokens, below 2,048. [Compact diagnostic](rejected_recovery_tokenizer_preflight_2026-11-21.json) retains model revisions, templates, input hashes and package versions. Its scratch input hashes match current accepted files. This is a diagnostic, with training_gate_passed=false.
- **Historical tokenizer command:** `HF_HUB_OFFLINE=1 HF_HOME=/tmp/amharic-bakeoff-hf /tmp/amharic-bakeoff-venv/bin/python -m students.preflight --dataset /tmp/rejected-recovery-preflight --unreviewed-diagnostic --out data/review/rejected_recovery_tokenizer_preflight_2026-11-21.json --summary experiments/rejected_recovery_tokenizer_preflight_2026-11-21.json`. The full report and temporary input copies were removed after retaining the compact summary.
- **Settings/cost:** direct assistant corrections, no new teacher generation or sampling seed. Raw teacher remains `google/gemma-4-26b-a4b-it` on DeepInfra. Native-review sampling used seed 42/category stratification. Existing cached packages/tokenizers only; $0, no paid API, GPU, commit or upload.
- **Next:** student audit reports only 100 pending native-speaker spot checks; no unresolved rejection dispositions or merge errors. Review the new corrected responses too, then freeze and repeat reviewed tokenizer preflight before GPU smoke/training.

### 2026-11-20 — Student bakeoff implementation and tokenizer diagnostic

- **Goal:** implement the post-review two-student QLoRA protocol with review/freeze gates, private Kaggle orchestration, resumable checkpoints, four-way evaluation, blinded native-speaker review, and Q4_K_M deployment checks.
- **Commands:**
  ```bash
  python3 -m students.data audit > /tmp/amharic-bakeoff-readiness.json
  HF_HUB_OFFLINE=1 HF_HOME=/tmp/amharic-bakeoff-hf /tmp/amharic-bakeoff-venv/bin/python -m students.preflight --dataset data/sft --unreviewed-diagnostic --out outputs/bakeoff/unreviewed-tokenizer-diagnostic.json --summary experiments/student_tokenizer_preflight_2026-11-20.json
  python3 -m unittest discover -s tests -v
  /tmp/amharic-bakeoff-venv/bin/python -m unittest discover -s tests -p test_students_torch.py -v
  python3 -m students.make_notebook
  python3 -m eval.build_constraints
  ```
- **Inputs/revisions:** current unreviewed `data/sft/sft_train.jsonl` (4,577), `eval_teacher.jsonl` (268), and all 300 pilot eval prompts. [Tokenizer diagnostic summary](student_tokenizer_preflight_2026-11-20.json) records input/template hashes and exact candidate model/tokenizer SHAs. Llama: `354e5403992da727615164b975594cde4529ab94`; Gemma: `4adf15748de9e1a0097b6274cee9c15836c41c37`.
- **Settings:** fixed seed 42, rank/alpha/dropout 16/32/0.05, NF4 double quantization, text-decoder adapters only, assistant/ending-token loss, length 2,048, lr 1e-4, two epochs, batch 1/accumulation 16, paged AdamW 8-bit, cosine/3% warmup, weight decay 0.01, clipping 1.0; category-stratified 10% validation. Native templates, fixed native Llama date header, Gemma thinking off. Greedy evaluation cap 1,000.
- **Environment:** tokenizer diagnostic uses Transformers 5.5.0, tokenizers 0.22.2 and Jinja2 3.1.6 on Python 3.12. CPU loss/architecture checks use torch 2.10.0+cpu and PEFT 0.18.1. GPU requirements remain candidates until both GPU training/resume/loading smoke tests pass; no working-GPU claim.
- **Cost/time:** $0; public tokenizer/package downloads and local CPU checks only. GPU cost, VRAM, training step time and runtime are not measured. Four CPU loss/real-architecture tests completed in about 66 s.
- **Results:** current readiness audit exits 2: all 307 review decisions are blank, 109 rejection dispositions and the approximately 100-answer native-speaker spot check remain pending. Both actual tokenizers pass all 4,845 current train/reference conversations and 300 generation prompts: zero masking failures or overlength IDs, maximum conversation length 923 Llama / 1,518 Gemma tokens. Thirteen dependency-free tests and four CPU library-contract tests pass; the real Llama decoder has exactly 11,272,192 trainable adapter parameters with original parameters frozen. Gemma text-only adapter/frozen-component checks pass on a meta-device architecture; its GPU count will be recorded in run metrics. Notebook code cells parse. Default prompt-bound format proxies cover all 15 format IDs, without inspecting generated answers.
- **Milestones:** `14cc382` dataset freeze/masking gates; `a030bbd` QLoRA/preflight/resume pins; `2ce21e7` evaluation, blind sheets, Kaggle notebook and deployment workflow. Commits remain on `teacher/pilot-run` while the existing teacher stage awaits commit/merge; creating `students/bakeoff` is deferred until that prerequisite is met. Existing teacher outputs/review edits are excluded from implementation commits.
- **Next:** finish review/triage/merge and native-speaker spot checks; freeze the dataset; rerun both-tokenizer preflight; run both Kaggle GPU smoke/resume tests before full Llama then Gemma training. No GPU smoke, adapter, full evaluation, human comparison or deployment outcome has been produced. Use the [run guide](../docs/RUN_student_bakeoff.md); every actual run records hashes/settings/revisions/runtime/cost in this log before final acceptance.

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
