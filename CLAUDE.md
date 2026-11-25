# CLAUDE.md: Amharic distillation project

This file tells Claude Code what this project is, what has been decided, where things live and how to work in this repo. Read it fully before doing anything. Update the **Status** and **Decisions log** sections whenever something changes.

## 1. What the project is

The aim is a small Amharic chatbot model (1B–4B parameters) that is usable for general chat in Amharic and can run cheaply, eventually on-device or on modest servers. Later uses include domain assistants such as banking, telecom and government services.

The method is response-level knowledge distillation:

1. A large teacher model answers a carefully built set of Amharic prompts. The teacher is Gemma 4 26B A4B-it, a mixture-of-experts model with 26B parameters in total and about 3.8B active per token. It's Apache 2.0, so its outputs can be used to train other models.
2. Bad answers are filtered out automatically, and the risky ones are reviewed by a person.
3. Small student models are fine-tuned (SFT) on the prompt and teacher-answer pairs.
4. The students are evaluated against held-out prompts and the teacher, and the best one is scaled up.

The project lead is Bereket Legesse, an ML engineer at iCog Labs. A team member prepares the prompts, and native Amharic speakers do the human review.

## 2. Status (as of 2026-11-25)

| Stage | State |
|---|---|
| 0. Review of the original 50k prompt set (`prompts.csv`, "v1") | Done. It was mostly task templates with little real chat, so about 20k of it was kept as a source. |
| 1. Pilot prompt set: 5,000 train + 300 eval | Done. Built by the team member, then fixed by Claude Code following `docs/SPEC_pilot_fix.md`. All validation checks pass. Reproducible with `bash data_prep/run_all.sh`. |
| 2. Native-speaker review of 150 prompts (`data/prompts/review_sample.csv`) | Pending. |
| 3. Teacher generation for 5,261 prompts | Done on 2026-11-19. All prompt IDs have outputs from `google/gemma-4-26b-a4b-it` on DeepInfra; 1,220,124 output tokens, $0.629123, 1.17 h. See `experiments/LOG.md`. |
| 4. Filter, review and merge the teacher answers | Native-speaker review completed by Yared on 2026-11-24: 95 spot checks accepted, 5 rejected. New `data/sft/pilot-v1` excludes those five, retains all other conversation bytes and documents 421 queue decisions and 122 dispositions. Accepted: 4,947 train + 299 eval; 15 total exclusions. Audit passes; local snapshot `outputs/bakeoff/frozen-v1` is frozen with hashes and seed-42 split. Original data/review evidence is preserved; see `data/sft/pilot-v1/README.md`. |
| 5. Student bake-off on reviewed pilot answers | First Llama Kaggle run complete: reviewed preflight, GPU smoke/resume/loading, promoted lock, two-epoch training (558 steps) and 300 original + 300 adapted eval outputs. Local artifacts: `outputs/llama_first_run_all_2`. Reference perplexity 36.81 → 15.28; language proxy 78.7% → 93.3%; repetition flags 32.7% → 59.3%. Blind native-speaker review and Q4_K_M deployment remain pending; no quality winner selected. See experiments/LOG.md. |
| Student selection research | Public cards/metadata checked for 26 checkpoints; 16 tokenizers measured on the current 4,952 training conversations. Amharic Llama 1B averages 1.81 tokens per word versus Gemma 4 E2B 3.21 and tested Qwen 2.5/3 6.45, Qwen 3.5 7.09. Run Llama first to assess distillation; defer the larger comparator, probably Gemma E2B, until those results are reviewed; answer-quality and deployment results remain pending. See docs/RESEARCH_student_selection.md. |
| 6. Scale-up to 30k–100k prompts | Not started (section 9). |

### Open items

- [ ] **Repo visibility:** the GitHub repo is currently public. Make it private until the source licences are checked (section 10, Licences).
- [x] Add the OpenRouter version of the teacher scripts, `docs/RUN_teacher.md` and `.env.example`. (Done 2026-11-19; checked end to end against a local mock server, not the real API.)
- [ ] `math_answers.csv`: the row "ልጄ የተወለደው በ2010 ዓ.ም ነው። በ2018 ዓ.ም መስከረም ላይ ስንት ዓመቱ ይሆናል?" says 8, but the answer is 7 or 8 depending on the birth month. Set it to missing, or add a birth month to the prompt.
- [x] Bring the Claude Code fix-run scripts and the `data/manual/` folder into `data_prep/`, so the pilot set can be reproduced. (Done 2026-11-19: `data_prep/` and `data_prep/manual/`; a rerun reproduces `data/prompts/` byte for byte.)
- [ ] Optional cleanup: remove 1 near-duplicate driving multiple-choice question, 2 prompts that ask the assistant about itself (e.g. "አንተ መጥፎ ልማድህ ምንድን ነው?"), and the awkward greetings that were added mechanically.

## 3. Key decisions (and why)

| Decision | Reason |
|---|---|
| Response-level distillation first, not logit-level | Simpler, and works with any student tokenizer. Logit-level distillation is an option later if the student shares Gemma's tokenizer (E2B/E4B). |
| Teacher: Gemma 4 26B A4B-it, thinking off | Its tokenizer is the most efficient measured for Amharic (2.47× the cost of English), it's cheap to run, and Apache 2.0. Thinking would waste tokens. |
| Teacher runs on OpenRouter (about $0.5–1 and 1–2 h for the pilot) | The local laptop manages only about 16 tok/s with Vulkan, so about 26–36 h. RunPod would cost $3–6 and needs more setup. Pin one provider so the teacher stays consistent. |
| General chat first; assistant roles and multi-turn later | Chosen by the project lead. |
| Every training prompt leads to an Amharic answer, except Amharic→English translation | English prompts carry an Amharic-answer request. No English-in, English-out examples for now. |
| A teacher system prompt is used but not saved in the training data | It steers length, formatting, calendar labels, accuracy and safety. The student learns the behaviour without the prompt. |
| Pilot size: 5k train | Enough to compare students. Final size is decided from a scaling curve (10k → 30k → 100k), not guessed. |
| Eval set kept strictly separate | No shared situation lines, no near-duplicates (char 5-gram Jaccard ≥ 0.8), and no shared reading passages. |
| No language removal by weight editing | It doesn't add Amharic ability. The weight-editing work the idea came from fixed a specific bug and didn't add capability. Vocabulary trimming of the student is an optional size experiment only. |
| Prompt generators: Gemma (Apache 2.0) or Claude Code acting directly | Check a provider's terms before using its outputs as training data. Gemini and Claude outputs may be restricted for training competing models. |

## 4. Repo layout

```
.
├── CLAUDE.md                 # this file
├── README.md                 # short human-facing overview
├── .env.example              # copy to .env; holds OPENROUTER_API_KEY (never commit .env)
├── docs/
│   ├── RUN_teacher.md        # step-by-step teacher run (OpenRouter)
│   ├── RUN_teacher_local.md  # older local run guide (Ollama / llama.cpp)
│   └── SPEC_pilot_fix.md     # spec used to fix the pilot prompt set (paths in it predate this layout)
├── data_prep/                # scripts that build and validate prompt sets
│   ├── run_all.sh            # rebuilds data/prompts/ from data/raw/
│   ├── validate.py           # schema, duplicates, leakage, language mix, PII, safety IDs
│   └── manual/               # hand-made judgements and prompts used by the scripts
├── teacher/
│   ├── gen_teacher.py        # teacher generation (Ollama, llama.cpp/vLLM, OpenRouter); resumable
│   ├── filter_teacher.py     # automatic checks → sft / review / rejected
│   └── merge_review.py       # merges human-approved rows back in
├── students/                 # student fine-tuning code, configs and private Kaggle workflow
├── eval/                     # evaluation scripts and blind human-review tooling
├── experiments/
│   └── LOG.md                # one entry per run: what, config, result, decision
└── data/                     # see data/README.md for what is committed
    ├── raw/                  # pilot inputs from the team member (read-only)
    ├── work/                 # intermediate files written by data_prep/
    └── sft/                  # completed training/eval data and required audit inputs
```

The original completed pilot is retained in `data/sft/`: 4,952 training
conversations, 299 teacher evaluation references and all 300 evaluation prompts.
The native-reviewed version is `data/sft/pilot-v1`: 4,947 training conversations
after five documented exclusions, with the original review sheet preserved.
Its audited local snapshot is `outputs/bakeoff/frozen-v1`. Intermediate
`data/prompts/`, `data/teacher/` and `data/review/` folders have been removed.

Run every command from the repo root. Rebuild `data/prompts/` before a new teacher run.

## 5. Data formats

Prompt row (`data/prompts/pilot_*.jsonl`):

```json
{"id": "pilot_<24 hex>", "category": "everyday", "subcategory": "...", "language_mode": "am",
 "messages": [{"role": "user", "content": "..."}], "source": "v2_generated",
 "origin_id": null, "needs_human_review": true, "split": "train"}
```

| Field | Values |
|---|---|
| `category` | native_chat, english_instruction, translated_instruction, summarization, translation_en_am, translation_am_en, news_classification, knowledge_qa, everyday, ethiopia_culture, writing, math, format, code_switch, latin_amharic, creative, safety |
| `language_mode` | am, en, mixed, latin_am |

Pilot composition:

- Train is 4,961 rows: 82.6% am, 9.1% en, 5.3% mixed, 3.0% latin_am. About 51% of the prompts are newly written.
- Eval is 300 rows.

Teacher output row (`data/sft/teacher_outputs.jsonl`): the prompt row's fields plus `response`, `finish_reason`, `output_tokens`, `prompt_tokens`, `max_tokens`, `seconds`, `teacher_model`, `gen_params`, and on OpenRouter also `cost_usd` and `provider`. The file is append-only. If an ID appears twice, the last line wins.

SFT row (`data/sft/sft_train.jsonl`):

```json
{"id": "...", "category": "...", "language_mode": "...",
 "messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

Other files:

- `math_answers.csv`: `id, prompt, type, status (ok|missing|contradiction), computed_answer, note`.
- `teacher_answer_review_ids.txt`: 224 IDs (170 safety, 54 sensitive). The teacher's answers to these must be read by a person before training.

## 6. Pipeline: teacher stage

Full commands are in `docs/RUN_teacher.md`. In short:

1. **Smoke test:** 20 prompts with `--openrouter --limit 20`. A person checks that the answers are in Amharic, few are cut off, ዓ.ም and እ.ኤ.አ. are used correctly, and there's no thinking text or HTML.
2. **Full run:** `--openrouter --workers 16 --retries 6`. It's resumable: re-running only fills the missing IDs.
3. **Filter:** `teacher/filter_teacher.py`.
   - Hard checks (reject): empty, truncated, thinking leak, HTML tags, looping text, wrong language, wrong math.
   - Language thresholds: at least 70% Ge'ez letters, or 45% for code-switch, Latin-script and English-instruction prompts. Amharic→English translations must be English.
   - Soft flags (sent to review): safety/sensitive IDs, possible refusal, invented phone numbers or links, unclear math, Gregorian-looking years written with ዓ.ም.
4. **Human review:** fill in `data/sft/review_queue.csv`, then run `teacher/merge_review.py`.
5. **Native-speaker spot check:** about 100 random answers from `sft_train.jsonl`. Repeated error patterns lead to a system-prompt fix and regeneration of the affected IDs.

Teacher settings:

- temperature 0.6, top_p 0.95, thinking off.
- Per-category output caps: 900 by default; writing and creative 1,000; math 800; format and news_classification 250; summarization 400; translation 450–500.

Known teacher failure modes from the first local smoke test:

- Long, heavily formatted answers. 8 of 20 hit the old 600-token cap.
- Gregorian years labelled with ዓ.ም instead of እ.ኤ.አ.
- Wrong multiple-choice answers. Example: the Roman-emperor months question; it picked ግንቦት/ሰኔ instead of ሐምሌ/ነሐሴ.
- Made-up answers when a question needs a reading passage that wasn't given. Example: "ቄስ ብሌክ ልዑል አለማየሁ…".
- Stray HTML tags.
- English glosses in brackets after ordinary words.

The system prompt was tightened to address these. Check again after the real run.

## 7. Environments

| Where | What |
|---|---|
| OpenRouter | Model `google/gemma-4-26b-a4b-it`. Key in `.env` as `OPENROUTER_API_KEY`. Don't use the `:free` variant. |
| Local laptop | ASUS ZenBook 14: Ryzen AI 7 (Radeon 860M iGPU, XDNA 2 NPU), 32 GB RAM, Arch Linux. Details below. |
| NPU | Not usable for the 26B teacher. It's relevant later for running a small distilled student (E2B-class) on-device. |
| Training compute | Private Kaggle notebook, one NVIDIA GPU with nominal >=16 GB VRAM. Llama first; Gemma deferred until result review. Each selected model must pass unchanged-protocol smoke/resume tests. |

Local laptop details:

- llama.cpp built with Vulkan at `~/llama.cpp`.
- The GPU memory limit was raised to about 20 GB with `/etc/modprobe.d/ttm-gpu-memory.conf` (`options ttm pages_limit=5242880 page_pool_size=5242880`).
- Teacher speed is about 16 tok/s total with 2 parallel requests.
- Ollama must be stopped before running llama.cpp, because two copies of the model don't fit in memory.

## 8. Student bakeoff (stage 5)

The first comparison is `rasyosef/Llama-3.2-1B-Amharic-Instruct` before and
after response-level QLoRA distillation. Review its results before selecting a
larger second student, probably `google/gemma-4-E2B-it`; Gemma support and
candidate pins remain available. The active branch is `students/llama-first`,
created from `students/distillation`.
See [the protocol](docs/PLAN_student_bakeoff.md) and [run guide](docs/RUN_student_bakeoff.md).

- Review/merge every queued answer, document rejection dispositions, and have a
  native speaker spot-check approximately 100 accepted training answers before freezing.
- Every included model shares frozen training IDs and a seed-42, category-stratified 10%
  validation split. Rank 16, alpha 32, dropout 0.05, two epochs, lr 1e-4, NF4,
  sequence length 2,048, microbatch 1 and accumulation 16 are fixed.
- Use native templates with assistant-only supervision including the ending token;
  disable Gemma thinking and freeze original/embedding/vision/audio parameters.
- Use private Kaggle compute after preflight and GPU smoke/resume/loading
  checks for each included model; candidate package/checkpoint pins are not a working-environment claim.
- Generate all 300 eval IDs for original/adapted Llama first, then any added
  model, with greedy
  decoding and 1,000 output tokens. Approved references only; report coverage.
- No paid judge calls. Blind native-speaker scores/preferences cover 50 stratified
  prompts plus separate review of all 20 safety prompts. Report categories and
  uncertainty, the single seed and small human sample; compare perplexity only
  within each tokenizer pair. Current approved-reference coverage is 299/300.
- Export each included full-run adapter to Q4_K_M and compare local RAM, speed and answer
  quality on identical deployment prompts. Keep weights, checkpoints and sheets private.

Adding Gemma later requires tokenizer preflight, matching GPU smoke/resume/loading
evidence, a newly promoted lock, full training, original/adapted generation and
reference losses, new combined blind sheets/report, and Q4_K_M deployment review.
Repeat Llama smoke if needed to match packages, precision and frozen hashes.
`verified_models` limits lock authorization to the models actually checked;
Llama evidence cannot authorize Gemma. Preserve stage-one artifacts and never
overwrite locks or blind mappings. Adding a model leaves `SETTINGS` and the frozen
data/split unchanged.

Scripts and the Kaggle notebook are implemented. Earlier unreviewed-tokenizer
checks on the 4,952 corrected training conversations passed with no overlength or masking failures
(maximum 1,065 Llama tokens, 1,965 Gemma tokens; length limit 2,048). Queue decisions,
hard-rejection dispositions and merges are complete. Yared completed the
100-answer native-speaker sheet in `data/sft/spot_check.csv`: 95 accepted, 5 rejected.
The five rejected answers are excluded in `data/sft/pilot-v1`; its audit and local
freeze passed. The user completed FP16 Tesla T4 preflight, smoke/resume/loading,
training and original/adapted evaluation on 2026-11-24–25. Evidence is at
`outputs/llama_first_run_all_2`; [experiments/LOG.md](experiments/LOG.md) records
results and integrity checks. The comparison reproduces against the local frozen
dataset. All four blind-review CSVs are unfilled, and Q4_K_M deployment remains
pending. Complete those reviews before adding Gemma or scaling.

## 9. Later stages

- **Scaling curve:** train the chosen student on 10k, 30k and 100k pairs from the same pool. Scale further only if 30k→100k still clearly helps; otherwise spend the effort on data quality.
- **More prompts:** generate from the category, persona and style matrix in `docs/SPEC_pilot_fix.md`, adding assistant roles with system prompts (bank, telecom, clinic, university, shop, government desk) and multi-turn conversations. Write follow-up turns after the teacher has answered.
- **Optional experiments:**
  - Logit-level distillation for Gemma-tokenizer students.
  - Vocabulary trimming of the student (keep Ge'ez, Latin, digits, punctuation and code tokens) to cut size.
  - Rejection sampling: 2–4 teacher samples per prompt, keep the best by judge score.
- **Deployment check:** quantize the chosen student (GGUF Q4/Q8) and measure speed and quality on CPU, iGPU and, for E2B-class models, the NPU.

## 10. How to work in this repo (rules for Claude Code)

### Data

- Never edit files in `data/prompts/` by hand or in place. Changes to a prompt set are made by a script in `data_prep/` that writes a new version (e.g. `pilot_train_v2.jsonl`) and logs what changed and why.
- Never drop rows silently. Every removal goes into a `removed.csv` with a reason, and every change into a `changed.csv`.
- Every new or rewritten Amharic prompt gets `needs_human_review: true`.
- The eval set is frozen. Don't train on it, and don't tune prompts or system prompts by looking at individual eval answers. Use train samples for that.
- Keep IDs stable across versions.

### Amharic text you write

- Write natural, everyday Amharic as real users type it.
- No made-up facts, statistics or real officials' names.
- No real private individuals, phone numbers or addresses.
- Gregorian years use እ.ኤ.አ.; ዓ.ም is for Ethiopian-calendar dates.

### Money and external calls

- Ask the project lead before any paid API run larger than a smoke test, or any GPU rental.
- Always run a smoke test first.
- Never print, log or commit API keys. Keys live only in `.env`.

### Reproducibility

- Every dataset, model or eval result must come from a script in the repo with its exact command written in `experiments/LOG.md`, including seed, model versions and the provider used.

### Git

- Work on a branch per task (`data/pilot-v2`, `teacher/pilot-run`, `students/bakeoff`) and open a PR into `main`. Keep commits small with clear messages.
- Tag dataset versions when they're frozen, e.g. `data-pilot-v1`.
- Don't commit model weights, checkpoints or files over about 50 MB. Use Hugging Face Hub (private repos) or Git LFS for those, and record the link in `experiments/LOG.md`.

### Licences

- The pilot mixes sources: Aya, Dolly, wiki-split, NQ-Open, Tatoeba, an Amharic news corpus, AddisGPT and amharic-qa. Their licences haven't all been checked.
- Keep the repo and any dataset uploads private until each source's licence is confirmed.

### Validation

- After changing data, run the validator (`data_prep/validate.py`, or the whole `bash data_prep/run_all.sh`). It checks the schema, duplicates and near-duplicates, train/eval leakage, template repetition, language mix, length, personal data, and the safety-ID list.

### When unsure

- If something is unclear or a decision isn't covered here, stop and ask rather than guess. Write the open question into `experiments/LOG.md`.

## 11. Decisions log

Add a line whenever a decision is made or changed.

| Date | Decision |
|---|---|
| 2026-11-25 | Remove obsolete Kaggle repair instructions and bundle/test dependencies on the local spot-check preparation helper before committing the reviewed dataset, Kaggle fixes and run status. Leave data_prep/resolve_spot_checks.py uncommitted at the user's request; retain its historical command and preparation evidence. |
| 2026-11-25 | Record the user's completed Kaggle Llama training/evaluation from outputs/llama_first_run_all_2. Preserve downloaded artifacts and the original manifest; all retained hashes and the recomputed comparison match, with only 16 bytecode caches missing. Lower reference loss and better language-script compliance are accompanied by more repetition flags; keep quality selection pending until the existing blind native-speaker sheets and deployment checks are complete. |
| 2026-11-10 | Response-level distillation from Gemma 4 26B into a 1–4B student. Prompt target is 50k–150k eventually, starting with a 5k pilot. |
| 2026-11-10 | Rejected removing other languages by weight editing. Vocabulary trimming kept as an optional size experiment. |
| 2026-11-10 | v1 prompt review: keep about 20k, write about 30k new chat-style prompts. General chat comes before assistant roles. |
| 2026-11-11 | Pilot fix done by Claude Code from the spec. All English prompts lead to Amharic answers. |
| 2026-11-11 | Tightened teacher system prompt (length, formatting, ዓ.ም/እ.ኤ.አ., accuracy, safety) and raised the output caps. |
| 2026-11-19 | Teacher generation moves to OpenRouter instead of the local laptop. The project is managed in a GitHub repo, with this file as the guide. |
| 2026-11-19 | OpenRouter teacher pipeline added (`teacher/`, `docs/RUN_teacher.md`, `.env.example`). |
| 2026-11-19 | Teacher system prompt tightened again: no bracketed English glosses, ዓ.ም never on foreign-event years, translations output only the translation. Pin DeepInfra (bf16/fp8) for teacher runs. |
| 2026-11-19 | Repo restructured to the layout in section 4: pilot-fix scripts in `data_prep/`, final prompt sets in `data/prompts/`, teacher scripts in `teacher/`. |
| 2026-11-20 | Full teacher outputs are complete and automatically filtered. Keep the current filter thresholds; human review and rejected-answer triage are required before finalizing SFT data. |
| 2026-11-20 | Implemented the reviewed-response Llama/Gemma QLoRA workflow with fixed rank 16, private Kaggle execution, assistant-only loss, shared frozen IDs, four-way greedy evaluation and native-speaker comparisons. No paid judge calls; training waits for review and GPU smoke/resume evidence. |
| 2026-11-21 | At the user's request, corrected 102 review-queue answers, retained 197 and excluded 8 with reasons, then ran the existing merge command. Preserve raw teacher/frozen prompt files and prior accepted bytes. Assistant decisions are distinct from native-speaker approval; no training freeze yet. |
| 2026-11-21 | Reviewed all 109 hard rejections at the user's request; corrected/recovered 107 and excluded 2 with reasons. Merged 103 train and 4 eval rows using the existing merger. Retain raw rejection records as provenance, document two matcher/language exceptions, and preserve earlier accepted answers. Both tokenizer diagnostics pass; native-speaker review remains pending. |
| 2026-11-21 | At the user's request, removed redundant review snapshots, manifests, reports and one-off scripts. Keep the completed queue, change/exclusion/disposition ledgers, native-review sheet, one overview and compact tokenizer evidence. Existing filter/merge commands rebuild corrected accepted files in scratch. |
| 2026-11-22 | Researched small Amharic students and measured actual tokenizer costs. Keep the existing Llama/Gemma bakeoff: Amharic Llama 1B has a substantial content-token advantage, while final quality is unproven. Identified Amharic Gemma adapters, vocabulary-trimmed Gemma candidates and Tiny Aya as optional leads, with their evaluation, artifact and licensing limits documented. No training-protocol change or model winner declared. |
| 2026-11-24 | Separate answer-generation and distillation branches. Keep the completed dataset and required audit inputs in data/sft; remove intermediate prompt, response and review folders. |
| 2026-11-24 | Run Llama 1B alone first to measure response-distillation benefit at the mentor’s request. Review results before selecting a larger second student, probably Gemma E2B. Keep both candidate pins; scope lock verification, evaluation and acceptance to included models without changing SETTINGS or frozen data. Gemma requires its own preflight, matching GPU smoke/resume/loading, full training, original/adapted outputs, combined blind review and deployment evidence in new artifact directories. No GPU/model results produced by this code change. |
| 2026-11-24 | Yared completed all 100 native-speaker spot checks (95 yes, 5 no) and requested audit/freeze. Preserve the review decisions and original dataset; exclude the five rejected answers in a scripted new version. Audit requires documented exclusions with matching reviewed hashes and at least 100 completed spot checks. Freeze 4,947 training conversations, 299 references and 300 eval prompts at outputs/bakeoff/frozen-v1; keep SETTINGS unchanged and rerun reviewed tokenizer preflight before GPU smoke/training. |
| 2026-11-24 | Kaggle reports Python 3.13.16 and offers T4 x2. Accept Python 3.11–3.13 after verifying exact pin metadata and wheel availability; expose only the first T4 before CUDA imports. Keep package pins, SETTINGS, frozen data and all install/preflight/GPU smoke gates. The notebook prints and records the actual interpreter version; metadata compatibility is not a GPU verification result. |
| 2026-11-24 | Kaggle's nominal 16 GB T4 reports 14.56 GiB. Allow T4 allocations reporting >=14 GiB while retaining the >=15 GiB threshold for other GPUs and requiring exactly one visible GPU. Remove unused preinstalled torchvision/torchaudio/cuda-python conflicts, preserve the pinned dependencies and use a writable code copy for consistent subprocess repairs. Actual VRAM and smoke evidence remain required; no training result inferred. |
