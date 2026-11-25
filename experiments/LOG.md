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

### 2026-11-25 — Clean up the completed Llama workflow for commit

- **Scope:** retain the reviewed pilot, Kaggle compatibility fixes and completed-run evidence. Remove obsolete notebook repair instructions and duplicate result detail from the project guide. At the user's request, leave `data_prep/resolve_spot_checks.py` uncommitted; remove its bundle dependency and helper-specific test so a clean checkout works without it. Keep audit exclusion tests, the reviewed dataset/removal ledger and the helper's historical preparation command. Downloaded run artifacts are preserved and remain ignored by Git.
- **Validation commands:**
  ```bash
  python3 -m unittest discover -s tests -v
  python3 -m students.data audit --dataset data/sft/pilot-v1
  python3 -m students.make_notebook
  python3 -m students.bundle --dataset outputs/bakeoff/frozen-v1 --out /tmp/amharic-llama-commit-bundle
  git diff --cached --check -- . ':(exclude)data/sft/pilot-v1/*.csv'
  ```
- **Results:** 39 tests ran: 36 passed, three optional Transformers/PEFT checks skipped. The same suite passes in the fresh `/tmp/amharic-llama-commit-bundle/repo` without the excluded helper (`python3 -m unittest discover -s tests -v` from that directory). Reviewed-data audit passes with 4,947 train / 299 reference / 300 eval rows, bundle file hashes match, notebook code cells parse and non-CSV whitespace checks pass. Newly tracked review CSVs retain their original CRLF and embedded answer whitespace to preserve frozen hashes and review evidence. No GPU rerun, dataset-row change or upload was needed for cleanup.

### 2026-11-25 — First complete Kaggle Llama run received and checked

- **Input/evidence:** user-supplied private Kaggle outputs at `outputs/llama_first_run_all_2` (plural `outputs`). The [downloaded run log](../outputs/llama_first_run_all_2/experiments/LOG.md) retains exact original commands; the [comparison](../outputs/llama_first_run_all_2/comparison.md) and companion JSON retain category metrics and generation provenance. Training was recorded on 2026-11-24; evaluation on 2026-11-25. No new GPU training or generation was performed locally.
- **Model/data:** `rasyosef/Llama-3.2-1B-Amharic-Instruct`, model/tokenizer revision `354e5403992da727615164b975594cde4529ab94`; frozen manifest SHA-256 `f87586bdeb22ad20a30a60006cf31e8aac6a1ceb59e6c766811c9d1ea40519f1`, split SHA-256 `479d179b8e055015549002ff564911ec1c4327e733f87f1592810d064bcce945`. The local frozen-v1 snapshot matches: 4,947 conversations, split into 4,452 training and 495 validation IDs; 299 references and all 300 held-out prompts.
- **Original full-training command:**
  ```bash
  /usr/bin/python3 -m students.train --dataset /kaggle/input/datasets/yaredmengiste/amharic-bakeoff-private/frozen --model llama --lock /kaggle/working/bakeoff/model_lock.json --preflight /kaggle/working/bakeoff/preflight.json --out /kaggle/working/bakeoff/train-llama --precision fp16
  ```
- **Protocol/environment:** unchanged seed 42, rank 16 / alpha 32 / dropout 0.05, learning rate 1e-4, two epochs, length 2,048, microbatch 1 / accumulation 16, NF4 double quantization, assistant-only loss. One Tesla T4, FP16; torch 2.10.0, transformers 5.5.0, peft 0.18.1, bitsandbytes 0.49.2, accelerate 1.12.0, tokenizers 0.22.2. Exact packages and settings are retained in run metrics/identities. Reviewed preflight passed; smoke resumed from step 2 to 4 with adapter loading verified, and the promoted lock authorizes only Llama.
- **Training result:** 558/558 optimizer steps, epoch 2.0, 558 finite-gradient steps, final adapter reload verified. Best token-weighted validation loss **2.7121407** at checkpoint 558; validation declined from 2.90745 at step 100. Recorded wall time **3,817.97 s (63.63 min)**; peak allocated/reserved GPU memory **4.10 / 7.62 GiB**. Kaggle records cost $0; this is training runtime, excluding separate smoke and evaluation processes.
- **Evaluation:** original and adapted each cover all 300 unique frozen eval IDs using matching templates, FP16/NF4 and greedy decoding with a 1,000-token cap. Reference coverage is 299/300. Assistant-token loss **3.60568 → 2.72628**, perplexity **36.8066 → 15.2759** on the same 37,453 reference tokens. Language-script proxy **222/282 (78.7%) → 266/285 (93.3%)**; truncation **50/300 → 38/300**; repetition flags **98/300 (32.7%) → 178/300 (59.3%)**. Numeric-presence math proxy remains **1/24** and format-shape proxy **0/15** for both. These are automated proxies, not semantic correctness or fluency scores.
- **Local verification command(s):**
  ```bash
  python3 -m students.verify_artifacts --directory outputs/llama_first_run_all_2
  python3 -m eval.report --dataset outputs/bakeoff/frozen-v1 --outputs outputs/llama_first_run_all_2/eval --human outputs/llama_first_run_all_2/human --out /tmp/llama-first-run-all-2-comparison.json
  ```
  The strict artifact verifier exits 1 because 16 indexed `repo/**/__pycache__/*.cpython-313.pyc` files are absent. Independent SHA-256 verification matches all **182 retained files out of 198 indexed**, with zero changed files; no other files are missing. The original manifest is preserved. Recomputed comparison JSON equals the saved JSON exactly, validating frozen hashes, paired settings, output/reference coverage and blind-sheet mappings.
- **Decision/remaining work:** training and automated evaluation are complete. Human review is entirely unfilled: 100 general answer ratings, 50 general preferences, 40 safety answer ratings and 20 safety preferences (210 rows). `comparison_ready=false`; no GGUF/deployment evidence exists in this download. Complete the existing blinded native-speaker review and Q4_K_M deployment comparison before selecting a winner, adding Gemma or scaling. The repetition increase and weak translation/math/format proxies need quality review; do not infer success from perplexity alone. Single-seed and small-human-sample limits still apply.

### 2026-11-24 — Remove obsolete Transformers checkpoint argument

- **Trigger:** the user's fresh FP16 smoke attempt loaded all weights and passed the longest-example forward/backward/finite-gradient check, then failed at TrainingArguments construction: `unexpected keyword argument 'save_safetensors'`. No optimizer smoke steps or checkpoint/resume result was supplied.
- **Exact API check:** downloaded the official Transformers 5.5.0 wheel and verified its SHA-256 against saved PyPI metadata. All configured FP16/BF16 full/smoke argument names were compared against its TrainingArguments dataclass fields: only `save_safetensors` was unsupported. Its Trainer save path already uses safetensors or model/PEFT `save_pretrained` defaults. Optimizer/scheduler checkpoint filenames remain `optimizer.pt`/`scheduler.pt`.
- **Change:** remove only the obsolete save flag from `training_argument_values`; SETTINGS and the intended checkpoint format stay unchanged. Construct TrainingArguments before model weights and run metadata so later API failures stop early. Add an optional installed-Transformers signature contract test covering full/smoke FP16/BF16 dictionaries.
- **Existing-session repair:** `outputs/bakeoff/kaggle-training-arguments-repair-cell.py` patches the writable Kaggle config, verifies every remaining argument against the installed TrainingArguments signature, preserves the failed attempt under a timestamped name if checkpoint-free, and leaves checkpoints/same-precision memory failures for inspection. The original smoke/resume cell and canonical lock/report paths then remain valid.
- **Validation:** exact-wheel source comparison passes after removing the flag. Temporary-directory repair execution with the exact source-derived signature verifies the one-key edit, failed-attempt preservation, repeatability and checkpoint refusal. `python3 -m unittest discover -s tests -v`: 40 tests ran, 37 passed and three skipped (two optional PEFT architecture checks; the installed-Transformers signature check because Transformers is absent locally). No model weights or GPU dependencies installed locally for these checks. The pinned-source check supplies the exact-version API evidence.
- **Cost/state:** $0; public wheel/source reads and local checks only. Package pins, training SETTINGS and frozen data unchanged. The user must rerun repaired smoke/resume in Kaggle; no completed GPU smoke or training result inferred.

### 2026-11-24 — Preserve an interrupted smoke attempt before recovery

- **Trigger:** the user's FP16 smoke-start command failed with `Output exists; use --resume or a new run directory` at `WORK/smoke-llama`. The notebook attempted a fresh two-step run because `checkpoint-2` was absent; an earlier attempt left the output directory nonempty. The underlying earlier failure and the directory contents have not been supplied.
- **Recovery:** keep the trainer's overwrite/resume guard. The explicit local `outputs/bakeoff/kaggle-smoke-recovery-cell.py` checks for any checkpoints and same-precision memory-failure evidence before renaming a metadata-only failed attempt to `smoke-llama-failed-<timestamp>`. The original smoke cell can then start at its canonical output path, preserving lock-promotion paths and the prior evidence. Existing checkpoints or a prior FP16 OOM stop recovery for inspection rather than silently retrying.
- **Validation:** separate temporary-directory execution confirmed preservation of the original metadata bytes, checkpoint-directory refusal and same-precision OOM refusal. No core training implementation, package pins, SETTINGS or frozen data changed. No new GPU run/result has been produced by the assistant; the user must execute the recovery and original smoke cells in Kaggle.
- **Cost:** $0 for local recovery preparation/checks.

### 2026-11-24 — Require native BF16 support for automatic precision

- **Trigger:** the user reran Kaggle dependencies/preflight successfully through installation and GPU eligibility, then printed `Tesla T4`, 15,636,037,632 reported VRAM bytes and incorrectly selected `bf16`. HF Hub emitted an unauthenticated-download warning; no completed preflight report or smoke result was supplied.
- **Cause/fix:** PyTorch's `is_bf16_supported()` includes emulation by default. Automatic BF16 selection now requires NVIDIA compute capability >=8 and `is_bf16_supported(including_emulation=False)`. T4/P100 select FP16; explicit BF16 is rejected without native support. Existing Kaggle notebooks can set `precision, gpu = gpu_precision('fp16')` after tokenization preflight completes, before smoke; their subprocess commands already pass that explicit precision through smoke/training/evaluation.
- **Validation:** `python3 -m unittest discover -s tests -p test_students_modeling.py -v`: all four mocked-CUDA tests pass. New emulation cases verify FP16 fallback and BF16 refusal on both a T4 and a capability-8 GPU whose runtime reports only emulated BF16. Native A100 BF16 remains accepted; other memory/single-GPU checks remain enforced.
- **State/cost:** $0 for local checks. Package versions, SETTINGS and frozen data unchanged. HF warnings alone do not establish preflight success; require completed `preflight.json` with `passed=true` and FP16 before the user starts the GPU smoke/resume gate. No smoke/training result has been produced or inferred.

### 2026-11-24 — Correct Kaggle T4 eligibility and preinstalled package conflicts

- **Trigger:** Kaggle installed all six candidate pins on Python 3.13.16, replacing Torch 2.11.0+cu130 with 2.10.0 and related CUDA libraries. Its preinstalled torchvision 0.26.0+cu130 and cuda-python 13.0.3 reported conflicts. Preflight then stopped before tokenizer execution because the T4 exposed 14.56 GiB and the eligibility check required 15 GiB.
- **GPU correction:** recognize a nominal 16 GB T4 reporting at least 14 GiB after reserved memory. Other GPU models retain the 15 GiB threshold; small T4 allocations and multiple visible GPUs still fail. FP16 selection on T4 and all smoke/actual-memory gates are unchanged. Actual reported VRAM continues to be recorded rather than relabelled as 16 GiB.
- **Environment correction:** remove unused Kaggle torchvision, torchaudio and cuda-python packages before installing the existing candidates; keep Torch-required NVIDIA libraries and cuda-bindings. Detect a loaded Torch version that differs from its installed metadata and request a kernel restart. Setup copies the allowlisted source into WORK/repo so notebook imports and training subprocesses share writable code, while the frozen data remains in the read-only input. The bundle now includes the preparation helper needed by its copied data tests.
- **Existing-session repair:** local `outputs/bakeoff/kaggle-t4-repair-cell.py` can be pasted after setup, before dependencies/preflight. It copies the old attached repo, applies the T4-specific correction, updates REPO/CANDIDATE and module imports, removes the unused conflicts and detects stale Torch. The original input is preserved; training subprocesses load the repaired code.
- **Validation:** `python3 -m students.make_notebook`; `python3 -m unittest discover -s tests -v`: 38 ran, 36 passed and two optional PEFT architecture checks skipped. New mocked-CUDA checks cover 14.56 GiB T4 acceptance and recorded VRAM, rejection of 12/13.99 GiB T4 and smaller other GPUs, P100 acceptance, single-GPU enforcement and FP16/BF16 selection. Separate repair-cell execution with mocked pip verifies source preservation, writable-code patching and repeatability; all notebook code cells parse. GPU execution and installation conflict resolution have not been run locally.
- **Updated bundle commands:** `python3 -m students.bundle --dataset outputs/bakeoff/frozen-v1 --out outputs/bakeoff/kaggle-bundle-v2`; from that directory, `python3 -m zipfile -c ../kaggle-bundle-v2.zip repo frozen bundle_manifest.json`. The prior uploaded bundle is preserved. Frozen data, SETTINGS and all six package pins remain unchanged.
- **Cost/state:** $0 for local fixes/tests. No model training, generation, dataset rewrite or upload performed by the assistant. Rerun dependencies/preflight in Kaggle, then run the GPU smoke/resume/loading gate before full training.

### 2026-11-24 — Accept the actual Kaggle Python runtime

- **Trigger:** the imported setup cell refused Kaggle's actual Python 3.13.16 interpreter. The original Python-3.12-only guard, and the subsequent 3.11/3.12 guard, were narrower than the pinned package metadata required.
- **Compatibility evidence:** official PyPI release JSON for torch 2.10.0, transformers 5.5.0, peft 0.18.1, bitsandbytes 0.49.2, accelerate 1.12.0 and jinja2 3.1.6 accepts Python 3.11–3.13. Torch publishes Linux CPython 3.13 wheels; the other five pins publish universal Python wheels. Commands followed `curl -L --fail --silent --show-error --max-time 25 https://pypi.org/pypi/<package>/<version>/json -o /tmp/amharic-kaggle-<package>-<version>.json`; failed accelerate/jinja2 transfers were retried with `--retry 2 --max-time 20`. This verifies package declarations/wheel availability, not GPU execution.
- **Change:** the generated notebook now accepts Python 3.11, 3.12 and 3.13, prints the actual interpreter version and records it in the final artifact index. Setup selects the first T4 with `CUDA_VISIBLE_DEVICES=0` before CUDA imports. Training still requires one visible GPU, dependency installation, tokenizer preflight and GPU smoke/resume/loading. All package versions, SETTINGS and frozen data remain unchanged. Existing imported Kaggle cells must be edited or the corrected local notebook reimported; the previously uploaded bundle is preserved.
- **Validation:** `python3 -m students.make_notebook`; every generated code cell parses; the extracted interpreter guard accepts 3.11/3.12/3.13 and rejects 3.10/3.14. `load_frozen('outputs/bakeoff/frozen-v1')` verifies unchanged snapshot hashes. `python3 -m unittest discover -s tests -v` ran on local Python 3.11.15: 33 passed, two optional PEFT architecture checks skipped. No Python 3.13 package installation or Kaggle GPU validation has been performed locally; the notebook's next gates supply that evidence.
- **Cost/state:** $0; public package metadata reads and local code checks only. No training, dataset changes or uploads.

### 2026-11-24 — Prepare private Kaggle upload bundle

- **Purpose:** prepare the frozen Llama-first dataset and notebook for the user's Kaggle training setup.
- **Command:** `python3 -m students.bundle --dataset outputs/bakeoff/frozen-v1 --out outputs/bakeoff/kaggle-bundle-v1`. From `outputs/bakeoff/kaggle-bundle-v1`: `python3 -m zipfile -c ../kaggle-bundle-v1.zip repo frozen bundle_manifest.json`.
- **Artifact:** local `outputs/bakeoff/kaggle-bundle-v1.zip`, 5.09 MiB. Archive roots are `repo/`, `frozen/` and `bundle_manifest.json`. All 57 indexed file hashes match both the bundle and ZIP contents; ZIP integrity, notebook code syntax and frozen dataset/split loading pass. No `.env` or `.git` is included. Exact copied code is hashed even though the workspace changes are not committed.
- **Protocol:** Llama only; 4,452 training IDs, 495 validation IDs, two epochs and 558 expected optimizer steps. The notebook runs tokenizer preflight, interrupted GPU smoke/resume/loading and lock promotion before full training, then original/adapted evaluation and output hashing.
- **State:** no upload, Kaggle session or GPU execution has occurred. Package/GPU compatibility remains to be verified by the smoke gate. Cost $0.

### 2026-11-24 — Native-speaker review resolved and pilot v1 frozen

- **Goal:** audit the completed `data/sft/spot_check.csv` review until it passes, then freeze the dataset at the user's request.
- **Review:** Yared filled all 100 rows: 95 `yes`, 5 `no`, with no written rejection reasons. The original audit exited 2 with only those five unresolved training spot checks. The five answers are excluded on the recorded native-review decisions; no replacement answers or human approvals were invented.
- **Preparation:** `data_prep/resolve_spot_checks.py` writes a new version atomically, preserves all retained conversation bytes and all original source files, copies the human sheet byte for byte, and documents the five removals in `removed.csv`, queue `no` decisions and excluded dispositions. `preparation.json` records the nine source hashes. The audit now resolves a `no` only when its matching training teacher ID is absent from accepted data, its queue decision is `no`, its excluded disposition has notes, its reviewed answer hash matches and its reviewer is named. It still requires at least 100 completed spot checks; it also verifies sheet answer text against its hash. `--dataset` selects versioned inputs, with individual flags still available.
- **Commands:**
  ```bash
  python3 -m students.data audit
  python3 -m unittest discover -s tests -p test_students_data.py -v
  python3 -m data_prep.resolve_spot_checks --source data/sft --out data/sft/pilot-v1
  python3 -m students.data audit --dataset data/sft/pilot-v1 > outputs/bakeoff/pilot-v1-audit.json
  python3 -m unittest discover -s tests -v
  python3 -m students.data freeze --dataset data/sft/pilot-v1 --out outputs/bakeoff/frozen-v1
  python3 -m students.data audit --dataset outputs/bakeoff/frozen-v1 > outputs/bakeoff/frozen-v1-audit.json
  python3 -c "from students.data import load_frozen; load_frozen('outputs/bakeoff/frozen-v1'); print('Frozen hashes and split verified')"
  git diff --check
  ```
- **Results:** both versioned and frozen audits exit 0 with `ready=true` and no errors. [Reviewed source](../data/sft/pilot-v1/README.md): **4,947 training conversations, 299 references, 300 eval prompts**. All 421 queue decisions (406 yes / 15 no) and 122 dispositions (107 corrected / 15 excluded) are complete. The five new exclusions span native_chat, news_classification, code_switch, translated_instruction and math; their IDs and reviewed hashes are in [removed.csv](../data/sft/pilot-v1/removed.csv). Reference coverage remains 299/300.
- **Freeze:** local [snapshot manifest](../outputs/bakeoff/frozen-v1/manifest.json), nine required data/review inputs plus two preparation/removal records; all 11 file hashes and the split hash verify through `load_frozen`. Seed 42, category-stratified 10% validation: **4,452 training IDs + 495 validation IDs**. `SETTINGS` and model/package candidates are unchanged; no model or provider inference was run. Original teacher: `google/gemma-4-26b-a4b-it` / DeepInfra.
- **Hashes:** training `eaba5bfc3f47923c2536ba95ffb36dc000866396ac0720e47cf3a4f959ecf7cd`; split `479d179b8e055015549002ff564911ec1c4327e733f87f1592810d064bcce945`; manifest `f87586bdeb22ad20a30a60006cf31e8aac6a1ceb59e6c766811c9d1ea40519f1`.
- **Validation:** 35 tests ran, 33 passed and 2 optional PEFT architecture checks skipped because PEFT is unavailable. New fixtures verify that rejected answers cannot remain accepted, undocumented exclusions/missing reviewers/stale hashes fail, 99 spot checks fail, version preparation preserves source and retained bytes, and existing versions cannot be overwritten. Separate local byte/hash comparisons confirm the original nine inputs are unchanged by preparation, the eight inputs other than the user's sheet still match Git HEAD, all 4,947 retained conversations are byte-identical and every frozen file matches the versioned source. Snapshots remain under ignored `outputs/`.
- **Cost and next step:** $0; local CPU preparation/audit/tests only. Reviewed tokenizer preflight and Llama GPU smoke/resume/loading are next; no GPU training, model evaluation, deployment or upload performed.

### 2026-11-24 — Llama-only first student experiment (code change only)

- **Why:** the mentor requested a small-model run first to assess response-distillation benefit. Review Llama results before choosing a larger second student, probably Gemma E2B.
- **Branch:** `students/llama-first`, created from `students/distillation`.
- **Changes:** add a Llama-only candidate copied from the retained two-model pins; allow promotion with only Llama smoke evidence; record `verified_models` and enforce model-specific authorization in training, evaluation, export and deployment. Keep all supplied-report protocol/package/hash/precision and resume/loading checks. Preflight compares only locked models. The generated notebook defaults to Llama and has an optional Gemma switch. Evaluation, blind review, reports and acceptance derive complete original/adapted pairs from the models present. Reports show categories, approved-reference coverage and single-seed/small-human-sample limits. Acceptance requires Gemma run/deploy arguments together only for reports including Gemma, preserving cross-model consistency checks when applicable.
- **Commands:** before and after: `python3 -m unittest discover -s tests -v`; `python3 -m students.data audit`. Regeneration: `python3 -m students.make_notebook`; `python3 -m students.make_notebook --include-gemma`; `python3 -m students.make_notebook` (restore the default). Also `git diff --check` and read-only byte/AST comparisons against `students/distillation`.
- **Code validation:** baseline: 25 tests ran, 23 passed and 2 PEFT architecture checks skipped. After: 33 tests ran, 31 passed and the same 2 checks skipped because PEFT is unavailable. New CPU-only fixtures cover lock promotion/refusal, all supplied smoke checks, Llama-scoped preflight, blind sampling and order, a two-configuration report with 299/300 references, and acceptance without Gemma while rejecting changed splits/review hashes or omitted Gemma deliverables. Both notebook variants parse; default regeneration is byte-for-byte reproducible, with no executed cells.
- **Integrity/approval state:** audit exits 2 before and after, with only the same 100 pending native-speaker spot checks; counts remain 4,952 train, 299 references and 300 eval. All 13 tracked protected data files match the base branch byte for byte. `SETTINGS`, the existing two-model candidate and candidate package pins are unchanged. No filter, freeze or data rewrite was run.
- **Limits and next step:** this entry records implementation and unit checks only. No GPU smoke, model training, generated answers, model evaluation, deployment benchmark or associated results were produced. Candidate pins remain unverified. Complete the native-speaker spot check before freezing/running Llama. Any later Gemma stage needs its own preflight, matching GPU smoke/resume/loading evidence, a new lock/output directory, training, both output/reference sets, new combined blind review and deployment quality evidence; preserve stage one.

### 2026-11-24 — Reduce student loss memory and unify protocol settings

- **Goal:** remove retained FP32 loss buffers and prevent recorded settings from disagreeing with training, evaluation or completion checks.
- **Changes:** assistant CE selects only supervised positions and checkpoints 64-position chunks with non-reentrant recomputation. Trainer arguments, quantization, chat-template options, split/blind-review seeds, generation limits and optimizer-step accounting now use `students/config.py`. Validation batches use the configured evaluation batch size. The regenerated Kaggle notebook shares the completion calculation.
- **Protocol:** the existing rank, learning rate, two epochs, sequence cap and other first-experiment values are unchanged. Loss chunk size is now recorded in the protocol; older frozen manifests/checkpoints require a new freeze and fresh smoke evidence. Original dataset bytes and original distillation commits are preserved.
- **Commands:** `python3 -m students.make_notebook`; `python3 -m unittest discover -s tests -v`; `git diff --check`; `git diff HEAD -- data/sft`.
- **Validation:** 23 tests pass; two optional PEFT architecture checks are skipped because PEFT is unavailable. Loss values and gradients match full cross-entropy in FP32/FP16/BF16, including unequal answer lengths and partial chunks. Autograd saved-tensor checks confirm no retained FP32 vocabulary-sized loss buffers after forward. Masked nonfinite logits do not affect loss or gradients. Changed-protocol tests verify split seeds, Trainer arguments, partial-batch/fractional-epoch completion, generation caps and blind-review mappings. Source/notebook cells parse, and all 13 committed SFT files are byte-for-byte unchanged.
- **Environment and limits:** local PyTorch 2.15.0+cpu; no CUDA, Transformers or PEFT installed in this test environment. Candidate GPU package pins remain unchanged. The model still produces full logits; both pinned-model GPU smoke/resume runs remain required to measure real VRAM and training behavior.
- **Cost:** $0; no model generation, training, GPU rental or upload.

### 2026-11-24 — Separate completed answers from distillation implementation

- **Result:** answer-generation work is committed independently from `main`; existing distillation commits are preserved on `students/distillation`.
- **Dataset:** 4,952 training conversations and 299 references retained unchanged. All 300 eval prompts and audit inputs now live in `data/sft/`.
- **Cleanup:** removed intermediate prompt, teacher-response and review folders. Retained original preparation inputs/scripts; regenerated folders are ignored.
- **Validation:** filter/merge replay reproduces both accepted files byte for byte; original SFT hashes, unique IDs and held-out separation pass. The existing suite passes 15 tests, with 2 optional PEFT architecture tests skipped because PEFT is unavailable. Default-path student audit resolves all nine inputs and reports only the 100 pending native-speaker approvals. Format constraints reproduce byte for byte.
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
