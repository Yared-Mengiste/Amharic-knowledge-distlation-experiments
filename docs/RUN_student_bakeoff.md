# Run the reviewed-response student bakeoff

Use [the frozen protocol](PLAN_student_bakeoff.md). This repository supplies the
workflow; human approval and GPU smoke results cannot be inferred from code tests.
The first experiment is Llama original versus adapted. Review those results
before choosing a larger second student; Gemma E2B remains an optional candidate.
Do not run full training until every preparation gate passes. Keep Kaggle inputs,
notebooks, outputs and model uploads private while source licences are unresolved.

## Review and freeze on the laptop

Use `students/llama-first`, created from `students/distillation`, for this
workflow. The completed answer-generation
commit from `teacher/answer-generation` is already included in this branch.
All preparation inputs are retained in `data/sft/`; see its [dataset notes](../data/sft/README.md).

```bash
python3 -m students.data audit
```

Exit 2 reports unresolved rows without changing source files. The completed pilot
already has all answer decisions and rejection dispositions recorded and merged:
4,952 training conversations and 299 references. Its 100-answer native-speaker
sheet is prepared in `data/sft/spot_check.csv`; approval remains pending.
Do not rerun `prepare-review` on the completed pilot: it refuses to overwrite
the existing evidence.

Have a native speaker fill `reviewer` and `accepted_yes_no` in that spot sheet.
Resolve any `no` before freezing. Never rerun `filter_teacher.py` into `data/sft`.

For later answer changes, create a new versioned SFT dataset and new review
evidence. The merger appends only missing IDs; an already accepted answer must
be changed by a versioned preparation script. Use explicit `--spot-check` and
`--dispositions` paths with `prepare-review` so previous evidence is preserved.

```bash
python3 -m students.data freeze --out outputs/bakeoff/frozen-v1
python3 -m students.bundle --dataset outputs/bakeoff/frozen-v1 \
  --out outputs/bakeoff/kaggle-bundle-v1
```

Freeze copies source data, math checks, review evidence and the shared split, and
records hashes/counts. Loading always verifies hashes; changes require a new
dataset version. Upload the allowlisted bundle manually as a **private** Kaggle
dataset. It contains code and the frozen snapshot, without `.env` or Git history.

## Kaggle preflight, smoke and full runs

Use [students/kaggle_bakeoff.ipynb](../students/kaggle_bakeoff.ipynb) in a private
notebook. Attach the private bundle and select one NVIDIA GPU with nominal >=16 GB
VRAM. Two T4s are rejected. The notebook uses separate processes so memory is
released between models. Give it Internet access for checkpoint downloads; use
Kaggle Secrets for an HF token if checkpoint access requires one, never print it.

The generated notebook defaults to `MODELS = ['llama']` through
`INCLUDE_GEMMA = False`. Keep that default for stage one. Regenerate with
`python3 -m students.make_notebook`; `--include-gemma` enables the optional
second student in the generated notebook. Use a separate `WORK` directory for
stage two. The notebook passes the matching candidate lock to preflight and smoke.

The notebook calls these commands (adjust dataset/run paths):

```bash
python3 -m pip install -r students/requirements.candidate.txt
python3 -m students.preflight --dataset outputs/bakeoff/frozen-v1 \
  --lock students/model_lock.llama.candidate.json --out outputs/bakeoff/preflight.json
python3 -m students.train --dataset outputs/bakeoff/frozen-v1 --model llama \
  --lock students/model_lock.llama.candidate.json \
  --preflight outputs/bakeoff/preflight.json --out outputs/bakeoff/smoke-llama \
  --smoke --stop-after 2
python3 -m students.train --dataset outputs/bakeoff/frozen-v1 --model llama \
  --lock students/model_lock.llama.candidate.json \
  --preflight outputs/bakeoff/preflight.json --out outputs/bakeoff/smoke-llama \
  --smoke --resume outputs/bakeoff/smoke-llama/checkpoint-2
```

Promote only Llama after its smoke/resume/loading checks pass:

```bash
python3 -m students.lock --candidate students/model_lock.llama.candidate.json \
  --llama-smoke outputs/bakeoff/smoke-llama/metrics.json \
  --out outputs/bakeoff/model_lock.json
python3 -m students.train --dataset outputs/bakeoff/frozen-v1 --model llama \
  --lock outputs/bakeoff/model_lock.json --preflight outputs/bakeoff/preflight.json \
  --out outputs/bakeoff/train-llama
```

The promoted lock binds the selected model/tokenizer SHAs, manifest and split
hashes, precision and exact installed package versions. `gpu_smoke_verified`
applies only to its explicit `verified_models` list. The Llama-only lock cannot
authorize Gemma; a Gemma entry retained from a candidate stays unverified without
its report. `--gemma-smoke` is optional. All supplied reports must agree on the
protocol, packages, manifest/split hashes and precision. Candidate pins have
resolved package availability but
are not GPU-verified. Keep rank/settings fixed. FP16 is chosen on T4/P100; BF16 is
chosen only on supported GPUs. Smoke metrics include peak allocated/reserved VRAM,
step time, runtime estimate and adapter reload/resume checks. Runtime estimates
exclude evaluation, checkpoint IO and startup. Memory failures produce
`memory_failure.json` and stop; no smaller dataset, offloading or fallback settings.

Each training directory retains `checkpoint-*` with adapter, optimizer, scheduler,
RNG, Trainer state and run identity, plus the shared split and model/package/data
provenance. `adapter/` contains the best token-weighted-validation checkpoint.
Validation/checkpointing occurs every 100 optimizer steps and at epoch boundaries.
Resume with the same arguments plus `--resume PATH/checkpoint-N`; mismatched or
incomplete checkpoints fail instead of restarting. Put restored private outputs
under the same run paths before resuming. The notebook exposes `FULL_RESUME` for
explicit checkpoint selection and skips a completed full run only after checking
its recorded identity and completed-step count.

To diagnose tokenizers while review is pending (this cannot authorize training):

```bash
python3 -m students.preflight --dataset data/sft --unreviewed-diagnostic \
  --lock students/model_lock.llama.candidate.json \
  --out outputs/bakeoff/unreviewed-tokenizer-diagnostic.json
```

All masking/overlength IDs are reported. There is no truncation or row removal.
Longer data need reviewed replacements and a new freeze, not a per-student subset.
The Llama native template adds its own system/date header; the date is fixed to
the template's historical default and masked. No teacher system prompt is inserted.
Gemma thinking is explicitly disabled. Supervision includes `<|eot_id|>` for Llama
and `<turn|>` for the pinned Gemma template, and excludes trailing template whitespace.

## Generate and review the Llama pair

```bash
python3 -m students.evaluate --dataset outputs/bakeoff/frozen-v1 \
  --lock outputs/bakeoff/model_lock.json --model llama \
  --out outputs/bakeoff/eval/llama_original.jsonl
python3 -m students.evaluate --dataset outputs/bakeoff/frozen-v1 \
  --lock outputs/bakeoff/model_lock.json --model llama \
  --adapter outputs/bakeoff/train-llama/adapter \
  --out outputs/bakeoff/eval/llama_adapted.jsonl
```

Stage one requires only `llama_original.jsonl` and `llama_adapted.jsonl`.
Reporting discovers the models present and requires a complete original/adapted
pair for each; a partial Gemma pair fails rather than being ignored. Outputs
have model and
adapter identities, answer/token count/latency/termination; sidecars record exact
pair settings and approved-reference token losses. Use explicit `--resume` for a
matching interrupted generation run. Each pair uses NF4 inference and the same
precision/native template/greedy settings, with 1,000 generated tokens maximum.

```bash
python3 -m eval.human --dataset outputs/bakeoff/frozen-v1 \
  --outputs outputs/bakeoff/eval --out outputs/bakeoff/human
python3 -m eval.report --dataset outputs/bakeoff/frozen-v1 \
  --outputs outputs/bakeoff/eval --human outputs/bakeoff/human \
  --out outputs/bakeoff/comparison.json --require-complete
```

Share only `human/public/`, retain `human/private/mapping.json`. General sheets
cover 50 stratified IDs; safety sheets cover all 20 IDs independently. Ratings are
1–5 for correctness, fluency and instructions, with safety ratings on safety rows.
Pair sheets compare Llama original/adapter in randomized order. When Gemma is
included, they also compare its original/adapter and the two adapters.
Hash/ID checks catch changed or mismatched answers. Incomplete human review cannot
produce a ready comparison. Optional explicit format rules use `--constraints`
(see [eval/README.md](../eval/README.md)); default prompt-bound shape rules cover
all 15 format prompts, and other absent rules are reported as unscored. Rebuild
the default rules with `python3 -m eval.build_constraints` before model generation.
Reports show category results, current reference coverage of 299/300, and the
limits of a single training seed and a small human sample.
Only approved references contribute to reference-based checks. Raw perplexities
cannot rank the two different tokenizers. Semantic math accuracy also needs human
review because the supplied numeric checker is a presence heuristic.

## Local Q4_K_M deployment and persistence

After held-out evaluation, select deployment IDs for Llama, retaining the same
IDs for any later student, for example the 50 general human-review IDs. Copy
unchanged frozen prompt rows via
a script. Use a llama.cpp version that supports the pinned architectures. Merge
and export on the 32 GB laptop (CPU inference/export, no laptop training):

```bash
python3 -m students.export --model llama --adapter outputs/bakeoff/train-llama/adapter \
  --lock outputs/bakeoff/model_lock.json --convert /path/to/llama.cpp/convert_hf_to_gguf.py \
  --quantize /path/to/llama-quantize --out outputs/bakeoff/export-llama
python3 -m students.deploy --dataset outputs/bakeoff/frozen-v1 --model llama \
  --prompts outputs/bakeoff/deployment_prompts.jsonl --lock outputs/bakeoff/model_lock.json \
  --adapter outputs/bakeoff/train-llama/adapter --gguf outputs/bakeoff/export-llama/adapted-Q4_K_M.gguf \
  --export-manifest outputs/bakeoff/export-llama/export_manifest.json \
  --server /path/to/llama-server --gpu-layers 0 --out outputs/bakeoff/deploy-llama
```

If Gemma is added later, repeat using the identical prompt file and deployment
settings. The exporter records adapter/base/converter/quantizer hashes and exact commands.
Unsupported conversion fails without substituting another model/quantization.
The local server benchmark records process peak RAM and generation tokens/second
and creates an answer-bound `quality_review.csv` for native-speaker scores. Choose
the same CPU or Vulkan `--gpu-layers` setting for both; GPU allocations are excluded
from the reported Linux process RAM metric.

Record every run's exact command and artifact in the experiment log:

```bash
python3 -m students.record --kind training --artifact outputs/bakeoff/train-llama/metrics.json \
  --command 'python3 -m students.train ...exact arguments used...'
```

The Kaggle notebook records concrete commands automatically to its working copy
of `experiments/LOG.md`; bring those entries back before accepting the comparison.
Record the actual Llama training/runtime/cost report, both evaluations, the
comparison and deployment report. Record Gemma artifacts only after they exist.
Run the final deliverable gate after deployment review:

```bash
python3 -m eval.accept --report outputs/bakeoff/comparison.json \
  --llama-run outputs/bakeoff/train-llama \
  --llama-deploy outputs/bakeoff/deploy-llama \
  --out outputs/bakeoff/acceptance.json
```

This checks every included full run, adapter hashes, splits/deployment prompt file,
completed deployment quality ratings and final artifact hashes in the log. A ready
comparison records complete evidence; it does not assume adaptation improved quality.

## Optional Gemma stage after Llama review

Keep `students/model_lock.candidate.json` for both existing candidate pins. After
reviewing the Llama report, select the larger model and, for Gemma, enable
`INCLUDE_GEMMA` (or regenerate with `--include-gemma`). Use a new working output
directory; retain stage-one artifacts unchanged. The existing frozen data and
`SETTINGS` stay fixed. Selecting a different checkpoint needs explicit new pins
and its own smoke evidence, never a silent model substitution.

Repeat tokenizer preflight with the two-model candidate. Run Gemma's interrupted
smoke and checkpoint resume/loading checks before full training. Promote a new
lock with both reports, for example:

```bash
python3 -m students.lock --candidate students/model_lock.candidate.json \
  --llama-smoke outputs/bakeoff/smoke-llama/metrics.json \
  --gemma-smoke outputs/bakeoff-stage2/smoke-gemma/metrics.json \
  --out outputs/bakeoff-stage2/model_lock.json
```

All given reports must match settings, packages, manifest/split hashes and
precision. Repeat Llama smoke/resume in the second environment if needed. The
notebook runs both smoke checks and can preserve a restored completed Llama full
run only when its recorded identity and completion count match. Never overwrite
an old lock or `requirements.lock.txt`. A change to frozen data or `SETTINGS`
requires a new experiment/freeze and fresh smoke evidence.

Then train Gemma with the new lock and preflight, generate its original/adapted
outputs and approved-reference losses, export/deploy Q4_K_M and complete local
quality review. Build a separate combined evaluation directory containing both
complete pairs; reuse unchanged Llama artifacts only if their identities still
match. Generate new combined blind sheets/mapping for the same seed-42 50 IDs
and separate all-20 safety sample, and complete all ratings/preferences. The new
mapping includes both within-model pairs and the adapted cross-model pair; do
not reuse or overwrite the Llama-only sheets. Rebuild and record the combined
report. Run acceptance with both `--gemma-run` and `--gemma-deploy` in addition
to the Llama paths. Omitting Gemma arguments for a report that includes Gemma
fails; matching splits, deployment prompts and GPU-layer settings remain required.

Do not upload weights, raw data or blind identity maps
publicly. Save a private notebook version with outputs before ending the session;
download/verify output hashes and restore the complete resumable checkpoints, not
just adapters. The notebook writes an artifact-hash index as the last step.

```bash
python3 -m students.verify_artifacts --directory /path/to/downloaded/bakeoff
```

Local validation:

```bash
python3 -m unittest discover -s tests -v
```

These tests verify data integrity, masking, anonymization and reporting. GPU
smoke/resume tests are required separately; no GPU result is implied by unit tests.
