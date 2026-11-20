# Run the reviewed-response student bakeoff

Use [the frozen protocol](PLAN_student_bakeoff.md). This repository supplies the
workflow; human approval and GPU smoke results cannot be inferred from code tests.
Do not run full training until every preparation gate passes. Keep Kaggle inputs,
notebooks, outputs and model uploads private while source licences are unresolved.

## Review and freeze on the laptop

Finish the teacher stage and merge it before moving to `students/bakeoff`.
Implementation milestone commits can be prepared on the current teacher branch
while review is pending; branch creation remains deferred until that merge.

```bash
python3 -m students.data audit
```

Exit 2 reports unresolved rows without changing source files. Fill every
`data/sft/review_queue.csv` decision with `yes` or `no`. A correction needs the
complete `fixed_response` and `yes`. Merge approved answers:

```bash
python3 teacher/merge_review.py --review data/sft/review_queue.csv \
  --teacher data/teacher/teacher_outputs.jsonl --outdir data/sft
python3 -m students.data prepare-review
```

The preparation command creates `data/review/rejected_dispositions.csv` and a
100-answer stratified `spot_check.csv`. It refuses to overwrite evidence. Run it
after the review queue decisions and corrections are final, so the spot check
binds to accepted answer hashes. If answers change later, create a new spot sheet
with `--spot-check` and a new dispositions path, preserving the old evidence.

Give each rejected/no row a disposition (`corrected`, `regenerated`, `excluded`)
and nonempty notes. For corrected hard rejections, add the ID to the review queue
with the full replacement and `yes`, then merge. For regenerated rejections, review
the new answer in the queue with `yes`, merge, and record regeneration in notes.
No-decision rows may not remain in SFT; change to `yes` after replacement approval.
The existing merge command appends only missing IDs: if an already accepted answer
changes, prepare a new versioned SFT file with a script rather than rerunning the
filter or silently retaining an old answer.

Have a native speaker fill `reviewer` and `accepted_yes_no` in the spot sheet.
Resolve any `no` before freezing. Never rerun `filter_teacher.py` into `data/sft`.

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

The notebook calls these commands (adjust dataset/run paths):

```bash
python3 -m pip install -r students/requirements.candidate.txt
python3 -m students.preflight --dataset outputs/bakeoff/frozen-v1 \
  --out outputs/bakeoff/preflight.json
python3 -m students.train --dataset outputs/bakeoff/frozen-v1 --model llama \
  --preflight outputs/bakeoff/preflight.json --out outputs/bakeoff/smoke-llama \
  --smoke --stop-after 2
python3 -m students.train --dataset outputs/bakeoff/frozen-v1 --model llama \
  --preflight outputs/bakeoff/preflight.json --out outputs/bakeoff/smoke-llama \
  --smoke --resume outputs/bakeoff/smoke-llama/checkpoint-2
```

Repeat the two smoke commands for Gemma before promoting any pins:

```bash
python3 -m students.lock --llama-smoke outputs/bakeoff/smoke-llama/metrics.json \
  --gemma-smoke outputs/bakeoff/smoke-gemma/metrics.json \
  --out outputs/bakeoff/model_lock.json
python3 -m students.train --dataset outputs/bakeoff/frozen-v1 --model llama \
  --lock outputs/bakeoff/model_lock.json --preflight outputs/bakeoff/preflight.json \
  --out outputs/bakeoff/train-llama
python3 -m students.train --dataset outputs/bakeoff/frozen-v1 --model gemma \
  --lock outputs/bakeoff/model_lock.json --preflight outputs/bakeoff/preflight.json \
  --out outputs/bakeoff/train-gemma
```

The promoted lock binds both model/tokenizer SHAs, the smoke dataset hash and exact
installed package versions. Candidate pins have resolved package availability but
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
  --out outputs/bakeoff/unreviewed-tokenizer-diagnostic.json
```

All masking/overlength IDs are reported. There is no truncation or row removal.
Longer data need reviewed replacements and a new freeze, not a per-student subset.
The Llama native template adds its own system/date header; the date is fixed to
the template's historical default and masked. No teacher system prompt is inserted.
Gemma thinking is explicitly disabled. Supervision includes `<|eot_id|>` for Llama
and `<turn|>` for the pinned Gemma template, and excludes trailing template whitespace.

## Generate and review all four configurations

```bash
python3 -m students.evaluate --dataset outputs/bakeoff/frozen-v1 \
  --lock outputs/bakeoff/model_lock.json --model llama \
  --out outputs/bakeoff/eval/llama_original.jsonl
python3 -m students.evaluate --dataset outputs/bakeoff/frozen-v1 \
  --lock outputs/bakeoff/model_lock.json --model llama \
  --adapter outputs/bakeoff/train-llama/adapter \
  --out outputs/bakeoff/eval/llama_adapted.jsonl
```

Repeat for `gemma_original.jsonl` and `gemma_adapted.jsonl`. Outputs have model and
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
Pair sheets compare each original/adapter and the two adapters in randomized order.
Hash/ID checks catch changed or mismatched answers. Incomplete human review cannot
produce a ready comparison. Optional explicit format rules use `--constraints`
(see [eval/README.md](../eval/README.md)); default prompt-bound shape rules cover
all 15 format prompts, and other absent rules are reported as unscored. Rebuild
the default rules with `python3 -m eval.build_constraints` before model generation.
Only approved references contribute to reference-based checks. Raw perplexities
cannot rank the two different tokenizers. Semantic math accuracy also needs human
review because the supplied numeric checker is a presence heuristic.

## Local Q4_K_M deployment and persistence

After held-out evaluation, select the same deployment IDs for both students, for
example the 50 general human-review IDs, copying unchanged frozen prompt rows via
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

Repeat for Gemma using the identical prompt file and deployment settings. The
exporter records adapter/base/converter/quantizer hashes and exact commands.
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
Record both training/runtime/cost reports, four evaluations, the comparison and
both deployment reports. Run the final deliverable gate after deployment review:

```bash
python3 -m eval.accept --report outputs/bakeoff/comparison.json \
  --llama-run outputs/bakeoff/train-llama --gemma-run outputs/bakeoff/train-gemma \
  --llama-deploy outputs/bakeoff/deploy-llama --gemma-deploy outputs/bakeoff/deploy-gemma \
  --out outputs/bakeoff/acceptance.json
```

This checks the full runs, adapter hashes, identical splits/deployment prompt file,
completed deployment quality ratings and final artifact hashes in the log. A ready
comparison records complete evidence; it does not assume adaptation improved quality.

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
