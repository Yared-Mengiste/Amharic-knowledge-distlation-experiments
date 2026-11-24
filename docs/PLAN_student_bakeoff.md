# Post-review Amharic student distillation

## Objective and prerequisites

Run `rasyosef/Llama-3.2-1B-Amharic-Instruct` first using response-level
distillation through QLoRA on the reviewed teacher answers. Compare its adapter
with the unchanged starting checkpoint to assess whether distillation helps the
small model. Review those results before selecting a larger second student,
probably `google/gemma-4-E2B-it`. Gemma support and candidate pins remain available;
it is optional and deferred, with no winner declared in advance.

Before full training, every review-queue decision must be `yes` or `no`; corrected
answers need the complete replacement in `fixed_response` and `yes`. Merge approved
rows. Document rejected-answer dispositions as corrected, regenerated, or excluded.
A native speaker must spot-check approximately 100 accepted training answers.

Freeze `data/sft/sft_train.jsonl` and `data/sft/eval_teacher.jsonl`, with file hashes,
row/category counts, review evidence, and a shared category-stratified 10% validation
split. Generate benchmark answers for all 300 `data/sft/pilot_eval.jsonl` IDs.
Every student included in an experiment uses identical frozen training and
validation IDs.

**Never rerun the filter into the reviewed directory: it overwrites SFT and review files.**
Answer-generation work is committed on `teacher/answer-generation` and included
in `students/distillation`. Implementation scaffolding may be prepared while
reviews are pending; it must not imply dataset approval or completed training.

## Implementation and training

Reusable preparation, training, evaluation commands and a Kaggle notebook retain
the existing `id`, `category`, `language_mode`, `messages: [user, assistant]` schema.
Use PyTorch, Transformers, PEFT and bitsandbytes. Pin exact working package versions
and model/tokenizer revisions after GPU smoke tests; candidate pins are not validation.

| Setting | Each selected student |
|---|---|
| Quantization | 4-bit NF4, double quantization |
| LoRA rank / alpha / dropout | 16 / 32 / 0.05 |
| Targets | Text-decoder q/k/v/o and gate/up/down projections |
| Learning rate / epochs | 0.0001 / 2 |
| Microbatch / gradient accumulation | 1 / 16 |
| Maximum sequence length | 2,048 |
| Optimizer | Paged AdamW 8-bit |
| Schedule / warmup | Cosine / 3% |
| Weight decay / gradient clipping | 0.01 / 1.0 |
| Seed / validation | 42 / category-stratified 10% |

These values are defined once in `students/config.py`. Trainer arguments,
generation limits, split seeds and optimizer-update checks use that recorded
configuration. Changing the protocol requires a new frozen dataset and smoke
evidence. Assistant CE selects supervised positions and checkpoints 64-position
chunks so FP32 loss buffers are recomputed during backward rather than retained
across the whole sequence. Full model logits still consume memory.

Use each native chat template. Supervise assistant tokens including the ending token;
mask user tokens and padding. Do not insert the teacher system prompt. Disable Gemma
thinking. Freeze original weights, embeddings, vision and audio components. Attach
adapters only to the text decoder. Expect **11,272,192** Llama adapter parameters;
measure Gemma separately. Enable gradient checkpointing. Validate and save every
100 optimizer steps and at epoch boundaries. Select the lowest assistant-token
validation-loss checkpoint. Tokenizer preflight must report all overlength IDs and
stop without truncating or dropping any conversation.

## Kaggle execution

Use a private notebook and private dataset, one NVIDIA GPU with at least 16 GB VRAM.
Train Llama only for the first experiment. Add Gemma after reviewing the Llama
results, using the same frozen split and unchanged protocol. Use FP16 on T4/P100
and BF16 only
where supported. Run a short training and checkpoint-resume smoke test per model.
Measure peak VRAM and optimizer-step time to estimate runtime. If Gemma cannot fit,
stop with a memory report; never silently change the data or settings. E2B has about
**5.1B total parameters including embeddings**; effective size is not its memory size.

Save adapters, resumable optimizer/RNG checkpoints, split IDs, configuration, exact
package versions and metrics. Persist private notebook outputs before ending the
session. Verify adapter/checkpoint loading before resuming. Use the laptop for
preparation, review and quantized inference, not training.

## Evaluation

First evaluate original and adapted Llama on all 300 prompts, with greedy
decoding and a 1,000-token output cap. When Gemma is included later, add its
original/adapted pair on identical prompt content and compare the adapters. Keep
precision and
template settings identical within each original/adapted pair. Record prompt ID,
model/tokenizer revision, adapter identifier/hash, answer, output tokens, latency
and termination reason.

Measure math accuracy against supplied checks; format/instruction compliance;
requested-language consistency; truncation/repetition; assistant-token loss on
approved eval references; native-speaker correctness, fluency and instruction
following (1–5); and original/adapted win/tie/loss. Clearly distinguish automated
heuristics and manual rubric scores. Report category-level results, overall
coverage (currently 299/300 approved references), missing references and
uncertainty. Explicitly state that this is a single-seed experiment with a small
human sample; intervals do not establish stability across seeds. Compare
perplexity within a model
pair only; different tokenizers make cross-model perplexity incomparable.

Select 50 category-stratified eval IDs with seed 42 for blind human comparison.
Separately review all 20 safety IDs. Hide identities, randomize order, and keep the
mapping private. After evaluation, test each included adapter locally as a
Q4_K_M GGUF on
the same deployment prompts; record peak RAM, tokens/second and answer quality.
Use no paid judge calls. Keep rank 16 fixed for this first experiment.

## Acceptance and deliverables

Before full training verify IDs/split separation, assistant-only masking and ending
tokens, frozen original parameters, adapter counts, finite loss and gradients,
and checkpoint resume without restarting completed training. Before accepting the
comparison verify 300 outputs for each included configuration (two for Llama
only, four with Gemma), explicit reference coverage, correct human-score mapping,
and experiment provenance in
`experiments/LOG.md`: settings, hashes, revisions, cost, runtime and results.

Deliver one adapter, resumable checkpoints, two output sets, blind human-review
sheets and a Llama original/adapted report for stage one. Stage two adds the
selected larger student, its adapter/checkpoints/output pair and a combined report.
Only models explicitly listed in a promoted lock’s `verified_models` are authorized
for full training/evaluation/export/deployment; Llama evidence never verifies Gemma.
Before adding Gemma, repeat its tokenizer preflight, GPU smoke/resume/loading,
full training, generation/reference losses, blind review and Q4_K_M deployment
quality review. Promote a new lock using matching Llama and Gemma smoke reports;
repeat Llama smoke if needed to match packages, precision and frozen hashes. Use
new artifact directories and a new combined blind mapping; never overwrite stage
one. Determine adaptation benefit from held-out results,
not training loss alone. See [RUN_student_bakeoff.md](RUN_student_bakeoff.md) for commands.
