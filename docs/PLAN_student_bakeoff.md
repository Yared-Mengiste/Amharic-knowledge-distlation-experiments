# Post-review Amharic student distillation

## Objective and prerequisites

Compare `rasyosef/Llama-3.2-1B-Amharic-Instruct` and `google/gemma-4-E2B-it`
using response-level distillation through QLoRA on the same reviewed teacher answers.
Compare each adapter with its unchanged starting checkpoint, then compare adapters.

Before full training, every review-queue decision must be `yes` or `no`; corrected
answers need the complete replacement in `fixed_response` and `yes`. Merge approved
rows. Document rejected-answer dispositions as corrected, regenerated, or excluded.
A native speaker must spot-check approximately 100 accepted training answers.

Freeze `data/sft/sft_train.jsonl` and `data/sft/eval_teacher.jsonl`, with file hashes,
row/category counts, review evidence, and a shared category-stratified 10% validation
split. Generate benchmark answers for all 300 `data/sft/pilot_eval.jsonl` IDs.
Both students use identical frozen training and validation IDs.

**Never rerun the filter into the reviewed directory: it overwrites SFT and review files.**
Create `students/bakeoff` after the teacher-generation work has been committed and
merged. Implementation scaffolding may be prepared while reviews are pending;
it must not imply dataset approval or completed training.

## Implementation and training

Reusable preparation, training, evaluation commands and a Kaggle notebook retain
the existing `id`, `category`, `language_mode`, `messages: [user, assistant]` schema.
Use PyTorch, Transformers, PEFT and bitsandbytes. Pin exact working package versions
and model/tokenizer revisions after GPU smoke tests; candidate pins are not validation.

| Setting | Both students |
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
Train Llama then Gemma using the same frozen split. Use FP16 on T4/P100 and BF16 only
where supported. Run a short training and checkpoint-resume smoke test per model.
Measure peak VRAM and optimizer-step time to estimate runtime. If Gemma cannot fit,
stop with a memory report; never silently change the data or settings. E2B has about
**5.1B total parameters including embeddings**; effective size is not its memory size.

Save adapters, resumable optimizer/RNG checkpoints, split IDs, configuration, exact
package versions and metrics. Persist private notebook outputs before ending the
session. Verify adapter/checkpoint loading before resuming. Use the laptop for
preparation, review and quantized inference, not training.

## Evaluation

Evaluate original Llama, adapted Llama, original Gemma and adapted Gemma on identical
prompt content, greedy decoding and a 1,000-token output cap. Keep precision and
template settings identical within each original/adapted pair. Record prompt ID,
model/tokenizer revision, adapter identifier/hash, answer, output tokens, latency
and termination reason.

Measure math accuracy against supplied checks; format/instruction compliance;
requested-language consistency; truncation/repetition; assistant-token loss on
approved eval references; native-speaker correctness, fluency and instruction
following (1–5); and original/adapted win/tie/loss. Clearly distinguish automated
heuristics and manual rubric scores. Report category-level results, overall
coverage, missing references and uncertainty. Compare perplexity within a model
pair only; different tokenizers make cross-model perplexity incomparable.

Select 50 category-stratified eval IDs with seed 42 for blind human comparison.
Separately review all 20 safety IDs. Hide identities, randomize order, and keep the
mapping private. After evaluation, test both adapters locally as Q4_K_M GGUFs on
the same deployment prompts; record peak RAM, tokens/second and answer quality.
Use no paid judge calls. Keep rank 16 fixed for this first experiment.

## Acceptance and deliverables

Before full training verify IDs/split separation, assistant-only masking and ending
tokens, frozen original parameters, adapter counts, finite loss and gradients,
and checkpoint resume without restarting completed training. Before accepting the
comparison verify 300 outputs for each of four configurations, explicit reference
coverage, correct human-score mapping, and experiment provenance in
`experiments/LOG.md`: settings, hashes, revisions, cost, runtime and results.

Deliver two adapters, resumable checkpoints, four output sets, blind human-review
sheets and a comparison report. Determine adaptation benefit from held-out results,
not training loss alone. See [RUN_student_bakeoff.md](RUN_student_bakeoff.md) for commands.
