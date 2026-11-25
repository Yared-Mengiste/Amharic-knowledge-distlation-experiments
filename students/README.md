# Student bakeoff

The Llama-first QLoRA protocol is in [the plan](../docs/PLAN_student_bakeoff.md).
Follow [the run guide](../docs/RUN_student_bakeoff.md) for review, dataset freezing,
Kaggle smoke/resume tests, Llama training, original/adapted evaluation and local
deployment. Review the Llama results before adding a larger student; Gemma E2B
support and its candidate pins remain available for an optional second stage.

All commands run from the repository root as Python modules. Preparation/review,
offline metrics and unit tests use the standard library. GPU commands need Python
3.11–3.13 and [candidate dependencies](requirements.candidate.txt); these become working
pins only for models with successful GPU smoke, resume and adapter-loading checks.
Use `model_lock.llama.candidate.json` for stage one; the two-model candidate is
unchanged. Promoted locks list `verified_models`; Llama evidence does not authorize
Gemma training, evaluation, export or deployment.

```bash
python3 -m students.data audit --dataset data/sft/pilot-v1
python3 -m unittest discover -s tests -v
```

An audit exit code of 2 means review is incomplete. No source SFT/review files are
overwritten. Snapshots, checkpoints, adapters, generated answers and blind sheets
belong under ignored `outputs/` and must remain private.

The completed native-speaker review is frozen at `outputs/bakeoff/frozen-v1`:
4,947 training conversations after five documented exclusions, plus 299 held-out
references and 300 evaluation prompts. See the [version notes](../data/sft/pilot-v1/README.md).

`SETTINGS` in [config.py](config.py) controls training arguments, data-split seeds,
quantization, template options and generation limits. Completion checks and the
Kaggle notebook derive optimizer updates from those same settings, including
partial batches and epochs. The documented first-experiment values remain fixed.

Assistant loss selects supervised tokens and checkpoints 64-position chunks,
recomputing their FP32 loss buffers during backward. This reduces retained loss
memory; the model still produces full logits, so actual GPU capacity must be
measured by each included model’s smoke/resume run.

Loss chunk size is recorded in the protocol. Existing frozen manifests or
checkpoints created before this change require a new freeze and fresh GPU smoke
evidence; changing settings does not reuse an older approved snapshot or lock.

The notebook defaults to Llama only. Regenerate with `python3 -m students.make_notebook`;
`--include-gemma` enables Gemma. A later Gemma stage needs a new lock/output
directory, matching smoke evidence, tokenizer preflight, full training, both
evaluations/reference losses, new combined blind sheets, and deployment review.
Do not overwrite stage-one evidence. The frozen dataset/split and `SETTINGS` stay
unchanged when adding a model. Reports include category results, reference coverage
(currently 299/300), and the limits of one seed and a small human sample.
