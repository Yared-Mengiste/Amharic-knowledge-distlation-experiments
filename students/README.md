# Student bakeoff

The two-model QLoRA protocol is in [the plan](../docs/PLAN_student_bakeoff.md).
Follow [the run guide](../docs/RUN_student_bakeoff.md) for review, dataset freezing,
Kaggle smoke/resume tests, training, four-way evaluation and local deployment.

All commands run from the repository root as Python modules. Preparation/review,
offline metrics and unit tests use the standard library. GPU commands need Python
3.12 and [candidate dependencies](requirements.candidate.txt); these become working
pins only after **both** GPU smoke and resume tests pass.

```bash
python3 -m students.data audit
python3 -m unittest discover -s tests -v
```

An audit exit code of 2 means review is incomplete. No source SFT/review files are
overwritten. Snapshots, checkpoints, adapters, generated answers and blind sheets
belong under ignored `outputs/` and must remain private.

`SETTINGS` in [config.py](config.py) controls training arguments, data-split seeds,
quantization, template options and generation limits. Completion checks and the
Kaggle notebook derive optimizer updates from those same settings, including
partial batches and epochs. The documented first-experiment values remain fixed.

Assistant loss selects supervised tokens and checkpoints 64-position chunks,
recomputing their FP32 loss buffers during backward. This reduces retained loss
memory; the model still produces full logits, so actual GPU capacity must be
measured with both smoke/resume runs.

Loss chunk size is recorded in the protocol. Existing frozen manifests or
checkpoints created before this change require a new freeze and fresh GPU smoke
evidence; changing settings does not reuse an older approved snapshot or lock.
