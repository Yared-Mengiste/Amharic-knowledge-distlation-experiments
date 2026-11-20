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
