# Reviewed pilot v1

Yared completed all 100 native-speaker spot checks: 95 accepted, 5 rejected.
The five rejected training conversations were excluded without changing any
retained conversation, evaluation prompt or reference. Their IDs, reviewed answer
hashes, reviewer and exclusion reasons are in [removed.csv](removed.csv).
The original review sheet is copied byte for byte, including every `no` decision.

This version contains 4,947 training conversations, 299 held-out teacher
references and all 300 evaluation prompts. There are 15 total exclusions,
including the 10 earlier exclusions. All queue decisions and dispositions are
complete. [preparation.json](preparation.json) records the original input hashes.

The audit accepts a native-review `no` only when the answer is absent from the
accepted data and has a matching training teacher ID, an explicit queue `no`, a
documented excluded disposition and the reviewed answer hash. At least 100
completed spot checks remain required; excluded answers are never relabelled as
approved.

The version was prepared on 2026-11-24 using a local `resolve_spot_checks.py`
helper, which is intentionally not committed. Its command and original input
hashes are retained in `preparation.json` and the experiment log. Audit and freeze:

```bash
python3 -m students.data audit --dataset data/sft/pilot-v1
python3 -m students.data freeze --dataset data/sft/pilot-v1 --out outputs/bakeoff/frozen-v1
```

The frozen snapshot already exists; choose a fresh output path for another
freeze. To verify the existing local snapshot:

```bash
python3 -m students.data audit --dataset outputs/bakeoff/frozen-v1
python3 -c "from students.data import load_frozen; load_frozen('outputs/bakeoff/frozen-v1'); print('Frozen hashes and split verified')"
```

The frozen seed-42 category-stratified split has 4,452 training and 495 validation
IDs. The snapshot hashes all nine data/review inputs plus the removal ledger and
preparation record. Training-file SHA-256:
`eaba5bfc3f47923c2536ba95ffb36dc000866396ac0720e47cf3a4f959ecf7cd`.

The first Llama run completed preflight, smoke/resume, training and automated
evaluation using this snapshot. Blind human review and deployment remain pending;
see the [experiment log](../../../experiments/LOG.md).
