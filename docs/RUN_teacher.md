# How to run the teacher generation (OpenRouter)

The completed pilot is retained in `data/sft/`: 4,952 training conversations
and 299 teacher evaluation references, plus all 300 evaluation prompts and the
evidence required by the student audit. Intermediate `data/prompts/`,
`data/teacher/` and `data/review/` folders have been removed. Native-speaker
spot checks remain pending; preparing this dataset does not record their approval.

The commands below describe regenerating a teacher run. First run
`bash data_prep/run_all.sh` to recreate the intermediate prompt folder.
See [the retained dataset](../data/sft/README.md) for completed-pilot files
and scratch rebuild commands.

This runs Gemma 4 26B on OpenRouter to answer the pilot prompts, then filters the answers into a training set.

- **Time:** about 1–2 hours
- **Cost:** under $1

## 1. Check the input files

Run every command from the repo root. The prompt files live in `data/prompts/`:

```bash
wc -l data/prompts/pilot_train.jsonl data/prompts/pilot_eval.jsonl    # should show 4961 and 300
ls data/prompts/math_answers.csv data/prompts/teacher_answer_review_ids.txt
```

## 2. Set your OpenRouter key

1. Create a key at openrouter.ai.
2. Add about $5 credit to the account.
3. Set the key in your terminal:

```bash
cp .env.example .env      # then put your key in .env (never commit it)
set -a; source .env; set +a
```

## 3. Smoke test (20 prompts, about 2 minutes)

```bash
python teacher/gen_teacher.py --train data/prompts/pilot_train.jsonl --eval data/prompts/pilot_eval.jsonl \
  --out data/teacher/smoke.jsonl --openrouter --workers 8 --limit 20 --log_every 10
```

Read the answers:

```bash
python - <<'EOF'
import json
for l in open('data/teacher/smoke.jsonl'):
    r = json.loads(l)
    print(f"\n==== {r['category']} | {r['output_tokens']} tok | {r['finish_reason']}")
    print("Q:", r['messages'][-1]['content'][:150])
    print("A:", r['response'][:400])
EOF
```

**Check before going on:**

- [ ] Answers are in Amharic
- [ ] At most 1–2 show `length` (cut off)
- [ ] Gregorian years use እ.ኤ.አ., not ዓ.ም
- [ ] No thinking text or HTML in the answers

If something looks wrong, stop here and fix it first.

## 4. Full run

```bash
python teacher/gen_teacher.py --train data/prompts/pilot_train.jsonl --eval data/prompts/pilot_eval.jsonl \
  --out data/teacher/teacher_outputs.jsonl --openrouter --workers 16 --retries 6 2>> data/teacher/gen_errors.log
```

- Every 30 seconds it prints progress, speed, time left and cost so far.
- **If it stops** (Ctrl+C, network drop), run the same command again. It continues where it stopped.
- **Many `429` errors in `data/teacher/gen_errors.log`:** lower `--workers` to 8.
- **Optional:** to keep one provider for the whole run, add `--provider DeepInfra --quantizations bf16,fp8`. Use any provider listed on the model page.

When it finishes:

```bash
wc -l data/teacher/teacher_outputs.jsonl    # should be 5261
```

## 5. Filter the answers

```bash
python teacher/filter_teacher.py --teacher data/teacher/teacher_outputs.jsonl --math data/prompts/math_answers.csv \
  --review_ids data/prompts/teacher_answer_review_ids.txt --outdir data/sft
cat data/sft/filter_report.md
```

| File | What it is |
|---|---|
| `data/sft/sft_train.jsonl` | Accepted training rows, ready to use |
| `data/sft/eval_teacher.jsonl` | Teacher answers for the eval prompts, used as references |
| `data/sft/review_queue.csv` | Rows a person must check: safety prompts, unclear math, possible errors |
| `data/sft/rejected.jsonl` | Rows that failed a check, with the reason |

**If many rows are rejected as cut off or wrong language:**

1. Delete those IDs from `data/teacher/teacher_outputs.jsonl`.
2. Run step 4 again. Only the missing rows are regenerated.

## 6. Review and merge

1. Open `data/sft/review_queue.csv` in a spreadsheet.
2. Fill in `decision_keep_yes_no` for each row with `yes` or `no`.
3. If you corrected an answer, paste the corrected text into `fixed_response`.
4. Merge the kept rows back in:

```bash
python teacher/merge_review.py --review data/sft/review_queue.csv \
  --teacher data/teacher/teacher_outputs.jsonl --outdir data/sft
```

`data/sft/sft_train.jsonl` is now the training set for the student models.

## Before training

Have a native speaker read about 100 random answers from `sft_train.jsonl`. If the same kind of mistake keeps showing up (wrong dates, wrong facts, unnatural Amharic), fix the system prompt in `teacher/gen_teacher.py` and regenerate the affected rows.
