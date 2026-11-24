# Teacher generation for the Amharic distillation pilot (local run)

The completed pilot is retained in `data/sft/`: 4,952 training conversations
and 299 teacher evaluation references, plus all 300 evaluation prompts and the
evidence required by the student audit. Intermediate `data/prompts/`,
`data/teacher/` and `data/review/` folders have been removed. Native-speaker
spot checks remain pending; preparing this dataset does not record their approval.

The commands below describe regenerating a teacher run. First run
`bash data_prep/run_all.sh` to recreate the intermediate prompt folder.
See [the retained dataset](../data/sft/README.md) for completed-pilot files
and scratch rebuild commands.

This is the older local-run guide (Ollama / llama.cpp). The main run uses OpenRouter; see `docs/RUN_teacher.md`.

The three scripts need only Python 3.9+ (standard library) and a running Ollama or llama.cpp server.

| File | What it does |
|---|---|
| `gen_teacher.py` | Sends every prompt to the teacher and saves the answers. You can stop and restart it at any point. |
| `filter_teacher.py` | Checks the answers: language, truncation, looping, leaked thinking, math correctness. Splits them into accepted, review and rejected. |
| `merge_review.py` | Adds the rows you approved in the review sheet back into the training and eval files. |

Run every command from the repo root. Inputs are in `data/prompts/`; teacher outputs go to `data/teacher/` and filtered SFT data to `data/sft/`.

## 1. Set up Ollama

```bash
# Let Ollama answer 2 requests at once and keep the model in memory.
# Edit the service with `sudo systemctl edit ollama` and add:
#   [Service]
#   Environment="OLLAMA_NUM_PARALLEL=2"
#   Environment="OLLAMA_KEEP_ALIVE=24h"
#   Environment="OLLAMA_FLASH_ATTENTION=1"
sudo systemctl restart ollama
ollama list            # note the exact model name, e.g. gemma4:26b
```

Each parallel slot needs its own context memory. If memory is tight, keep `OLLAMA_NUM_PARALLEL=1` and `--workers 1`.

## 2. Smoke test (20 prompts, about 10–15 minutes)

```bash
python teacher/gen_teacher.py --train data/prompts/pilot_train.jsonl --eval data/prompts/pilot_eval.jsonl \
  --out data/teacher/smoke.jsonl --model gemma4:26b --workers 1 --limit 20 --log_every 10
```

Open `data/teacher/smoke.jsonl` and read 5–10 answers. Check that they're in Amharic, have no thinking text, and aren't cut off. Note the `out tok/s` figure.

Then try `--workers 2` on another 20 prompts. Keep 2 only if the total tok/s actually goes up.

## 3. Full run

```bash
tmux new -s teacher      # so the run survives a closed SSH session
systemd-inhibit --what=sleep:idle python teacher/gen_teacher.py \
  --train data/prompts/pilot_train.jsonl --eval data/prompts/pilot_eval.jsonl \
  --out data/teacher/teacher_outputs.jsonl --model gemma4:26b --workers 2 2>> data/teacher/gen_errors.log
# detach: Ctrl+B then D       reattach: tmux attach -t teacher
```

- **Order:** the 300 eval prompts run first, then train prompts mixed across categories. If you stop early, what you have is still a balanced sample.
- **Stop:** press Ctrl+C once. It finishes the requests in flight, then exits. Run the same command again to continue.
- **Progress:** a line every 30 seconds with rows done, tokens/s and ETA. You can also run `wc -l data/teacher/teacher_outputs.jsonl`.

**Time estimate:** 5,261 prompts × about 350 output tokens ≈ 1.8M tokens. At 10 tok/s that's about 50 hours, so 5–6 nights, or about 2 days running nonstop. More parallel slots can cut that if your hardware has room.

If you need to cut the time, use `--only eval` first, then train. The output-length caps per category are at the top of `gen_teacher.py`.

**Thinking is turned off** (`think: false`). If your Ollama version rejects that setting, the script retries without it, and the filter rejects any answer with leaked thinking text.

### Teacher system prompt

A short system prompt asks the teacher to answer in Amharic, not to invent numbers or links, and to handle self-harm and abuse carefully. It is only sent to the teacher and is not saved into the training data. Use `--system none` to turn it off, or `--system my_prompt.txt` to use your own.

## 4. Filter

```bash
python teacher/filter_teacher.py --teacher data/teacher/teacher_outputs.jsonl --math data/prompts/math_answers.csv \
  --review_ids data/prompts/teacher_answer_review_ids.txt --outdir data/sft
```

| Output | Contents |
|---|---|
| `data/sft/sft_train.jsonl` | Ready-to-train rows: user message plus the teacher's answer. |
| `data/sft/eval_teacher.jsonl` | The teacher's answers to the eval prompts, used as references when judging students. |
| `data/sft/review_queue.csv` | Rows a person must read: all 224 safety and sensitive IDs, unclear math, possible refusals, and invented phone numbers or links. |
| `data/sft/rejected.jsonl` | Rows that failed a hard check, with the reasons. |
| `data/sft/filter_report.md` | Counts per check and per category. |

**Hard checks (reject):**
- Empty answer.
- Truncated at the token cap.
- Thinking leaked into the answer.
- Looping or repeated text.
- Wrong language. Amharic answers need at least 70% Ge'ez letters (45% for code-switch, Latin-script and English-instruction prompts), and Amharic→English translations must be English.
- Wrong math answer compared with `math_answers.csv`.

**After filtering:**
- **Truncated rows:** rerun them with a higher cap. Delete their lines from `teacher_outputs.jsonl` (the IDs are in `rejected.jsonl`), raise the cap for that category, and run `gen_teacher.py` again. Only the missing rows are regenerated.
- **Wrong-language and looping rows:** rerunning them once with a different `--seed` usually fixes them.

## 5. Review and merge

Open `data/sft/review_queue.csv` in a spreadsheet. For each row:
1. Set `decision_keep_yes_no` to `yes` or `no`.
2. If you fixed the answer, paste the fixed text into `fixed_response`.

Then run:

```bash
python teacher/merge_review.py --review data/sft/review_queue.csv --teacher data/teacher/teacher_outputs.jsonl --outdir data/sft
```

`data/sft/sft_train.jsonl` is then the pilot training set.
