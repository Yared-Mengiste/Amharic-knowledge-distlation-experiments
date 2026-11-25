# Amharic distillation

The goal is a small Amharic chatbot model (1B–4B), built by distilling answers from Gemma 4 26B A4B.

- **Project guide** (goals, status, decisions, rules): [`CLAUDE.md`](CLAUDE.md)
- **How to run the teacher generation:** [`docs/RUN_teacher.md`](docs/RUN_teacher.md)
- **Reviewed pilot data:** [data/sft/pilot-v1/README.md](data/sft/pilot-v1/README.md); 4,947 training conversations and 299 held-out teacher references, frozen locally at `outputs/bakeoff/frozen-v1`. Required audit inputs and original data are retained.
- **Experiment log:** [`experiments/LOG.md`](experiments/LOG.md)
- **Student bakeoff:** [protocol](docs/PLAN_student_bakeoff.md), [run guide](docs/RUN_student_bakeoff.md), and [private Kaggle notebook](students/kaggle_bakeoff.ipynb). Full training requires completed teacher review and successful GPU smoke/resume tests.
- **First Llama run:** Kaggle training and original/adapted evaluation completed; local artifacts are in `outputs/llama_first_run_all_2`. Reference perplexity improved from 36.81 to 15.28, while repetition flags increased from 32.7% to 59.3%. Blind native-speaker review and deployment checks remain pending; see [the experiment log](experiments/LOG.md).

## Quick start

For a new teacher run, first rebuild the removed intermediate prompt folder with
`bash data_prep/run_all.sh`. The completed pilot is already available in `data/sft/`.

```bash
cp .env.example .env            # add your OpenRouter key
set -a; source .env; set +a
python teacher/gen_teacher.py --train data/prompts/pilot_train.jsonl --eval data/prompts/pilot_eval.jsonl \
  --out data/teacher/smoke.jsonl --openrouter --workers 8 --limit 20
```

Teacher scripts use Python 3.9+ and the standard library. Student preparation and
offline evaluation use the standard library; student GPU runs use Python 3.11–3.13
and package pins validated by smoke/resume tests for each selected model.
