# Completed pilot SFT dataset

| File | Purpose |
|---|---|
| `sft_train.jsonl` | 4,952 accepted user/assistant training conversations. |
| `eval_teacher.jsonl` | 299 accepted held-out teacher references. |
| `pilot_eval.jsonl` | All 300 held-out generation prompts, including the ID without an accepted reference. |
| `review_queue.csv` | Completed answer decisions and corrected responses used by the merger and student audit. |
| `rejected.jsonl` | Original hard-rejection records used by the audit. |
| `rejected_dispositions.csv` | Decisions for all rejected and explicitly excluded rows. |
| `spot_check.csv` | Prepared 100-answer native-speaker sheet; reviewer approvals remain blank. |
| `teacher_outputs.jsonl` | Original teacher records required to check accepted answer provenance. |
| `math_answers.csv` | Frozen math checks required by the audit. |
| `teacher_answer_review_ids.txt` | Original safety/sensitive IDs needed to reproduce filtering. |
| `filter_report.md` | Historical automatic-filter report, before reviewed corrections were merged. |

The final training and reference files are unchanged from the completed review.
The original queue had 102 corrections, 197 retained answers and 8 exclusions;
107 hard rejections were recovered and 2 excluded. There are 10 exclusions in
total. Assistant review is separate from native-speaker approval, which remains
pending. Do not overwrite this dataset by rerunning the filter into this folder.

To verify the accepted files with the existing teacher scripts, rebuild into a
new scratch directory and compare both accepted JSONL files:

```bash
python3 teacher/filter_teacher.py --teacher data/sft/teacher_outputs.jsonl \
  --math data/sft/math_answers.csv --review_ids data/sft/teacher_answer_review_ids.txt \
  --outdir /tmp/amharic-sft-rebuild
python3 teacher/merge_review.py --review data/sft/review_queue.csv \
  --teacher data/sft/teacher_outputs.jsonl --outdir /tmp/amharic-sft-rebuild
cmp data/sft/sft_train.jsonl /tmp/amharic-sft-rebuild/sft_train.jsonl
cmp data/sft/eval_teacher.jsonl /tmp/amharic-sft-rebuild/eval_teacher.jsonl
```
