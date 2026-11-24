# Teacher output filter report

Rows read: **5261** · accepted: **4845** · to review: **307** · rejected: **109**

## Flags (a row can have several)

| flag | rows |
|---|---|
| math_pass | 223 |
| check_calendar_label | 58 |
| lang:expected_am | 57 |
| math_manual | 33 |
| math_wrong | 29 |
| truncated | 19 |
| repetition | 7 |
| phone_or_link_in_answer | 2 |
| lang:expected_en | 2 |
| html_tag | 1 |
| very_short | 1 |

## By category

| category | accepted | review | rejected |
|---|---|---|---|
| code_switch | 225 | 1 | 1 |
| creative | 144 | 0 | 1 |
| english_instruction | 192 | 1 | 12 |
| ethiopia_culture | 498 | 4 | 2 |
| everyday | 872 | 0 | 8 |
| format | 133 | 0 | 0 |
| knowledge_qa | 359 | 8 | 3 |
| latin_amharic | 168 | 0 | 0 |
| math | 220 | 36 | 29 |
| native_chat | 488 | 52 | 10 |
| news_classification | 97 | 2 | 1 |
| safety | 0 | 170 | 0 |
| summarization | 292 | 23 | 10 |
| translated_instruction | 222 | 0 | 3 |
| translation_am_en | 201 | 7 | 2 |
| translation_en_am | 231 | 3 | 26 |
| writing | 503 | 0 | 1 |

After review: rows marked `yes` in review_queue.csv (with `fixed_response` if you edited the answer) can be merged into sft_train.jsonl / eval_teacher.jsonl with `merge_review.py`.
