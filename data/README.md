# data/

| Folder | Contents | Committed? |
|---|---|---|
| `raw/` | Original pilot inputs from the team member. Read-only. | Yes |
| `work/` | Existing intermediate files written by `data_prep/`. | Yes |
| `sft/` | Completed SFT dataset, evaluation prompts and required audit evidence; see [sft/README.md](sft/README.md). | Yes |

The completed pilot is retained in `data/sft/`: 4,952 training conversations
and 299 teacher evaluation references, plus all 300 evaluation prompts and the
evidence required by the student audit. Intermediate `data/prompts/`,
`data/teacher/` and `data/review/` folders have been removed. Native-speaker
spot checks remain pending; preparing this dataset does not record their approval.

For a new teacher run, `bash data_prep/run_all.sh` recreates the prompt folder.
Regenerated prompt, teacher-response and review folders are ignored by Git.
The existing source data and preparation scripts remain available.

Files over about 50 MB go to a private Hugging Face repo or Git LFS instead;
record the link in `experiments/LOG.md`. Keep the repo private until source
licences have been checked.
