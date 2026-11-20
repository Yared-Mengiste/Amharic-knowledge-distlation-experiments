# Student evaluation

`python3 -m students.evaluate` generates all 300 benchmark answers per original or
adapted student and measures assistant-token loss on approved references.

`python3 -m eval.human` prepares blind individual 1–5 rating sheets and pairwise
preferences on 50 stratified prompts, plus separate sheets for all 20 safety
prompts. Give reviewers only `public/`; retain `private/mapping.json` with the lead.

`python3 -m eval.report` checks all four output sets, original/adapted settings,
reference coverage and returned human-score mappings. It reports categories,
Wilson intervals for rates and paired bootstrap intervals for human-score changes.
No command makes judge API calls.

Math numeric-presence checks and script consistency are explicitly labeled proxies.
Semantic correctness, fluency and instruction following come from native speakers.
To measure an explicit formatting/instruction rule, supply `--constraints` JSONL:

```json
{"id":"an-existing-eval-id","rules":[{"kind":"json_keys","keys":["name","age"]}]}
```

Supported kinds: `json`, `json_keys`, `line_count` (`count`), `bullet_count` (`count`),
`required_text`/`forbidden_text` (`texts`) and `regex` (`pattern`, full match).
Derive rules from prompt instructions before seeing model answers. Missing rules
remain unscored; do not infer semantic compliance from shape or script alone.

See [the run guide](../docs/RUN_student_bakeoff.md) for exact commands.
