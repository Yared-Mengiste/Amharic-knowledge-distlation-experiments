"""Build prompt-bound format proxies without inspecting any model answers."""
import argparse
import hashlib
import json
from pathlib import Path

from students.data import read_jsonl

# Shapes transcribed from the frozen pilot instructions. Semantic correctness is
# scored independently by native speakers. No generated answer influenced rules.
RULES = {
    "pilot_9de80ce8ab044379c818eca1": [{"kind": "regex", "pattern": r"1\. [^\n]+\n2\. [^\n]+\n3\. [^\n]+\n4\. [^\n]+\n?"}],
    "pilot_bc72dfc7d3ea4230a987c1b9": [{"kind": "json_type", "type": "object"}],
    "pilot_419eba8f92812425d354c6cf": [{"kind": "yaml_fields", "fields": ["name", "qty", "price"], "records": 3}],
    "pilot_3dee5de50ca8ad338d2052d9": [{"kind": "regex", "pattern": r"\s*\d+\s*;\s*\d+\s*;\s*\d+\s*"}],
    "pilot_c79a2be367d04dbd9e12a0b7": [{"kind": "json_keys", "keys": ["day", "month", "year"]}],
    "pilot_9c88b11dbcda4cb127fbeeea": [{"kind": "markdown_table", "columns": 2, "data_rows": 3}],
    "pilot_fe7c3f21c965bdc7da9ccdc2": [{"kind": "line_count", "count": 3}],
    "pilot_84e987c17f5dccfecae1aad8": [{"kind": "regex", "pattern": r"\s*[\d.]+\s*\n\s*[\d.]+\s*"}],
    "pilot_a74030ce651bb56a1242be30": [{"kind": "markdown_table", "columns": 2, "data_rows": 7}],
    "pilot_03a414d9d306fe3fcc8e8ab9": [{"kind": "regex", "pattern": r"(?:[^\n]+ — \d+(?:\.\d+)? ብር\n){3}[^\n]+ — \d+(?:\.\d+)? ብር\n?"}],
    "pilot_0fc2ef4df9ea665d8ec4b0f2": [{"kind": "python_syntax"}],
    "pilot_db8b8455283d498f15977060": [{"kind": "csv_columns", "columns": 2, "min_rows": 3, "max_rows": 4}],
    "pilot_f742b2c38f504e70d382a1cb": [{"kind": "regex", "pattern": r"(?:[^\s]+\n){4}[^\s]+\n?"}],
    "pilot_93648298163b1ac4bc25216e": [{"kind": "json_type", "type": "array"}],
    "pilot_b53828998f74b149a6949a8f": [{"kind": "regex", "pattern": r"ከፍተኛ፦ [^\n]+፣ ዝቅተኛ፦ [^\n]+\n?"}],
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompts", default="data/sft/pilot_eval.jsonl")
    parser.add_argument("--out", default=str(Path(__file__).with_name("format_constraints.jsonl")))
    args = parser.parse_args()
    prompts = {r["id"]: r for r in read_jsonl(args.prompts) if r["category"] == "format"}
    if set(prompts) != set(RULES):
        raise ValueError("These rules require the unchanged 15 pilot format IDs")
    rows = [{"id": rid, "prompt_sha256": hashlib.sha256(prompts[rid]["messages"][0]["content"].encode()).hexdigest(), "rules": RULES[rid], "scope": "Format/shape proxy only; content correctness is human-scored."} for rid in sorted(RULES)]
    Path(args.out).write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))


if __name__ == "__main__":
    main()
