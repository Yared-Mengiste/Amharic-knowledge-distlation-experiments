"""Preflight each locked tokenizer; report every overlength or masking failure."""
import argparse
import importlib.metadata
from pathlib import Path

from students.config import DEFAULT_LOCK, SETTINGS, read_lock
from students.data import load_frozen, read_jsonl, sha256, write_json
from students.modeling import load_tokenizer
from students.tokenization import encode_conversation, render_prompt, template_hash


def run(dataset, lock, output, unreviewed=False, prompts=None, summary=None):
    if not unreviewed:
        load_frozen(dataset)
    result = {"manifest_sha256": None if unreviewed else sha256(Path(dataset) / "manifest.json"),
              "models": {}, "passed": not unreviewed, "tokenization_passed": True,
              "unreviewed_diagnostic": unreviewed}
    for key, spec in lock["models"].items():
        tokenizer = load_tokenizer(spec)
        lengths, failures, overlength = {}, [], []
        for name in ("sft_train.jsonl", "eval_teacher.jsonl"):
            for row in read_jsonl(Path(dataset) / name):
                try:
                    feature = encode_conversation(tokenizer, row, key)
                    lengths[row["id"]] = len(feature["input_ids"])
                    if len(feature["input_ids"]) > SETTINGS["max_length"]:
                        overlength.append({"id": row["id"], "source": name, "tokens": len(feature["input_ids"])})
                except ValueError as error:
                    failures.append({"id": row["id"], "source": name, "reason": str(error)})
        for row in read_jsonl(prompts if unreviewed else Path(dataset) / "pilot_eval.jsonl"):
            prompt = render_prompt(tokenizer, row["messages"])
            length = len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
            if length > SETTINGS["max_length"]:
                overlength.append({"id": row["id"], "source": "pilot_eval.jsonl", "tokens": length})
        result["models"][key] = {"spec": spec, "template_sha256": template_hash(tokenizer),
                                 "lengths": lengths, "overlength": overlength, "masking_failures": failures,
                                 "maximum_tokens": max(lengths.values(), default=0)}
        result["tokenization_passed"] &= not overlength and not failures
    result["passed"] &= result["tokenization_passed"]
    write_json(output, result)
    if summary:
        from datetime import datetime, timezone
        inputs = [Path(dataset) / name for name in ("sft_train.jsonl", "eval_teacher.jsonl")]
        inputs.append(Path(prompts) if unreviewed else Path(dataset) / "pilot_eval.jsonl")
        write_json(summary, {"date": datetime.now(timezone.utc).date().isoformat(),
            "unreviewed_diagnostic": unreviewed, "training_gate_passed": result["passed"],
            "tokenization_passed": result["tokenization_passed"],
            "models": {key: {field: model[field] for field in ("spec", "template_sha256", "maximum_tokens", "overlength", "masking_failures")} | {"reference_and_train_rows": len(model["lengths"])} for key, model in result["models"].items()},
            "inputs": {str(path): sha256(path) for path in inputs},
            "packages": {name: importlib.metadata.version(name) for name in ("transformers", "tokenizers", "jinja2")},
            "cost_usd": 0, "gpu_smoke_performed": False,
            "note": "Tokenizer checks only. Reviewed dataset preflight must be repeated after freeze."})
    if not result["tokenization_passed"]:
        raise ValueError(f"Preflight failed; inspect all IDs in {output}. No rows were truncated or removed.")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--lock", default=str(DEFAULT_LOCK))
    parser.add_argument("--out", required=True)
    parser.add_argument("--unreviewed-diagnostic", action="store_true", help="Tokenizer check only; cannot authorize training")
    parser.add_argument("--prompts", default="data/sft/pilot_eval.jsonl")
    parser.add_argument("--summary", help="Optional compact, reproducible diagnostic summary")
    args = parser.parse_args()
    run(args.dataset, read_lock(args.lock), args.out, args.unreviewed_diagnostic, args.prompts, args.summary)


if __name__ == "__main__":
    main()
