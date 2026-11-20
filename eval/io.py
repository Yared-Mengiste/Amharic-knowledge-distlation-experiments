"""Fail closed when benchmark IDs, identities, or paired settings disagree."""
import json
import math
from pathlib import Path

from eval.metrics import CONFIGURATIONS
from students.data import read_jsonl


def load_outputs(directory, prompts, require_complete=True):
    prompt_map = {r["id"]: r for r in prompts}
    outputs, metadata, coverage = {}, {}, {}
    for configuration in CONFIGURATIONS:
        path = Path(directory) / (configuration + ".jsonl")
        meta = json.loads(path.with_suffix(".meta.json").read_text())
        if meta["configuration"] != configuration or meta["do_sample"] or meta["max_new_tokens"] != 1000:
            raise ValueError(f"Unexpected evaluation settings: {configuration}")
        rows = read_jsonl(path)
        by_id = {r["id"]: r for r in rows}
        if set(by_id) - set(prompt_map):
            raise ValueError(f"Unknown benchmark IDs: {configuration}")
        for rid, row in by_id.items():
            if any(row[k] != meta[k] for k in ("configuration", "model_revision", "tokenizer_revision", "adapter_identifier")):
                raise ValueError(f"Output identity differs: {configuration}/{rid}")
            if any(row[k] != prompt_map[rid][k] for k in ("category", "language_mode")):
                raise ValueError(f"Output prompt metadata differs: {rid}")
            if not isinstance(row["answer"], str) or not 0 <= row["output_tokens"] <= 1000 or not math.isfinite(row["latency_seconds"]) or row["latency_seconds"] < 0:
                raise ValueError(f"Invalid benchmark output: {rid}")
            if row["termination_reason"] not in {"eos", "length", "unexpected_stop"}:
                raise ValueError(f"Unknown termination reason: {rid}")
        missing = sorted(set(prompt_map) - set(by_id))
        coverage[configuration] = {"outputs": len(rows), "expected": len(prompts), "missing_ids": missing}
        if require_complete and missing:
            raise ValueError(f"{configuration}: missing {len(missing)} outputs")
        outputs[configuration], metadata[configuration] = by_id, meta
    for model in ("llama", "gemma"):
        original, adapted = (metadata[model + suffix] for suffix in ("_original", "_adapted"))
        for field in original:
            if field not in {"configuration", "adapter_identifier"} and original[field] != adapted[field]:
                raise ValueError(f"{model} pair disagrees on {field}")
        if original["adapter_identifier"] is not None or not adapted["adapter_identifier"]:
            raise ValueError("Original/adapted adapter identities are invalid")
    if len({m["manifest_sha256"] for m in metadata.values()}) != 1:
        raise ValueError("The four configurations used different datasets")
    return outputs, metadata, coverage
