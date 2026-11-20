"""Greedy four-configuration benchmark generation and approved-reference losses."""
import argparse
import json
import math
import time
from collections import defaultdict
from pathlib import Path

from students.config import SETTINGS, read_lock
from students.data import load_frozen, read_jsonl, sha256, write_json
from students.modeling import configure_generation, gpu_precision, load_model, load_tokenizer, package_versions
from students.tokenization import encode_conversation, render_prompt, template_hash
from students.train import assistant_ce


def reference_metrics(model, tokenizer, key, rows, precision):
    import torch
    totals = defaultdict(lambda: {"loss_sum": 0.0, "assistant_tokens": 0, "references": 0})
    per_id = []
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    with torch.no_grad(), torch.autocast("cuda", dtype=dtype):
        for row in rows:
            feature = encode_conversation(tokenizer, row, key)
            if len(feature["input_ids"]) > 2048:
                raise ValueError(f"Overlength eval reference: {row['id']}; no truncation allowed")
            batch = {k: torch.tensor([v], device="cuda") for k, v in feature.items()}
            outputs = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], use_cache=False)
            loss, count = assistant_ce(outputs.logits, batch["labels"])
            value, tokens = loss.item(), count.item()
            if not math.isfinite(value):
                raise ValueError(f"Nonfinite reference loss: {row['id']}")
            per_id.append({"id": row["id"], "category": row["category"], "loss_sum": value, "assistant_tokens": tokens})
            for category in (row["category"], "overall"):
                totals[category]["loss_sum"] += value
                totals[category]["assistant_tokens"] += tokens
                totals[category]["references"] += 1
    for metrics in totals.values():
        mean = metrics["loss_sum"] / metrics["assistant_tokens"]
        metrics.update(assistant_token_loss=mean, perplexity=math.exp(mean) if mean < 700 else None)
    return {"by_category": dict(totals), "per_id": per_id}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--lock", required=True)
    parser.add_argument("--model", required=True, choices=["llama", "gemma"])
    parser.add_argument("--adapter", help="Best full-run adapter directory; omit for original")
    parser.add_argument("--precision", choices=["auto", "fp16", "bf16"], default="auto")
    parser.add_argument("--out", required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    load_frozen(args.dataset)
    lock = read_lock(args.lock, verified=True)
    spec = lock["models"][args.model]
    prompts = read_jsonl(Path(args.dataset) / "pilot_eval.jsonl")
    refs = read_jsonl(Path(args.dataset) / "eval_teacher.jsonl")
    precision, gpu = gpu_precision(args.precision, training=False)
    versions = package_versions()
    if versions != lock["packages"]:
        raise ValueError("Evaluation packages differ from GPU-verified lock")
    tokenizer = load_tokenizer(spec)
    adapter_id = None
    if args.adapter:
        adapter_path = Path(args.adapter)
        identity = json.loads((adapter_path / "run_identity.json").read_text())
        if identity["spec"] != spec or identity["manifest_sha256"] != sha256(Path(args.dataset) / "manifest.json") or identity["smoke_steps"] is not None:
            raise ValueError("Adapter is not a full training run on this frozen dataset/model")
        if identity["template_sha256"] != template_hash(tokenizer) or identity["settings"] != SETTINGS:
            raise ValueError("Adapter chat template/training settings differ")
        adapter_id = sha256(adapter_path / "adapter_model.safetensors")
    configuration = args.model + ("_adapted" if args.adapter else "_original")
    output = Path(args.out)
    meta_path = output.with_suffix(".meta.json")
    identity = {"configuration": configuration, "model_id": spec["model_id"], "model_revision": spec["revision"],
                "tokenizer_revision": spec["tokenizer_revision"], "adapter_identifier": adapter_id,
                "template_sha256": template_hash(tokenizer), "template_settings": {"enable_thinking": False, "date_string": SETTINGS["date_string"]},
                "precision": precision, "quantization": "NF4-double", "max_new_tokens": 1000, "do_sample": False,
                "manifest_sha256": sha256(Path(args.dataset) / "manifest.json"), "packages": versions}
    previous = []
    if output.exists() or meta_path.exists():
        if not args.resume or not meta_path.exists() or json.loads(meta_path.read_text()) != identity:
            raise ValueError("Output already exists or evaluation identity differs; use explicit matching --resume")
        previous = read_jsonl(output) if output.exists() else []
        if any(row.get(k) != v for row in previous for k, v in identity.items() if k in ("configuration", "model_revision", "adapter_identifier", "tokenizer_revision")):
            raise ValueError("Existing rows have different model/adapter revisions")
        if not {r["id"] for r in previous} <= {r["id"] for r in prompts}:
            raise ValueError("Existing output contains unknown eval IDs")
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(meta_path, identity)
    import torch
    from peft import PeftModel
    from transformers import set_seed
    set_seed(42)
    model = load_model(spec, precision)
    if args.adapter:
        model = PeftModel.from_pretrained(model, args.adapter, is_trainable=False)
    model.eval()
    stops = configure_generation(model, tokenizer, args.model)
    done = {r["id"] for r in previous}
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    with output.open("a", encoding="utf-8") as handle, torch.no_grad(), torch.autocast("cuda", dtype=dtype):
        for row in prompts:
            if row["id"] in done:
                continue
            prompt = render_prompt(tokenizer, row["messages"])
            inputs = tokenizer(prompt, add_special_tokens=False, return_tensors="pt")
            if inputs["input_ids"].shape[1] > 2048:
                raise ValueError(f"Overlength eval prompt: {row['id']}")
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
            torch.cuda.synchronize()
            started = time.perf_counter()
            generated = model.generate(**inputs, do_sample=False, max_new_tokens=1000,
                                       eos_token_id=stops, pad_token_id=tokenizer.pad_token_id,
                                       use_cache=True)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            tokens = generated[0, inputs["input_ids"].shape[1]:].tolist()
            reason = "eos" if tokens and tokens[-1] in stops else "length" if len(tokens) == 1000 else "unexpected_stop"
            result = {"id": row["id"], "category": row["category"], "language_mode": row["language_mode"],
                      "configuration": configuration, "model_revision": spec["revision"],
                      "tokenizer_revision": spec["tokenizer_revision"], "adapter_identifier": adapter_id,
                      "answer": tokenizer.decode(tokens, skip_special_tokens=True), "output_tokens": len(tokens),
                      "latency_seconds": elapsed, "termination_reason": reason}
            handle.write(json.dumps(result, ensure_ascii=False, allow_nan=False) + "\n")
            handle.flush()
    reference = reference_metrics(model, tokenizer, args.model, refs, precision)
    missing = sorted({r["id"] for r in prompts} - {r["id"] for r in refs})
    reference.update(identity=identity, reference_coverage=len(refs) / len(prompts), missing_reference_ids=missing,
                     gpu=gpu, note="Perplexity can be compared only within each original/adapted tokenizer pair.")
    write_json(output.with_suffix(".references.json"), reference)


if __name__ == "__main__":
    main()
