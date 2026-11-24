"""Offline comparison report with coverage, paired human evidence and uncertainty."""
import argparse
import hashlib
import json
import math
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from eval.human import answer_hash
from eval.io import load_outputs
from eval.metrics import CONFIGURATIONS, automatic_metrics, summarize_metrics, wilson
from students.config import SETTINGS
from students.data import load_frozen, read_csv, read_jsonl, sha256, stratified_ids, write_json


def human_results(directory, outputs, prompts):
    directory = Path(directory)
    mapping = json.loads((directory / "private/mapping.json").read_text())
    prompt_map = {r["id"]: r for r in prompts}
    expected_samples = {"general": set(stratified_ids(prompts, 50)), "safety": {r["id"] for r in prompts if r["category"] == "safety"}}
    if mapping["seed"] != SETTINGS["seed"] or any(set(mapping["sample_ids"][sample]) != ids or len(mapping["sample_ids"][sample]) != len(ids) for sample, ids in expected_samples.items()):
        raise ValueError("Human sample IDs differ from the required stratified/safety selections")
    for sample, ids in expected_samples.items():
        expected_answers = {(rid, c) for rid in ids for c in CONFIGURATIONS}
        mapped_answers = [(r["prompt_id"], r["configuration"]) for r in mapping["answers"].values() if r["sample"] == sample]
        if set(mapped_answers) != expected_answers or len(mapped_answers) != len(expected_answers):
            raise ValueError("Human mapping lacks exactly one answer per configuration and prompt")
        expected_pairs = {(rid, frozenset(pair)) for rid in ids for pair in (("llama_original", "llama_adapted"), ("gemma_original", "gemma_adapted"), ("llama_adapted", "gemma_adapted"))}
        mapped_pairs = [(r["prompt_id"], frozenset((r["A"], r["B"]))) for r in mapping["pairs"].values() if r["sample"] == sample]
        if set(mapped_pairs) != expected_pairs or len(mapped_pairs) != len(expected_pairs):
            raise ValueError("Human mapping has missing/duplicate pairs")
    scores, pairs, missing = [], [], []
    seen_answers, seen_pairs = set(), set()
    for sample in ("general", "safety"):
        for row in read_csv(directory / "public" / (sample + "_scores.csv")):
            if row["id"] not in mapping["answers"]:
                raise ValueError("Unknown anonymized answer ID")
            identity = mapping["answers"][row["id"]]
            seen_answers.add(row["id"])
            rid = identity["prompt_id"]
            source = outputs[identity["configuration"]][rid]["answer"]
            if identity["sample"] != sample or row["prompt_id"] != rid or answer_hash(row["answer"]) != identity["answer_sha256"] or row["answer"] != source:
                raise ValueError(f"Human score mapped to the wrong answer: {row['id']}")
            if row["prompt"] != prompt_map[rid]["messages"][0]["content"] or row["category"] != prompt_map[rid]["category"]:
                raise ValueError("Human sheet prompt/category changed")
            dimensions = ["correctness", "fluency", "instruction_following"] + (["safety"] if sample == "safety" else [])
            if not row["reviewer"].strip() or any(not row[d].strip() for d in dimensions):
                missing.append(row["id"])
                continue
            values = {d: float(row[d]) for d in dimensions}
            if any(not math.isfinite(v) or v < 1 or v > 5 for v in values.values()):
                raise ValueError("Human scores must be finite and between 1 and 5")
            scores.append({**identity, "category": row["category"], "scores": values})
        for row in read_csv(directory / "public" / (sample + "_pairs.csv")):
            if row["id"] not in mapping["pairs"]:
                raise ValueError("Unknown anonymized pair ID")
            identity = mapping["pairs"][row["id"]]
            seen_pairs.add(row["id"])
            rid = identity["prompt_id"]
            if identity["sample"] != sample or row["prompt_id"] != rid:
                raise ValueError("Human pair prompt/sample changed")
            if row["prompt"] != prompt_map[rid]["messages"][0]["content"] or row["category"] != prompt_map[rid]["category"]:
                raise ValueError("Human pair prompt/category changed")
            for label in ("A", "B"):
                if answer_hash(row["answer_" + label]) != identity[label + "_sha256"] or row["answer_" + label] != outputs[identity[label]][rid]["answer"]:
                    raise ValueError("Human pair mapped to the wrong answer")
            winner = row["winner_A_B_tie"].strip().lower()
            if not winner or not row["reviewer"].strip():
                missing.append(row["id"])
                continue
            if winner not in {"a", "b", "tie"}:
                raise ValueError("Pair winner must be A/B/tie")
            pairs.append({**identity, "category": prompt_map[rid]["category"], "winner": "tie" if winner == "tie" else identity[winner.upper()]})
    if seen_answers != set(mapping["answers"]) or seen_pairs != set(mapping["pairs"]):
        raise ValueError("Human sheets have missing or extra anonymized rows")
    summary = {"complete": not missing, "missing_score_or_preference_ids": missing,
               "sample_ids": mapping["sample_ids"], "score_means": {}, "win_tie_loss": {}, "paired_score_differences": {}}
    for sample in ("general", "safety"):
        summary["score_means"][sample] = {}
        for config in CONFIGURATIONS:
            rows = [s for s in scores if s["sample"] == sample and s["configuration"] == config]
            by_category = {}
            for category in ["overall"] + sorted({r["category"] for r in prompts}):
                subset = rows if category == "overall" else [r for r in rows if r["category"] == category]
                by_category[category] = {"count": len(subset), "means": {d: statistics.mean(r["scores"][d] for r in subset) if subset else None for d in (["correctness", "fluency", "instruction_following"] + (["safety"] if sample == "safety" else []))}}
            summary["score_means"][sample][config] = by_category
        summary["win_tie_loss"][sample] = {}
        for original, adapted in (("llama_original", "llama_adapted"), ("gemma_original", "gemma_adapted"), ("llama_adapted", "gemma_adapted")):
            name = adapted + "_vs_" + original
            selected = [p for p in pairs if p["sample"] == sample and {p["A"], p["B"]} == {original, adapted}]
            counts = Counter("tie" if p["winner"] == "tie" else "win" if p["winner"] == adapted else "loss" for p in selected)
            summary["win_tie_loss"][sample][name] = {**{k: counts[k] for k in ("win", "tie", "loss")}, "win_rate_excluding_ties": wilson(counts["win"], counts["win"] + counts["loss"])}
            for dimension in ("correctness", "fluency", "instruction_following"):
                first = {r["prompt_id"]: r["scores"][dimension] for r in scores if r["sample"] == sample and r["configuration"] == original}
                second = {r["prompt_id"]: r["scores"][dimension] for r in scores if r["sample"] == sample and r["configuration"] == adapted}
                delta = [second[rid] - first[rid] for rid in sorted(set(first) & set(second))]
                rng = random.Random(42)
                samples = sorted(statistics.mean(rng.choices(delta, k=len(delta))) for _ in range(1000)) if delta else []
                summary["paired_score_differences"][sample + "/" + name + "/" + dimension] = {"paired_prompts": len(delta), "mean_adapted_minus_original": statistics.mean(delta) if delta else None, "bootstrap_ci95": [samples[24], samples[974]] if samples else None}
    return summary


def build_report(dataset, outputs_directory, math_path, constraints_path=None, human_directory=None, allow_partial=False):
    manifest, _ = load_frozen(dataset)
    prompts = read_jsonl(Path(dataset) / "pilot_eval.jsonl")
    refs = read_jsonl(Path(dataset) / "eval_teacher.jsonl")
    outputs, metadata, coverage = load_outputs(outputs_directory, prompts, require_complete=not allow_partial)
    if any(m["manifest_sha256"] != sha256(Path(dataset) / "manifest.json") for m in metadata.values()):
        raise ValueError("Benchmark outputs refer to a different frozen dataset")
    frozen_math = Path(dataset) / "math_answers.csv"
    if math_path and sha256(math_path) != sha256(frozen_math):
        raise ValueError("Math checks differ from the frozen dataset")
    math_checks = {r["id"]: r for r in read_csv(frozen_math)}
    # Reference-based checks use only approved reference IDs, including numeric checks.
    approved = {r["id"] for r in refs}
    math_checks = {rid: r for rid, r in math_checks.items() if rid in approved}
    constraint_rows = read_jsonl(constraints_path) if constraints_path else []
    constraints = {r["id"]: r["rules"] for r in constraint_rows}
    if set(constraints) - {r["id"] for r in prompts}:
        raise ValueError("Instruction constraints contain unknown eval IDs")
    prompt_map = {r["id"]: r for r in prompts}
    for row in constraint_rows:
        if row.get("prompt_sha256") and hashlib.sha256(prompt_map[row["id"]]["messages"][0]["content"].encode()).hexdigest() != row["prompt_sha256"]:
            raise ValueError("Instruction constraint was authored for a different prompt")
    result = {"manifest_sha256": sha256(Path(dataset) / "manifest.json"), "dataset_counts": manifest["counts"],
              "coverage": coverage, "evaluation_metadata": metadata,
              "reference_coverage": {"approved": len(refs), "expected": len(prompts), "missing_ids": sorted({r["id"] for r in prompts} - approved)},
              "automated": {}, "reference_loss": {}, "human": None,
              "artifacts": {},
              "notes": ["Script consistency is a heuristic, not language identification.", "Numeric presence checks reuse the supplied teacher checker; they can accept a number in incorrect reasoning and do not prove semantic math accuracy.", "Instruction constraints score only supplied explicit rules; missing rules remain unscored.", "Wilson 95% intervals describe measured proportions; human-score intervals are paired bootstrap intervals with seed 42.", "General and safety human samples are reported separately, including overlapping IDs.", "Compare perplexity within each tokenizer pair only."]}
    for config in CONFIGURATIONS:
        rows = list(outputs[config].values())
        scored = [(row, automatic_metrics(row, math_checks, constraints)) for row in rows]
        result["automated"][config] = {}
        for category in ["overall"] + sorted({r["category"] for r in prompts}):
            selected = scored if category == "overall" else [(r, m) for r, m in scored if r["category"] == category]
            metrics = summarize_metrics([m for _, m in selected])
            metrics.update(outputs=len(selected), median_latency_seconds=statistics.median(r["latency_seconds"] for r, _ in selected) if selected else None)
            result["automated"][config][category] = metrics
        loss_path = Path(outputs_directory) / (config + ".references.json")
        loss = json.loads(loss_path.read_text())
        if loss["identity"] != metadata[config] or {r["id"] for r in loss["per_id"]} != approved or len(loss["per_id"]) != len(approved):
            raise ValueError(f"Approved reference loss coverage/identity differs: {config}")
        result["reference_loss"][config] = loss
        result["artifacts"][config] = {"outputs_sha256": sha256(Path(outputs_directory) / (config + ".jsonl")), "references_sha256": sha256(loss_path)}
    if human_directory:
        result["human"] = human_results(human_directory, outputs, prompts)
    result["comparison_ready"] = all(not c["missing_ids"] for c in coverage.values()) and bool(result["human"] and result["human"]["complete"])
    return result


def markdown_report(result):
    lines = ["# Student comparison", "", f"Comparison ready: **{result['comparison_ready']}** (generation and human review).", "",
             f"Approved reference coverage: {result['reference_coverage']['approved']}/{result['reference_coverage']['expected']}.", "",
             "| Configuration | Outputs | Language proxy | Math check proxy | Truncated | Repeated |", "|---|---:|---:|---:|---:|---:|"]
    rate = lambda v: "unscored" if v["rate"] is None else f"{v['rate']:.1%} ({v['total']} scored)"
    for config in CONFIGURATIONS:
        metrics = result["automated"][config]["overall"]
        lines.append(f"| {config} | {metrics['outputs']} | {rate(metrics['language_script_consistency'])} | {rate(metrics['math_reference_check_proxy'])} | {rate(metrics['truncated'])} | {rate(metrics['repetition'])} |")
    lines += ["", "The companion JSON contains category results, 95% intervals, missing reference IDs, exact generation settings, and human score mappings/coverage.", ""]
    if result["human"]:
        lines += ["| Human comparison (general sample) | Wins | Ties | Losses |", "|---|---:|---:|---:|"]
        for name, counts in result["human"]["win_tie_loss"]["general"].items():
            lines.append(f"| {name} | {counts['win']} | {counts['tie']} | {counts['loss']} |")
        lines.append("")
    lines += ["Automated rates are proxies; correctness, fluency and instruction following require the blinded native-speaker scores.", "Perplexity comparisons apply only within each original/adapted tokenizer pair.", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--outputs", required=True)
    parser.add_argument("--math", help="Optional frozen-hash-matching math checks; defaults to snapshot")
    parser.add_argument("--constraints", default=str(Path(__file__).with_name("format_constraints.jsonl")))
    parser.add_argument("--human")
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--require-complete", action="store_true")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    result = build_report(args.dataset, args.outputs, args.math, args.constraints, args.human, args.allow_partial)
    write_json(args.out, result)
    Path(args.out).with_suffix(".md").write_text(markdown_report(result))
    if args.require_complete and not result["comparison_ready"]:
        raise SystemExit("Comparison is incomplete; inspect generation/human coverage in the report")


if __name__ == "__main__":
    main()
