"""Final deliverable gate: comparison, two full runs, local quality, and log entries."""
import argparse
import json
import math
from pathlib import Path

from eval.human import answer_hash
from students.config import SETTINGS
from students.data import read_csv, read_jsonl, sha256, write_json
from students.train import verify_resume


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True)
    parser.add_argument("--llama-run", required=True)
    parser.add_argument("--gemma-run", required=True)
    parser.add_argument("--llama-deploy", required=True)
    parser.add_argument("--gemma-deploy", required=True)
    parser.add_argument("--log", default="experiments/LOG.md")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    report = json.loads(Path(args.report).read_text())
    if not report["comparison_ready"]:
        raise ValueError("Benchmark generation or blind native-speaker review is incomplete")
    log = Path(args.log).read_text()
    if sha256(args.report) not in log:
        raise ValueError("Final comparison hash is not recorded in the experiment log")
    if any(artifact["references_sha256"] not in log for artifact in report["artifacts"].values()):
        raise ValueError("All four evaluation reference-metric artifacts must be recorded in the log")
    runs, deployments, splits = {}, {}, []
    for key, run_path, deploy_path in (("llama", args.llama_run, args.llama_deploy), ("gemma", args.gemma_run, args.gemma_deploy)):
        run, deployment = Path(run_path), Path(deploy_path)
        metrics = json.loads((run / "metrics.json").read_text())
        identity = metrics["identity"]
        evaluation = report["evaluation_metadata"][key + "_adapted"]
        if any(evaluation[field] != identity["spec"][source] for field, source in (("model_id", "model_id"), ("model_revision", "revision"), ("tokenizer_revision", "tokenizer_revision"))):
            raise ValueError("Evaluation base/tokenizer revisions differ from the training run")
        split = json.loads((run / "split.json").read_text())
        splits.append(split)
        if sha256(run / "split.json") != identity["split_sha256"] or set(split["train_ids"]) & set(split["validation_ids"]):
            raise ValueError("Training split changed or overlaps validation")
        if identity["model"] != key or identity["settings"] != SETTINGS or identity["smoke_steps"] is not None:
            raise ValueError("Training identity/settings are not the full fixed protocol")
        if identity["manifest_sha256"] != report["manifest_sha256"] or metrics["completed_steps"] != math.ceil(len(split["train_ids"]) / 16) * 2 or not metrics["adapter_load_verified"]:
            raise ValueError("Training did not complete with the evaluated snapshot and loadable adapter")
        if not math.isfinite(metrics["best_validation_loss"]) or (key == "llama" and metrics["adapter_parameters"] != 11_272_192):
            raise ValueError("Validation/adapters fail acceptance")
        if not metrics["finite_gradient_steps"] or not metrics["adapter_parameters"]:
            raise ValueError("Training gradient/adapter checks are missing")
        verify_resume(run / f"checkpoint-{metrics['completed_steps']}", identity)
        adapter_hash = sha256(run / "adapter/adapter_model.safetensors")
        if adapter_hash != report["evaluation_metadata"][key + "_adapted"]["adapter_identifier"]:
            raise ValueError("Comparison evaluated a different adapter")
        local = json.loads((deployment / "metrics.json").read_text())
        if local["quantization"] != "Q4_K_M" or local["identity"]["adapter_sha256"] != adapter_hash or local["identity"]["spec"] != identity["spec"]:
            raise ValueError("Deployment quantization/base/adapter differs")
        if sha256(deployment / "deployment_prompts.jsonl") != local["identity"]["deployment_prompts_sha256"]:
            raise ValueError("Local deployment prompt file changed")
        answers = {r["id"]: r for r in read_jsonl(deployment / "answers.jsonl")}
        reviewed = read_csv(deployment / "quality_review.csv")
        if set(answers) != {r["id"] for r in reviewed}:
            raise ValueError("Deployment quality review coverage differs")
        for row in reviewed:
            if answer_hash(answers[row["id"]]["answer"]) != row["answer_sha256"] or row["answer"] != answers[row["id"]]["answer"] or not row["reviewer"].strip():
                raise ValueError("Deployment quality score mapped to the wrong/unreviewed answer")
            if row["prompt"] != answers[row["id"]]["prompt"] or row["category"] != answers[row["id"]]["category"]:
                raise ValueError("Deployment quality-review prompt/category changed")
            for d in ("correctness", "fluency", "instruction_following"):
                score = float(row[d])
                if not math.isfinite(score) or not 1 <= score <= 5:
                    raise ValueError("Deployment quality scores must be 1–5")
        for artifact in (run / "metrics.json", deployment / "metrics.json"):
            if sha256(artifact) not in log:
                raise ValueError(f"Artifact not recorded in experiment log: {artifact}")
        runs[key], deployments[key] = metrics, local
    if splits[0] != splits[1]:
        raise ValueError("Students trained on different training/validation IDs")
    if deployments["llama"]["identity"]["deployment_prompts_sha256"] != deployments["gemma"]["identity"]["deployment_prompts_sha256"]:
        raise ValueError("Local deployment prompts differ")
    if deployments["llama"]["server_command"][-1] != deployments["gemma"]["server_command"][-1]:
        raise ValueError("Local deployment GPU-layer settings differ")
    write_json(args.out, {"accepted": True, "comparison_sha256": sha256(args.report),
        "runs": runs, "deployments": deployments, "log_sha256": sha256(args.log),
        "note": "Acceptance verifies complete comparable evidence; it does not assume adaptation improved quality. Assess held-out category/human results."})


if __name__ == "__main__":
    main()
