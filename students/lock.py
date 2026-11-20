"""Promote candidate pins only after both successful GPU smoke/resume tests."""
import argparse
import json
from pathlib import Path

from students.config import DEFAULT_LOCK, SETTINGS, read_lock
from students.data import sha256, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", default=str(DEFAULT_LOCK))
    parser.add_argument("--llama-smoke", required=True)
    parser.add_argument("--gemma-smoke", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    lock = read_lock(args.candidate)
    reports = {key: json.loads(Path(path).read_text()) for key, path in (("llama", args.llama_smoke), ("gemma", args.gemma_smoke))}
    for key, report in reports.items():
        identity = report["identity"]
        if not report["gpu_smoke_passed"] or not report["adapter_load_verified"] or report["resumed_from_step"] <= 0:
            raise ValueError(f"{key} smoke/resume/loading checks did not pass")
        if identity["model"] != key or identity["spec"] != lock["models"][key] or identity["settings"] != SETTINGS:
            raise ValueError(f"{key} smoke protocol/pins differ")
        if report["completed_steps"] != identity["smoke_steps"] or report["resumed_step_observed"] != report["resumed_from_step"]:
            raise ValueError(f"{key} smoke did not resume and complete")
    for field in ("packages", "manifest_sha256", "split_sha256", "precision"):
        if reports["llama"]["identity"][field] != reports["gemma"]["identity"][field]:
            raise ValueError(f"Smoke tests disagree on {field}")
    destination = Path(args.out)
    if destination.exists():
        raise ValueError("Never overwrite a verified lock")
    lock.update(gpu_smoke_verified=True, packages=reports["llama"]["identity"]["packages"],
                manifest_sha256=reports["llama"]["identity"]["manifest_sha256"],
                smoke_reports={"llama": sha256(args.llama_smoke), "gemma": sha256(args.gemma_smoke)})
    write_json(destination, lock)
    destination.with_name("requirements.lock.txt").write_text("".join(f"{name}=={version}\n" for name, version in sorted(lock["packages"].items())))


if __name__ == "__main__":
    main()
