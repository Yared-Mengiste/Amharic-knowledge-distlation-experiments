"""Promote pins for the models with successful GPU smoke/resume tests."""
import argparse
import json
from pathlib import Path

from students.config import DEFAULT_LOCK, SETTINGS, read_lock
from students.data import sha256, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", default=str(DEFAULT_LOCK))
    parser.add_argument("--llama-smoke", required=True)
    parser.add_argument("--gemma-smoke")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    lock = read_lock(args.candidate)
    paths = {"llama": args.llama_smoke}
    if args.gemma_smoke:
        paths["gemma"] = args.gemma_smoke
    if set(paths) - set(lock["models"]):
        raise ValueError("Smoke report model is absent from candidate lock")
    reports = {key: json.loads(Path(path).read_text()) for key, path in paths.items()}
    for key, report in reports.items():
        identity = report["identity"]
        if not report["gpu_smoke_passed"] or not report["adapter_load_verified"] or report["resumed_from_step"] <= 0:
            raise ValueError(f"{key} smoke/resume/loading checks did not pass")
        if identity["model"] != key or identity["spec"] != lock["models"][key] or identity["settings"] != SETTINGS:
            raise ValueError(f"{key} smoke protocol/pins differ")
        if report["completed_steps"] != identity["smoke_steps"] or report["resumed_step_observed"] != report["resumed_from_step"]:
            raise ValueError(f"{key} smoke did not resume and complete")
    identity = reports["llama"]["identity"]
    for field in ("settings", "packages", "manifest_sha256", "split_sha256", "precision"):
        if any(report["identity"][field] != identity[field] for report in reports.values()):
            raise ValueError(f"Smoke tests disagree on {field}")
    destination = Path(args.out)
    requirements = destination.with_name("requirements.lock.txt")
    if destination.exists() or requirements.exists():
        raise ValueError("Never overwrite a verified lock or package pins")
    lock.update(gpu_smoke_verified=True, verified_models=list(reports),
                packages=identity["packages"], manifest_sha256=identity["manifest_sha256"],
                split_sha256=identity["split_sha256"], precision=identity["precision"],
                smoke_reports={key: sha256(path) for key, path in paths.items()})
    lock["note"] = "Only verified_models passed GPU smoke/resume/loading; other candidate entries remain unverified."
    write_json(destination, lock)
    requirements.write_text("".join(f"{name}=={version}\n" for name, version in sorted(lock["packages"].items())))


if __name__ == "__main__":
    main()
