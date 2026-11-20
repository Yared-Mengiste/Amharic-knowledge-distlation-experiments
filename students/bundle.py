"""Build a private Kaggle input bundle without credentials or unrelated data."""
import argparse
import shutil
import subprocess
from pathlib import Path

from students.data import load_frozen, sha256, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    load_frozen(args.dataset)
    root = Path(__file__).resolve().parents[1]
    output = Path(args.out)
    if output.exists():
        raise ValueError("Bundle already exists; choose a new version")
    output.mkdir(parents=True)
    # Allowlist only source used by the notebook; no .env, .git, or raw sources.
    for name in ("students", "eval", "teacher", "tests", "docs"):
        shutil.copytree(root / name, output / "repo" / name, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    (output / "repo/experiments").mkdir()
    shutil.copyfile(root / "experiments/LOG.md", output / "repo/experiments/LOG.md")
    shutil.copytree(args.dataset, output / "frozen")
    write_json(output / "bundle_manifest.json", {"git_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        "dataset_manifest_sha256": sha256(Path(args.dataset) / "manifest.json"),
        "files": {str(path.relative_to(output)): sha256(path) for path in sorted(output.rglob("*")) if path.is_file()},
        "visibility": "private; upload manually to a private Kaggle dataset"})


if __name__ == "__main__":
    main()
