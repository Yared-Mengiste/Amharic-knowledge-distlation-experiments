"""Verify a downloaded/restored private Kaggle output directory before resuming."""
import argparse
import json
from pathlib import Path

from students.data import sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", required=True)
    args = parser.parse_args()
    directory = Path(args.directory).resolve()
    index = json.loads((directory / "artifact_manifest.json").read_text())
    failures = []
    for name, digest in index["files"].items():
        path = directory / name
        if not path.resolve().is_relative_to(directory) or not path.is_file() or sha256(path) != digest:
            failures.append(name)
    if failures:
        raise SystemExit("Missing/changed private output artifacts: " + ", ".join(failures))
    print(f"Verified {len(index['files'])} output artifacts")


if __name__ == "__main__":
    main()
