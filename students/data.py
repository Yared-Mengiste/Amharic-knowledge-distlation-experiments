"""Dependency-free integrity checks, immutable snapshots and shared splits."""
import argparse
import csv
import hashlib
import json
import random
import shutil
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

from students.config import SETTINGS


def sha256(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def write_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def read_jsonl(path):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    ids = [r["id"] for r in rows]
    if len(set(ids)) != len(ids):
        raise ValueError(f"Duplicate IDs in {path}")
    return rows


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len({r["id"] for r in rows}) != len(rows):
        raise ValueError(f"Duplicate IDs in {path}")
    return rows


def write_csv(path, rows, fields):
    with Path(path).open("x", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def validate_conversation(row, reference=True):
    for key in ("id", "category", "language_mode"):
        if not isinstance(row.get(key), str) or not row[key].strip():
            raise ValueError(f"Missing {key}: {row.get('id')}")
    expected = ["user", "assistant"] if reference else ["user"]
    if [m.get("role") for m in row["messages"]] != expected:
        raise ValueError(f"{row['id']}: expected {expected}; system/multi-turn data are unsupported")
    if any(not isinstance(m.get("content"), str) or not m["content"].strip() for m in row["messages"]):
        raise ValueError(f"{row['id']}: empty message")


def stratified_ids(rows, count, seed=42):
    """Largest-remainder allocation, at least one per category, stable under row order."""
    groups = defaultdict(list)
    for row in rows:
        groups[row["category"]].append(row["id"])
    if not len(groups) <= count <= len(rows):
        raise ValueError("Sample size must cover every category and fit the dataset")
    allocations = {c: max(1, int(count * len(ids) / len(rows))) for c, ids in groups.items()}
    while sum(allocations.values()) != count:
        direction = 1 if sum(allocations.values()) < count else -1
        eligible = [c for c in sorted(groups) if (allocations[c] < len(groups[c]) if direction == 1 else allocations[c] > 1)]
        chosen = max(eligible, key=lambda c: direction * (count * len(groups[c]) / len(rows) - allocations[c]))
        allocations[chosen] += direction
    rng = random.Random(seed)
    selected = []
    for category in sorted(groups):
        ids = sorted(groups[category])
        rng.shuffle(ids)
        selected.extend(ids[:allocations[category]])
    return sorted(selected)


def split_ids(rows):
    validation = stratified_ids(rows, round(len(rows) * SETTINGS["validation_fraction"]), SETTINGS["seed"])
    train = sorted({r["id"] for r in rows} - set(validation))
    if {r["category"] for r in rows if r["id"] in train} != {r["category"] for r in rows}:
        raise ValueError("Every category needs examples in training AND validation")
    return {"seed": 42, "train_ids": train, "validation_ids": validation}


def audit(args):
    train, refs, prompts = map(read_jsonl, (args.train, args.references, args.prompts))
    review, rejected = read_csv(args.review), read_jsonl(args.rejected)
    for r in train + refs:
        validate_conversation(r)
    for r in prompts:
        validate_conversation(r, False)
    errors = []
    accepted = {r["id"]: r for r in train + refs}
    train_ids, ref_ids, eval_ids = ({r["id"] for r in rs} for rs in (train, refs, prompts))
    if train_ids & (ref_ids | eval_ids) or not ref_ids <= eval_ids:
        errors.append("Training/validation/eval IDs overlap or reference IDs are not eval IDs")
    if len(prompts) != 300 or sum(r["category"] == "safety" for r in prompts) != 20:
        errors.append("Expected all 300 eval prompts, including exactly 20 safety prompts")
    prompt_text = lambda r: " ".join(r["messages"][0]["content"].split()).casefold()
    if {prompt_text(r) for r in train} & {prompt_text(r) for r in prompts}:
        errors.append("Training and eval have identical normalized prompt content")
    eval_map = {r["id"]: r for r in prompts}
    for r in refs:
        if r["id"] in eval_map and any(r[k] != eval_map[r["id"]][k] for k in ("category", "language_mode")):
            errors.append(f"Reference metadata differs: {r['id']}")
        if r["id"] in eval_map and r["messages"][:-1] != eval_map[r["id"]]["messages"]:
            errors.append(f"Reference prompt differs: {r['id']}")
    dispositions = read_csv(args.dispositions) if Path(args.dispositions).exists() else []
    triage = {r["id"]: r for r in dispositions}
    needs_triage = {r["id"] for r in rejected}
    undecided = []
    for r in review:
        decision = r.get("decision_keep_yes_no", "").strip().lower()
        if decision not in {"yes", "no"}:
            undecided.append(r["id"])
        if r.get("fixed_response", "").strip() and decision != "yes":
            errors.append(f"Corrected answer needs a yes decision: {r['id']}")
        if decision == "yes":
            expected = r.get("fixed_response", "").strip() or r["response"].strip()
            if r["id"] not in accepted or accepted[r["id"]]["messages"][-1]["content"].strip() != expected:
                errors.append(f"Approved answer not merged correctly: {r['id']}")
        elif decision == "no":
            needs_triage.add(r["id"])
            if r["id"] in accepted:
                errors.append(f"No-decision answer remains accepted: {r['id']}")
    if undecided:
        errors.append(f"{len(undecided)} review decisions are not yes/no")
    for rid in sorted(needs_triage):
        row = triage.get(rid, {})
        disposition = row.get("disposition", "").strip().lower()
        if disposition not in {"corrected", "regenerated", "excluded"} or not row.get("notes", "").strip():
            errors.append(f"Missing documented disposition: {rid}")
        elif (disposition == "excluded") == (rid in accepted):
            errors.append(f"Disposition contradicts accepted dataset: {rid}")
    unknown = set(triage) - needs_triage
    if unknown:
        errors.append(f"Disposition IDs do not match rejected/no rows: {sorted(unknown)}")
    spot = read_csv(args.spot_check) if Path(args.spot_check).exists() else []
    if len(spot) < 100:
        errors.append(f"Need approximately 100 native-speaker spot checks; found {len(spot)}")
    for row in spot:
        source = accepted.get(row["id"])
        if row["id"] not in train_ids or not row.get("reviewer", "").strip() or row.get("accepted_yes_no", "").strip().lower() != "yes":
            errors.append(f"Incomplete/unresolved training spot check: {row['id']}")
        if source and row.get("answer_sha256") != hashlib.sha256(source["messages"][-1]["content"].encode()).hexdigest():
            errors.append(f"Spot check refers to an older answer: {row['id']}")
    # Last teacher write wins, exactly as the existing generator and merge tool specify.
    teacher = {}
    for line in Path(args.teacher).read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            teacher[r["id"]] = r
    reviewed = {r["id"]: r for r in review}
    for rid, row in accepted.items():
        source = teacher.get(rid)
        if source is None:
            errors.append(f"Accepted ID missing from teacher outputs: {rid}")
            continue
        if (source.get("split") == "train") != (rid in train_ids):
            errors.append(f"Accepted ID in wrong split: {rid}")
        if row["messages"][:-1] != source["messages"] or any(row[k] != source[k] for k in ("category", "language_mode")):
            errors.append(f"Accepted prompt/metadata changed: {rid}")
        correction = reviewed.get(rid, {}).get("fixed_response", "").strip()
        disposition = triage.get(rid, {}).get("disposition")
        if disposition == "corrected" and not correction:
            errors.append(f"Corrected rejection must be added to review queue with complete fixed_response and yes: {rid}")
        if disposition not in {"corrected", "regenerated"} and not correction and row["messages"][-1]["content"].strip() != source["response"].strip():
            errors.append(f"Undocumented answer change: {rid}")
    accounted = set(accepted) | {r["id"] for r in review} | {r["id"] for r in rejected}
    if accounted != set(teacher):
        errors.append("Teacher IDs are missing from, or extra in, accepted/review/rejected data")
    paths = {"sft_train.jsonl": args.train, "eval_teacher.jsonl": args.references,
             "pilot_eval.jsonl": args.prompts, "review_queue.csv": args.review,
             "rejected.jsonl": args.rejected, "rejected_dispositions.csv": args.dispositions,
             "spot_check.csv": args.spot_check, "teacher_outputs.jsonl": args.teacher}
    report = {"ready": not errors, "errors": errors, "undecided_review_ids": undecided,
              "counts": {"train": len(train), "references": len(refs), "eval": len(prompts)},
              "category_counts": {name: dict(Counter(r["category"] for r in rows)) for name, rows in (("train", train), ("references", refs), ("eval", prompts))},
              "review_decisions": dict(Counter(r.get("decision_keep_yes_no", "").strip().lower() for r in review)),
              "dispositions": dict(Counter(r.get("disposition", "") for r in dispositions)),
              "missing_reference_ids": sorted(eval_ids - ref_ids),
              "files": {name: {"source": str(path), "sha256": sha256(path)} for name, path in paths.items() if Path(path).exists()}}
    return report, train, paths


def load_frozen(directory):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    if not manifest["ready"] or manifest["settings"] != SETTINGS:
        raise ValueError("Snapshot is not approved or training settings differ")
    for name, spec in manifest["files"].items():
        if sha256(directory / name) != spec["sha256"]:
            raise ValueError(f"Frozen file changed: {name}")
    split = json.loads((directory / "split.json").read_text())
    if sha256(directory / "split.json") != manifest["split_sha256"]:
        raise ValueError("Frozen split changed")
    rows = read_jsonl(directory / "sft_train.jsonl")
    train, val = set(split["train_ids"]), set(split["validation_ids"])
    if train & val or train | val != {r["id"] for r in rows}:
        raise ValueError("Invalid shared split")
    return manifest, split


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["audit", "prepare-review", "freeze"])
    defaults = {"train": "data/sft/sft_train.jsonl", "references": "data/sft/eval_teacher.jsonl", "prompts": "data/prompts/pilot_eval.jsonl", "review": "data/sft/review_queue.csv", "rejected": "data/sft/rejected.jsonl", "teacher": "data/teacher/teacher_outputs.jsonl", "dispositions": "data/review/rejected_dispositions.csv", "spot-check": "data/review/spot_check.csv"}
    for key, default in defaults.items():
        parser.add_argument("--" + key, default=default)
    parser.add_argument("--out", default="outputs/bakeoff/frozen")
    args = parser.parse_args()
    if args.command == "prepare-review":
        for path in (args.dispositions, args.spot_check):
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        review = read_csv(args.review)
        ids = {r["id"] for r in read_jsonl(args.rejected)} | {r["id"] for r in review if r["decision_keep_yes_no"].strip().lower() == "no"}
        write_csv(args.dispositions, [{"id": rid, "disposition": "", "notes": ""} for rid in sorted(ids)], ["id", "disposition", "notes"])
        train = read_jsonl(args.train)
        chosen = set(stratified_ids(train, 100))
        rows = [{"id": r["id"], "category": r["category"], "prompt": r["messages"][0]["content"], "answer": r["messages"][-1]["content"], "answer_sha256": hashlib.sha256(r["messages"][-1]["content"].encode()).hexdigest(), "reviewer": "", "accepted_yes_no": "", "notes": ""} for r in train if r["id"] in chosen]
        write_csv(args.spot_check, rows, ["id", "category", "prompt", "answer", "answer_sha256", "reviewer", "accepted_yes_no", "notes"])
        return
    report, train, paths = audit(args)
    if args.command == "audit":
        print(json.dumps(report, ensure_ascii=False, indent=2))
    elif report["ready"]:
        destination = Path(args.out)
        if destination.exists():
            raise ValueError("Snapshot already exists; choose a new version, never overwrite")
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=destination.parent) as scratch:
            snapshot = Path(scratch) / "snapshot"
            snapshot.mkdir()
            for name, path in paths.items():
                shutil.copyfile(path, snapshot / name)
                if sha256(snapshot / name) != report["files"][name]["sha256"]:
                    raise ValueError(f"Source changed while freezing: {path}")
            write_json(snapshot / "split.json", split_ids(train))
            report.update(settings=SETTINGS, split_sha256=sha256(snapshot / "split.json"))
            write_json(snapshot / "manifest.json", report)
            load_frozen(snapshot)
            snapshot.rename(destination)
        print(f"Frozen dataset: {destination}")
    else:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["ready"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
