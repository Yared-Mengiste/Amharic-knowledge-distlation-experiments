import argparse
import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from students.config import SETTINGS, read_lock
from students.data import audit, load_frozen, sha256, split_ids, stratified_ids, write_csv, write_json
from students.tokenization import encode_conversation, padded_features


class FakeTokenizer:
    all_special_tokens = ["<|eot_id|>", "<turn|>"]

    def __init__(self, ending="<|eot_id|>"):
        self.ending = ending

    def apply_chat_template(self, messages, tokenize, add_generation_prompt, **kwargs):
        text = "USER:" + messages[0]["content"].strip() + ":ASSISTANT:"
        if len(messages) == 2:
            text += messages[1]["content"].strip() + self.ending + "\n"
        return text

    def convert_tokens_to_ids(self, token):
        return 999

    def __call__(self, text, **kwargs):
        ids, offsets, i = [], [], 0
        while i < len(text):
            width = len(self.ending) if text.startswith(self.ending, i) else 1
            ids.append(999 if width > 1 else ord(text[i]))
            offsets.append((i, i + width))
            i += width
        return {"input_ids": ids, "offset_mapping": offsets}


def conversation(rid, category="chat"):
    return {"id": rid, "category": category, "language_mode": "am", "messages": [{"role": "user", "content": "Question " + rid}, {"role": "assistant", "content": "Answer " + rid}]}


class MaskingTests(unittest.TestCase):
    def test_user_and_post_ending_masked_padding_eos_preserved(self):
        row = conversation("a")
        feature = encode_conversation(FakeTokenizer(), row, "llama")
        labels = feature["labels"]
        self.assertEqual([x for x in labels if x != -100], [ord(c) for c in "Answer a"] + [999])
        self.assertEqual(labels[-1], -100)
        short = encode_conversation(FakeTokenizer(), conversation(""), "llama")
        padded = padded_features([feature, short], 999)
        self.assertEqual(padded["labels"][1][-1], -100)
        self.assertIn(999, padded["labels"][1])

    def test_gemma_turn_ending_and_thinking_disabled(self):
        tokenizer = FakeTokenizer("<turn|>")
        feature = encode_conversation(tokenizer, conversation("a"), "gemma")
        self.assertEqual([x for x in feature["labels"] if x != -100][-1], 999)

    def test_missing_ending_reserved_tokens_and_system_fail(self):
        with self.assertRaises(ValueError):
            encode_conversation(FakeTokenizer("BAD"), conversation("a"), "llama")
        row = conversation("a")
        row["messages"][1]["content"] += "<|eot_id|>"
        with self.assertRaises(ValueError):
            encode_conversation(FakeTokenizer(), row, "llama")
        row["messages"][0]["role"] = "system"
        with self.assertRaises(ValueError):
            encode_conversation(FakeTokenizer(), row, "llama")


class DataTests(unittest.TestCase):
    def test_stratification_exact_and_order_independent(self):
        rows = [conversation(str(i), str(i % 7)) for i in range(503)]
        first = split_ids(rows)
        self.assertEqual(first, split_ids(list(reversed(rows))))
        self.assertEqual(len(first["validation_ids"]), 50)
        self.assertFalse(set(first["train_ids"]) & set(first["validation_ids"]))
        self.assertEqual(len(stratified_ids(rows, 50)), 50)

    def test_candidate_is_not_full_training_authorization(self):
        self.assertFalse(read_lock()["gpu_smoke_verified"])
        with self.assertRaises(ValueError):
            read_lock(verified=True)

    def fixture(self, root):
        train = [conversation("train" + str(i)) for i in range(120)]
        refs = [conversation("eval" + str(i), "safety" if i < 20 else "chat") for i in range(300)]
        prompts = [{**r, "messages": r["messages"][:-1], "split": "eval"} for r in refs]
        teacher = [{**r, "messages": r["messages"][:-1], "response": r["messages"][-1]["content"], "split": "train" if r["id"].startswith("train") else "eval"} for r in train + refs]
        for name, rows in (("train", train), ("references", refs), ("prompts", prompts), ("teacher", teacher), ("rejected", [])):
            (root / (name + ".jsonl")).write_text("".join(json.dumps(r) + "\n" for r in rows))
        write_csv(root / "review.csv", [], ["id", "decision_keep_yes_no", "fixed_response", "response"])
        write_csv(root / "dispositions.csv", [], ["id", "disposition", "notes"])
        spots = [{"id": r["id"], "reviewer": "Native speaker", "accepted_yes_no": "yes", "answer_sha256": hashlib.sha256(r["messages"][-1]["content"].encode()).hexdigest()} for r in train[:100]]
        write_csv(root / "spot.csv", spots, ["id", "reviewer", "accepted_yes_no", "answer_sha256"])
        write_csv(root / "math.csv", [], ["id", "status", "computed_answer"])
        return argparse.Namespace(**{name: root / (name + ".jsonl") for name in ("train", "references", "prompts", "teacher", "rejected")}, review=root / "review.csv", dispositions=root / "dispositions.csv", spot_check=root / "spot.csv", math=root / "math.csv")

    def test_ready_snapshot_hashes_and_tamper(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = self.fixture(root)
            report, train, paths = audit(args)
            self.assertTrue(report["ready"], report["errors"])
            for name, path in paths.items():
                (root / name).write_bytes(path.read_bytes())
            write_json(root / "split.json", split_ids(train))
            report.update(settings=SETTINGS, split_sha256=sha256(root / "split.json"))
            write_json(root / "manifest.json", report)
            load_frozen(root)
            (root / "sft_train.jsonl").write_text("tampered")
            with self.assertRaisesRegex(ValueError, "Frozen file changed"):
                load_frozen(root)

    def test_incomplete_reviews_and_unmerged_corrections_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = self.fixture(root)
            args.review.write_text("id,decision_keep_yes_no,fixed_response,response\ntrain0,,,Answer train0\n")
            report, _, _ = audit(args)
            self.assertFalse(report["ready"])
            self.assertEqual(report["undecided_review_ids"], ["train0"])
            args.review.write_text("id,decision_keep_yes_no,fixed_response,response\ntrain0,yes,Corrected,Answer train0\n")
            report, _, _ = audit(args)
            self.assertTrue(any("not merged" in e for e in report["errors"]))

    def test_stale_spotcheck_and_eval_leakage_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = self.fixture(root)
            args.spot_check.write_text(args.spot_check.read_text(encoding="utf-8-sig").replace(",yes,", ",no,", 1))
            rows = [json.loads(line) for line in args.train.read_text().splitlines()]
            rows[0]["id"] = "eval0"
            args.train.write_text("".join(json.dumps(r) + "\n" for r in rows))
            report, _, _ = audit(args)
            self.assertFalse(report["ready"])
            self.assertTrue(any("overlap" in e for e in report["errors"]))

    def test_native_rejection_requires_exclusion_and_preserved_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = self.fixture(root)
            args.spot_check.write_text(args.spot_check.read_text(encoding="utf-8-sig").replace(",yes,", ",no,", 1))
            args.review.write_text("id,decision_keep_yes_no,fixed_response,response\ntrain0,no,,Answer train0\n")
            args.dispositions.write_text("id,disposition,notes\ntrain0,excluded,Native speaker rejected this answer\n")
            report, _, _ = audit(args)
            self.assertFalse(report["ready"], "A rejected answer still present in training must fail")
            lines = args.train.read_bytes().splitlines(keepends=True)
            args.train.write_bytes(b"".join(lines[1:]))
            report, _, _ = audit(args)
            self.assertTrue(report["ready"], report["errors"])
            self.assertEqual(report["spot_check_decisions"], {"no": 1, "yes": 99})
            self.assertEqual(report["excluded_spot_check_ids"], ["train0"])
            original_review, original_dispositions = args.review.read_bytes(), args.dispositions.read_bytes()
            for review, dispositions in (
                ("id,decision_keep_yes_no,fixed_response,response\n", original_dispositions.decode()),
                (original_review.decode(), "id,disposition,notes\ntrain0,excluded,\n"),
                (original_review.decode().replace("Answer train0", "Different answer"), original_dispositions.decode()),
                (original_review.decode().replace("no", "yes"), original_dispositions.decode()),
            ):
                with self.subTest(review=review, dispositions=dispositions):
                    args.review.write_text(review)
                    args.dispositions.write_text(dispositions)
                    failed, _, _ = audit(args)
                    self.assertFalse(failed["ready"])
            args.review.write_bytes(original_review)
            args.dispositions.write_bytes(original_dispositions)
            original_spot = args.spot_check.read_bytes()
            args.spot_check.write_text(args.spot_check.read_text().replace("Native speaker", "", 1))
            failed, _, _ = audit(args)
            self.assertFalse(failed["ready"])
            args.spot_check.write_bytes(original_spot)
            args.spot_check.write_bytes(b"".join(args.spot_check.read_bytes().splitlines(keepends=True)[:-1]))
            failed, _, _ = audit(args)
            self.assertTrue(any("found 99" in e for e in failed["errors"]))

if __name__ == "__main__":
    unittest.main()
