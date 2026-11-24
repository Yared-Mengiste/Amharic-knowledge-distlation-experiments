import json
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

from eval.human import generate
from eval.io import load_outputs
from eval.metrics import CONFIGURATIONS, constraint_result, wilson
from eval.report import human_results
from students.data import read_csv, write_json


class EvaluationTests(unittest.TestCase):
    def test_blind_sample_records_and_validates_the_configured_seed(self):
        from students.config import SETTINGS
        prompts, outputs = self.fixture()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "human"
            with patch.dict(SETTINGS, seed=7):
                mapping = generate(prompts, outputs, root)
                self.assertEqual(mapping["seed"], 7)
                human_results(root, outputs, prompts)
            with self.assertRaisesRegex(ValueError, "Human sample IDs differ"):
                human_results(root, outputs, prompts)

    def test_generation_limit_tracks_protocol_and_rejects_overlength_outputs(self):
        from students.config import SETTINGS
        prompts, outputs = self.fixture()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.save_outputs(root, prompts, outputs)
            with patch.dict(SETTINGS, max_new_tokens=12):
                for config in CONFIGURATIONS:
                    path = root / (config + ".meta.json")
                    meta = json.loads(path.read_text())
                    meta["max_new_tokens"] = 12
                    write_json(path, meta)
                load_outputs(root, prompts)
                path = root / (CONFIGURATIONS[0] + ".jsonl")
                rows = [json.loads(line) for line in path.read_text().splitlines()]
                rows[0]["output_tokens"] = 13
                path.write_text("".join(json.dumps(row) + "\n" for row in rows))
                with self.assertRaisesRegex(ValueError, "Invalid benchmark output"):
                    load_outputs(root, prompts)

    def fixture(self):
        prompts = [{"id": f"p{i}", "category": "safety" if i < 20 else "chat" if i % 2 else "math", "language_mode": "am", "messages": [{"role": "user", "content": f"Prompt {i}"}]} for i in range(300)]
        outputs = {c: {p["id"]: {"id": p["id"], "category": p["category"], "language_mode": "am", "configuration": c, "model_revision": "llama-sha" if c.startswith("llama") else "gemma-sha", "tokenizer_revision": c.split("_")[0], "adapter_identifier": c if c.endswith("adapted") else None, "answer": c + " answer " + p["id"], "output_tokens": 10, "latency_seconds": 1.0, "termination_reason": "eos"} for p in prompts} for c in CONFIGURATIONS}
        return prompts, outputs

    def save_outputs(self, directory, prompts, outputs):
        for config in CONFIGURATIONS:
            path = directory / (config + ".jsonl")
            path.write_text("".join(json.dumps(r) + "\n" for r in outputs[config].values()))
            first = next(iter(outputs[config].values()))
            write_json(path.with_suffix(".meta.json"), {k: first[k] for k in ("configuration", "model_revision", "tokenizer_revision", "adapter_identifier")} | {"do_sample": False, "max_new_tokens": 1000, "precision": "fp16", "manifest_sha256": "same"})

    def test_requires_all_ids_and_matching_precision(self):
        prompts, outputs = self.fixture()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.save_outputs(root, prompts, outputs)
            load_outputs(root, prompts)
            path = root / "llama_adapted.meta.json"
            meta = json.loads(path.read_text())
            meta["precision"] = "bf16"
            write_json(path, meta)
            with self.assertRaisesRegex(ValueError, "precision"):
                load_outputs(root, prompts)
            meta["precision"] = "fp16"
            write_json(path, meta)
            path = root / "gemma_original.jsonl"
            path.write_text("\n".join(path.read_text().splitlines()[1:]))
            with self.assertRaisesRegex(ValueError, "missing 1"):
                load_outputs(root, prompts)

    def test_blind_mapping_scores_and_win_direction(self):
        prompts, outputs = self.fixture()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "human"
            mapping = generate(prompts, outputs, root)
            self.assertEqual(len(mapping["sample_ids"]["general"]), 50)
            self.assertEqual(len(mapping["sample_ids"]["safety"]), 20)
            self.assertEqual(len(mapping["answers"]), 280)
            self.assertEqual(len(mapping["pairs"]), 210)
            result = human_results(root, outputs, prompts)
            self.assertFalse(result["complete"])
            self.assertEqual(len(result["missing_score_or_preference_ids"]), 490)
            import csv
            for sample in ("general", "safety"):
                for kind in ("scores", "pairs"):
                    path = root / "public" / (sample + "_" + kind + ".csv")
                    rows = read_csv(path)
                    for row in rows:
                        row["reviewer"] = "Native speaker"
                        if kind == "scores":
                            for d in ("correctness", "fluency", "instruction_following", "safety"):
                                row[d] = "4"
                        else:
                            identity = mapping["pairs"][row["id"]]
                            row["winner_A_B_tie"] = "A" if identity["A"].endswith("adapted") else "B"
                    with path.open("w", newline="") as handle:
                        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                        writer.writeheader()
                        writer.writerows(rows)
            result = human_results(root, outputs, prompts)
            self.assertTrue(result["complete"])
            self.assertEqual(result["win_tie_loss"]["general"]["llama_adapted_vs_llama_original"]["win"], 50)
            path = root / "public/general_scores.csv"
            path.write_text(path.read_text().replace(" answer ", " changed ", 1))
            with self.assertRaisesRegex(ValueError, "wrong answer"):
                human_results(root, outputs, prompts)

    def test_missing_constraints_unscored_and_exact_format(self):
        self.assertIsNone(constraint_result("anything", []))
        self.assertTrue(constraint_result('{"x":1}', [{"kind": "json_keys", "keys": ["x"]}]))
        self.assertFalse(constraint_result('preface {"x":1}', [{"kind": "json"}]))
        self.assertTrue(constraint_result("1. One\n2. Two", [{"kind": "bullet_count", "count": 2}]))
        self.assertIsNone(wilson(0, 0)["ci95"])
        lo, hi = wilson(5, 10)["ci95"]
        self.assertLess(lo, 0.5)
        self.assertGreater(hi, 0.5)

    def test_format_proxies_handle_single_digit_yaml_and_never_execute_code(self):
        self.assertTrue(constraint_result("- name: item\n  qty: 2\n  price: 3", [{"kind": "yaml_fields", "fields": ["name", "qty", "price"], "records": 1}]))
        self.assertTrue(constraint_result("```python\nprint('example')\n```", [{"kind": "python_syntax"}]))
        self.assertFalse(constraint_result("Here is code:\nprint('example')", [{"kind": "python_syntax"}]))
        self.assertTrue(constraint_result("[3,6,9]", [{"kind": "json_type", "type": "array"}]))
        self.assertFalse(constraint_result('{"3":6}', [{"kind": "json_type", "type": "array"}]))
        self.assertFalse(constraint_result('"unclosed,column', [{"kind": "csv_columns", "columns": 2, "min_rows": 1, "max_rows": 1}]))

    def test_report_reference_coverage_and_category_counts(self):
        from test_students_data import DataTests
        from eval.report import build_report
        from students.data import audit, read_jsonl, sha256, split_ids
        from students.config import SETTINGS
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "dataset"
            dataset.mkdir()
            args = DataTests().fixture(dataset)
            manifest, train, paths = audit(args)
            for name, path in paths.items():
                (dataset / name).write_bytes(path.read_bytes())
            write_json(dataset / "split.json", split_ids(train))
            manifest.update(settings=SETTINGS, split_sha256=sha256(dataset / "split.json"))
            write_json(dataset / "manifest.json", manifest)
            prompts = read_jsonl(dataset / "pilot_eval.jsonl")
            _, source_outputs = self.fixture()
            outputs = {c: {} for c in CONFIGURATIONS}
            for config in CONFIGURATIONS:
                for i, prompt in enumerate(prompts):
                    row = dict(source_outputs[config][f"p{i}"])
                    row.update(id=prompt["id"], category=prompt["category"])
                    outputs[config][prompt["id"]] = row
            out = root / "eval"
            out.mkdir()
            self.save_outputs(out, prompts, outputs)
            for config in CONFIGURATIONS:
                meta_path = out / (config + ".meta.json")
                meta = json.loads(meta_path.read_text())
                meta["manifest_sha256"] = sha256(dataset / "manifest.json")
                write_json(meta_path, meta)
                write_json(out / (config + ".references.json"), {"identity": meta, "per_id": [{"id": p["id"]} for p in prompts], "by_category": {}})
            report = build_report(dataset, out, None)
            self.assertEqual(report["reference_coverage"]["approved"], 300)
            self.assertEqual(report["automated"]["llama_original"]["safety"]["outputs"], 20)
            self.assertFalse(report["comparison_ready"])
            self.assertEqual(report["automated"]["gemma_adapted"]["overall"]["format_instruction_constraints"]["unscored"], 300)
            path = out / "llama_original.references.json"
            reference = json.loads(path.read_text())
            reference["per_id"].pop()
            write_json(path, reference)
            with self.assertRaisesRegex(ValueError, "coverage"):
                build_report(dataset, out, None)


if __name__ == "__main__":
    unittest.main()
