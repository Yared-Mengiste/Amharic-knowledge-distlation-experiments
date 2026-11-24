import json
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

from eval.human import generate
from eval.io import load_outputs
from eval.metrics import configurations, comparison_pairs, constraint_result, wilson
from students.config import MODELS
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
                for config in configurations(MODELS):
                    path = root / (config + ".meta.json")
                    meta = json.loads(path.read_text())
                    meta["max_new_tokens"] = 12
                    write_json(path, meta)
                load_outputs(root, prompts)
                path = root / (configurations(MODELS)[0] + ".jsonl")
                rows = [json.loads(line) for line in path.read_text().splitlines()]
                rows[0]["output_tokens"] = 13
                path.write_text("".join(json.dumps(row) + "\n" for row in rows))
                with self.assertRaisesRegex(ValueError, "Invalid benchmark output"):
                    load_outputs(root, prompts)

    def fixture(self, models=MODELS):
        prompts = [{"id": f"p{i}", "category": "safety" if i < 20 else "chat" if i % 2 else "math", "language_mode": "am", "messages": [{"role": "user", "content": f"Prompt {i}"}]} for i in range(300)]
        outputs = {c: {p["id"]: {"id": p["id"], "category": p["category"], "language_mode": "am", "configuration": c, "model_revision": "llama-sha" if c.startswith("llama") else "gemma-sha", "tokenizer_revision": c.split("_")[0], "adapter_identifier": c if c.endswith("adapted") else None, "answer": c + " answer " + p["id"], "output_tokens": 10, "latency_seconds": 1.0, "termination_reason": "eos"} for p in prompts} for c in configurations(models)}
        return prompts, outputs

    def save_outputs(self, directory, prompts, outputs):
        for config in outputs:
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

    def test_llama_only_blind_sample_and_incomplete_pair_rejection(self):
        prompts, outputs = self.fixture(["llama"])
        from students.data import stratified_ids
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.save_outputs(root, prompts, outputs)
            loaded, _, coverage = load_outputs(root, prompts)
            self.assertEqual(set(loaded), {"llama_original", "llama_adapted"})
            self.assertTrue(all(c["outputs"] == 300 for c in coverage.values()))
            human = root / "human"
            mapping = generate(prompts, outputs, human)
            self.assertEqual(set(mapping["sample_ids"]["general"]), set(stratified_ids(prompts, 50, seed=42)))
            self.assertEqual(len(mapping["sample_ids"]["safety"]), 20)
            self.assertEqual(len(mapping["answers"]), 140)
            self.assertEqual(len(mapping["pairs"]), 70)
            self.assertEqual({(p["A"], p["B"]) for p in mapping["pairs"].values()},
                             {("llama_original", "llama_adapted"), ("llama_adapted", "llama_original")})
            for sample in ("general", "safety"):
                rows = read_csv(human / "public" / (sample + "_scores.csv"))
                self.assertNotIn("configuration", rows[0])
            result = human_results(human, outputs, prompts)
            self.assertEqual(set(result["win_tie_loss"]["general"]), {"llama_adapted_vs_llama_original"})
            self.assertFalse(result["complete"])
            # A Gemma sidecar makes Gemma present, even if its answer file is missing.
            write_json(root / "gemma_original.meta.json", {})
            with self.assertRaisesRegex(ValueError, "original and adapted"):
                load_outputs(root, prompts)

    def test_accept_without_gemma_arguments_and_integrity_failures(self):
        import subprocess
        import sys
        from eval.human import answer_hash
        from students.config import SETTINGS, read_lock, training_steps
        from students.data import sha256, write_csv
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run, deploy = root / "run", root / "deploy"
            (run / "adapter").mkdir(parents=True)
            deploy.mkdir()
            adapter = run / "adapter/adapter_model.safetensors"
            adapter.write_bytes(b"synthetic adapter fixture")
            spec = read_lock()["models"]["llama"]
            split = {"train_ids": ["train0"], "validation_ids": ["validation0"]}
            write_json(run / "split.json", split)
            identity = {"model": "llama", "spec": spec, "settings": SETTINGS, "smoke_steps": None,
                        "manifest_sha256": "synthetic-manifest", "split_sha256": sha256(run / "split.json")}
            steps = training_steps(1)
            checkpoint = run / f"checkpoint-{steps}"
            checkpoint.mkdir()
            for name in ("optimizer.pt", "scheduler.pt", "rng_state.pth", "adapter_config.json", "adapter_model.safetensors"):
                (checkpoint / name).write_bytes(b"synthetic checkpoint fixture")
            write_json(checkpoint / "run_identity.json", identity)
            write_json(checkpoint / "trainer_state.json", {"global_step": steps})
            write_json(run / "adapter_parameters.json", {"trainable_parameters": 1})
            write_json(run / "metrics.json", {"identity": identity, "completed_steps": steps,
                       "adapter_load_verified": True, "best_validation_loss": 1.0,
                       "adapter_parameters": 1, "finite_gradient_steps": 1})
            prompt = {"id": "eval0", "category": "chat", "prompt": "synthetic prompt"}
            (deploy / "deployment_prompts.jsonl").write_text(json.dumps(prompt) + "\n")
            answer = {**prompt, "answer": "synthetic answer"}
            (deploy / "answers.jsonl").write_text(json.dumps(answer) + "\n")
            review = {**answer, "answer_sha256": answer_hash(answer["answer"]), "reviewer": "Test reviewer",
                      "correctness": "4", "fluency": "4", "instruction_following": "4"}
            write_csv(deploy / "quality_review.csv", [review], list(review))
            write_json(deploy / "metrics.json", {"quantization": "Q4_K_M", "server_command": ["test-server", "0"],
                       "identity": {"spec": spec, "adapter_sha256": sha256(adapter),
                                    "deployment_prompts_sha256": sha256(deploy / "deployment_prompts.jsonl")}})
            evaluation = {"model_id": spec["model_id"], "model_revision": spec["revision"],
                          "tokenizer_revision": spec["tokenizer_revision"], "adapter_identifier": sha256(adapter)}
            report = {"comparison_ready": True, "manifest_sha256": identity["manifest_sha256"],
                      "evaluation_metadata": {c: evaluation for c in configurations(["llama"])},
                      "artifacts": {c: {"references_sha256": "synthetic-reference-hash"} for c in configurations(["llama"])}}
            report_path, log, accepted = root / "report.json", root / "LOG.md", root / "accepted.json"
            write_json(report_path, report)
            def record_fixture_hashes():
                log.write_text("\n".join([sha256(report_path), "synthetic-reference-hash", sha256(run / "metrics.json"), sha256(deploy / "metrics.json")]))
            record_fixture_hashes()
            command = [sys.executable, "-m", "eval.accept", "--report", str(report_path),
                       "--llama-run", str(run), "--llama-deploy", str(deploy), "--log", str(log), "--out", str(accepted)]
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(set(json.loads(accepted.read_text())["runs"]), {"llama"})
            (run / "split.json").write_text("{}")
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("split changed", result.stderr)
            write_json(run / "split.json", split)
            quality = deploy / "quality_review.csv"
            quality.write_text(quality.read_text(encoding="utf-8-sig").replace(review["answer_sha256"], "changed"))
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("wrong/unreviewed answer", result.stderr)
            report["evaluation_metadata"]["gemma_original"] = evaluation
            report["evaluation_metadata"]["gemma_adapted"] = evaluation
            write_json(report_path, report)
            record_fixture_hashes()
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("exactly the report's models", result.stderr)
            result = subprocess.run(command + ["--gemma-run", str(run)], capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("must be supplied together", result.stderr)

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
        self.check_report_models(MODELS)

    def test_llama_only_report_with_299_references(self):
        self.check_report_models(["llama"], missing_reference=True)

    def check_report_models(self, models, missing_reference=False):
        from test_students_data import DataTests
        from eval.report import build_report, markdown_report
        from students.data import audit, read_jsonl, sha256, split_ids
        from students.config import SETTINGS
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "dataset"
            dataset.mkdir()
            args = DataTests().fixture(dataset)
            if missing_reference:
                lines = args.references.read_text().splitlines()
                removed = json.loads(lines.pop())
                args.references.write_text("\n".join(lines) + "\n")
                args.rejected.write_text(json.dumps({"id": removed["id"]}) + "\n")
                args.dispositions.write_text("id,disposition,notes\n" + removed["id"] + ",excluded,Test excluded reference\n")
            manifest, train, paths = audit(args)
            self.assertTrue(manifest["ready"], manifest["errors"])
            for name, path in paths.items():
                (dataset / name).write_bytes(path.read_bytes())
            write_json(dataset / "split.json", split_ids(train))
            manifest.update(settings=SETTINGS, split_sha256=sha256(dataset / "split.json"))
            write_json(dataset / "manifest.json", manifest)
            prompts = read_jsonl(dataset / "pilot_eval.jsonl")
            _, source_outputs = self.fixture(models)
            outputs = {c: {} for c in configurations(models)}
            for config in configurations(models):
                for i, prompt in enumerate(prompts):
                    row = dict(source_outputs[config][f"p{i}"])
                    row.update(id=prompt["id"], category=prompt["category"])
                    outputs[config][prompt["id"]] = row
            out = root / "eval"
            out.mkdir()
            self.save_outputs(out, prompts, outputs)
            for config in configurations(models):
                meta_path = out / (config + ".meta.json")
                meta = json.loads(meta_path.read_text())
                meta["manifest_sha256"] = sha256(dataset / "manifest.json")
                write_json(meta_path, meta)
                write_json(out / (config + ".references.json"), {"identity": meta, "per_id": [{"id": r["id"]} for r in read_jsonl(dataset / "eval_teacher.jsonl")], "by_category": {}})
            report = build_report(dataset, out, None)
            self.assertEqual(set(report["automated"]), set(configurations(models)))
            self.assertEqual(report["comparison_pairs"], comparison_pairs(models))
            self.assertEqual(report["reference_coverage"]["approved"], 299 if missing_reference else 300)
            rendered = markdown_report(report)
            self.assertIn("299/300" if missing_reference else "300/300", rendered)
            self.assertIn("Category-level", rendered)
            self.assertIn("single training seed", rendered)
            self.assertIn("small human sample", rendered)
            self.assertEqual(report["automated"]["llama_original"]["safety"]["outputs"], 20)
            self.assertFalse(report["comparison_ready"])
            self.assertEqual(report["automated"][configurations(models)[-1]]["overall"]["format_instruction_constraints"]["unscored"], 300)
            path = out / "llama_original.references.json"
            reference = json.loads(path.read_text())
            reference["per_id"].pop()
            write_json(path, reference)
            with self.assertRaisesRegex(ValueError, "coverage"):
                build_report(dataset, out, None)


if __name__ == "__main__":
    unittest.main()
