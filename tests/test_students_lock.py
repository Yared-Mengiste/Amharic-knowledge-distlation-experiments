"""CPU-only lock/protocol fixtures; these do not provide GPU verification."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from students import evaluate, lock as promotion, preflight, train
from students.config import DEFAULT_LOCK, SETTINGS, read_lock
from students.data import sha256, write_json
from test_students_data import FakeTokenizer, conversation


class LockTests(unittest.TestCase):
    def smoke(self, key, candidate):
        return {"identity": {"model": key, "spec": candidate["models"][key],
                "settings": copy.deepcopy(SETTINGS), "packages": {"test-package": "0.0"},
                "manifest_sha256": "test-manifest", "split_sha256": "test-split",
                "precision": "fp16", "smoke_steps": 4},
                "gpu_smoke_passed": True, "adapter_load_verified": True,
                "resumed_from_step": 2, "resumed_step_observed": 2, "completed_steps": 4}

    def promote(self, root, candidate, reports):
        candidate_path = root / "candidate.json"
        write_json(candidate_path, candidate)
        arguments = ["students.lock", "--candidate", str(candidate_path), "--out", str(root / "promoted/model_lock.json")]
        for key, report in reports.items():
            path = root / (key + ".smoke.json")
            write_json(path, report)
            arguments += ["--" + key + "-smoke", str(path)]
        with patch("sys.argv", arguments):
            promotion.main()
        return root / "promoted/model_lock.json"

    def test_llama_only_candidate_and_required_llama_and_exact_pins(self):
        candidate = read_lock(DEFAULT_LOCK.with_name("model_lock.llama.candidate.json"))
        self.assertEqual(set(candidate["models"]), {"llama"})
        self.assertEqual(candidate["models"]["llama"], read_lock()["models"]["llama"])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "lock.json"
            for models in ({}, {"gemma": read_lock()["models"]["gemma"]}):
                write_json(path, {"models": models})
                with self.assertRaisesRegex(ValueError, "contain llama"):
                    read_lock(path)
            for key in ("llama", "gemma"):
                for field in ("revision", "tokenizer_revision"):
                    broken = read_lock()
                    broken["models"][key][field] = "main"
                    write_json(path, broken)
                    with self.assertRaisesRegex(ValueError, "exact commit SHA"):
                        read_lock(path)

    def test_promote_llama_only_and_never_overwrite(self):
        candidate = read_lock(DEFAULT_LOCK.with_name("model_lock.llama.candidate.json"))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reports = {"llama": self.smoke("llama", candidate)}
            path = self.promote(root, candidate, reports)
            promoted = read_lock(path, verified=True, model="llama")
            self.assertEqual(promoted["verified_models"], ["llama"])
            self.assertEqual(promoted["split_sha256"], "test-split")
            self.assertEqual(promoted["precision"], "fp16")
            self.assertEqual(promoted["smoke_reports"], {"llama": sha256(root / "llama.smoke.json")})
            with self.assertRaisesRegex(ValueError, "Never overwrite"):
                self.promote(root, candidate, reports)

    def test_train_evaluate_refuse_absent_and_unverified_model_before_loading(self):
        for candidate in (read_lock(), read_lock(DEFAULT_LOCK.with_name("model_lock.llama.candidate.json"))):
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                path = self.promote(root, candidate, {"llama": self.smoke("llama", candidate)})
                for module in (train, evaluate):
                    arguments = [module.__name__, "--dataset", "unused", "--lock", str(path),
                                 "--model", "gemma", "--out", str(root / "unused")]
                    if module is train:
                        arguments += ["--preflight", "unused"]
                    with patch("sys.argv", arguments), patch.object(module, "load_frozen") as loading:
                        with self.assertRaisesRegex(ValueError, "gemma:.*(absent|not GPU smoke/resume verified)"):
                            module.main()
                        loading.assert_not_called()

    def test_all_given_smokes_must_match_and_resume(self):
        candidate = read_lock()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reports = {key: self.smoke(key, candidate) for key in candidate["models"]}
            for field, value in (("packages", {}), ("manifest_sha256", "other"),
                                 ("split_sha256", "other"), ("precision", "bf16"), ("settings", {})):
                broken = copy.deepcopy(reports)
                broken["gemma"]["identity"][field] = value
                with self.subTest(field=field), self.assertRaises(ValueError):
                    self.promote(root, candidate, broken)
            for field, value in (("resumed_from_step", 0), ("resumed_step_observed", 1),
                                 ("completed_steps", 3), ("adapter_load_verified", False), ("gpu_smoke_passed", False)):
                broken = copy.deepcopy(reports)
                broken["llama"][field] = value
                with self.subTest(field=field), self.assertRaises(ValueError):
                    self.promote(root, candidate, broken)
            promoted = read_lock(self.promote(root, candidate, reports), verified=True, model="gemma")
            self.assertEqual(promoted["verified_models"], ["llama", "gemma"])

    def test_preflight_and_training_comparison_only_cover_locked_models(self):
        candidate = read_lock(DEFAULT_LOCK.with_name("model_lock.llama.candidate.json"))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("sft_train.jsonl", "eval_teacher.jsonl"):
                (root / name).write_text(json.dumps(conversation(name)) + "\n")
            prompt = conversation("eval")
            prompt["messages"] = prompt["messages"][:1]
            (root / "pilot_eval.jsonl").write_text(json.dumps(prompt) + "\n")
            (root / "manifest.json").write_text("test-manifest")
            with patch.object(preflight, "load_frozen"), patch.object(preflight, "load_tokenizer", return_value=FakeTokenizer()) as loading, patch.object(preflight, "template_hash", return_value="test-template"):
                result = preflight.run(root, candidate, root / "preflight.json")
            self.assertTrue(result["passed"])
            self.assertEqual(set(result["models"]), {"llama"})
            loading.assert_called_once_with(candidate["models"]["llama"])
            train.verify_preflight(result, candidate, sha256(root / "manifest.json"))
            result["models"]["gemma"] = {"spec": "irrelevant extra preflight entry"}
            train.verify_preflight(result, candidate, sha256(root / "manifest.json"))
            result["models"]["llama"]["spec"] = {}
            with self.assertRaisesRegex(ValueError, "revisions differ"):
                train.verify_preflight(result, candidate, sha256(root / "manifest.json"))
