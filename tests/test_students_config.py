import unittest
from unittest.mock import patch

from students.config import SETTINGS, training_argument_values, training_steps
from students.data import split_ids, stratified_ids
from students.tokenization import template_kwargs


class ProtocolTests(unittest.TestCase):
    def test_changed_protocol_controls_trainer_split_and_completion(self):
        rows = [{"id": str(i), "category": str(i % 7)} for i in range(503)]
        with patch.dict(SETTINGS, seed=7, validation_fraction=0.2, epochs=2.5,
                        microbatch=3, gradient_accumulation=4, learning_rate=7e-5,
                        checkpoint_steps=5, gradient_checkpointing=False):
            split = split_ids(rows)
            arguments = training_argument_values("run", "bf16")
            self.assertEqual(len(split["validation_ids"]), 101)
            self.assertEqual(split["seed"], arguments["seed"])
            self.assertEqual(arguments["data_seed"], 7)
            self.assertEqual(arguments["learning_rate"], 7e-5)
            self.assertEqual(arguments["eval_steps"], arguments["save_steps"])
            self.assertEqual(arguments["save_steps"], 5)
            self.assertFalse(arguments["gradient_checkpointing"])
            self.assertEqual(arguments["num_train_epochs"], 2.5)
            self.assertEqual(arguments["per_device_train_batch_size"], 3)
            self.assertEqual(arguments["gradient_accumulation_steps"], 4)
            self.assertEqual(training_steps(len(split["train_ids"])), 85)
            self.assertEqual(stratified_ids(rows, 100), stratified_ids(rows, 100, seed=7))
            self.assertNotEqual(stratified_ids(rows, 100), stratified_ids(rows, 100, seed=42))

    def test_partial_batches_epochs_and_smoke_keep_consistent_settings(self):
        self.assertEqual(training_steps(4457), 558)
        with patch.dict(SETTINGS, microbatch=3, gradient_accumulation=4, epochs=2.5):
            self.assertEqual(training_steps(17), 5)
            full = training_argument_values("run", "fp16")
            smoke = training_argument_values("run", "fp16", smoke_steps=4)
            self.assertEqual({**full, "max_steps": 4}, smoke)
        with self.assertRaisesRegex(ValueError, "one example"):
            training_steps(0)
        with self.assertRaisesRegex(ValueError, "precision"):
            training_argument_values("run", "auto")

    def test_template_options_come_from_recorded_settings(self):
        with patch.dict(SETTINGS, enable_thinking=True, date_string="01 Jan 2020"):
            self.assertEqual(template_kwargs(), {"enable_thinking": True, "date_string": "01 Jan 2020"})


if __name__ == "__main__":
    unittest.main()
