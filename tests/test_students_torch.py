"""CPU checks for the loss and real architecture/PEFT contracts; no student training."""
import importlib.util
import unittest
from unittest.mock import patch

HAS_TORCH = importlib.util.find_spec("torch") is not None
HAS_PEFT = importlib.util.find_spec("peft") is not None


@unittest.skipUnless(HAS_TORCH, "PyTorch optional locally; GPU smoke tests remain required")
class TorchContractTests(unittest.TestCase):
    def test_shifted_loss_ignores_user_and_supervises_ending(self):
        import torch
        import torch.nn.functional as F
        from students.train import assistant_ce
        torch.manual_seed(42)
        logits = torch.randn(1, 6, 13, requires_grad=True)
        labels = torch.tensor([[-100, -100, -100, 4, 5, -100]])
        total, count = assistant_ce(logits, labels)
        expected = F.cross_entropy(logits[0, 2:4, :], torch.tensor([4, 5]), reduction="sum")
        self.assertEqual(count.item(), 2)
        self.assertTrue(torch.allclose(total, expected))
        total.backward()
        self.assertEqual(logits.grad[0, :2].abs().sum().item(), 0)
        self.assertGreater(logits.grad[0, 3].abs().sum().item(), 0)
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_chunked_loss_matches_full_token_loss_and_gradients(self):
        import torch
        import torch.nn.functional as F
        from students.train import assistant_ce
        torch.manual_seed(42)
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype):
                logits = torch.randn(2, 150, 19, dtype=dtype, requires_grad=True)
                reference = logits.detach().clone().requires_grad_(True)
                labels = torch.randint(0, 19, (2, 150))
                labels[:, :40] = -100
                labels[1, 125:] = -100  # Unequal answer lengths and a partial final chunk.
                total, count = assistant_ce(logits, labels)
                expected = F.cross_entropy(reference[:, :-1].reshape(-1, 19).float(), labels[:, 1:].reshape(-1), ignore_index=-100, reduction="sum")
                self.assertTrue(torch.allclose(total, expected))
                self.assertEqual(count.item(), 195)
                total.backward()
                expected.backward()
                self.assertTrue(torch.equal(logits.grad, reference.grad))
                with torch.no_grad():
                    evaluated, eval_count = assistant_ce(logits, labels)
                self.assertTrue(torch.allclose(evaluated, expected))
                self.assertEqual(eval_count.item(), count.item())

    def test_checkpointed_loss_does_not_retain_fp32_vocabulary_buffers(self):
        import torch
        from students.train import assistant_ce
        logits = torch.randn(1, 257, 1024, dtype=torch.float16, requires_grad=True)
        labels = torch.full((1, 257), -100, dtype=torch.long)
        labels[:, 70:] = 7
        saved = []

        def pack(tensor):
            saved.append(tensor)
            return tensor

        with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
            loss, _ = assistant_ce(logits, labels)
        self.assertFalse(any(t.dtype == torch.float32 and t.ndim >= 2 and t.shape[-1] == 1024 for t in saved))
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_masked_logits_are_excluded_even_when_nonfinite(self):
        import torch
        from students.train import assistant_ce
        logits = torch.randn(1, 150, 19)
        logits[:, :99] = float("nan")
        logits.requires_grad_(True)
        labels = torch.full((1, 150), -100, dtype=torch.long)
        labels[:, 100:] = 7
        loss, count = assistant_ce(logits, labels)
        self.assertEqual(count.item(), 50)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertEqual(logits.grad[:, :99].abs().sum().item(), 0)

    def test_loss_requires_supervision_and_positive_chunk_size(self):
        import torch
        from students.config import SETTINGS
        from students.train import assistant_ce
        logits = torch.randn(1, 3, 19, requires_grad=True)
        with self.assertRaisesRegex(ValueError, "No assistant tokens"):
            assistant_ce(logits, torch.full((1, 3), -100, dtype=torch.long))
        with patch.dict(SETTINGS, loss_chunk_size=0):
            with self.assertRaisesRegex(ValueError, "positive integer"):
                assistant_ce(logits, torch.tensor([[-100, 7, 7]]))

    @unittest.skipUnless(HAS_PEFT, "PEFT needed for architecture contracts")
    def test_real_llama_count_and_frozen_original_parameters(self):
        import torch
        from transformers import LlamaConfig, LlamaForCausalLM
        from students.modeling import attach_adapters
        # Same decoder dimensions as the pinned public checkpoint; meta avoids weights.
        config = LlamaConfig(hidden_size=2048, intermediate_size=8192, num_hidden_layers=16,
                             num_attention_heads=32, num_key_value_heads=8, head_dim=64,
                             vocab_size=143424, tie_word_embeddings=True)
        with torch.device("meta"):
            model = LlamaForCausalLM(config)
            model, info = attach_adapters(model, "llama")
        self.assertEqual(info["trainable_parameters"], 11_272_192)
        self.assertTrue(all(".lora_" in n for n, p in model.named_parameters() if p.requires_grad))

    @unittest.skipUnless(HAS_PEFT, "PEFT needed for architecture contracts")
    def test_gemma_targets_exclude_vision_audio_and_embeddings(self):
        import torch
        from transformers import Gemma4Config, Gemma4ForConditionalGeneration
        from students.modeling import attach_adapters
        config = Gemma4Config(text_config={"hidden_size": 1536, "intermediate_size": 6144,
            "num_hidden_layers": 35, "num_attention_heads": 8, "num_key_value_heads": 1,
            "head_dim": 256, "global_head_dim": 512, "num_kv_shared_layers": 20,
            "hidden_size_per_layer_input": 256, "vocab_size_per_layer_input": 262144,
            "vocab_size": 262144, "use_double_wide_mlp": True,
            "layer_types": ["full_attention" if i % 5 == 4 else "sliding_attention" for i in range(35)]})
        with torch.device("meta"):
            model = Gemma4ForConditionalGeneration(config)
            model, info = attach_adapters(model, "gemma")
        self.assertGreater(info["trainable_parameters"], 0)
        self.assertTrue(all("model.language_model.layers." in name for name in info["targets"]))
        self.assertTrue(all("language_model.layers." in n for n, p in model.named_parameters() if p.requires_grad))


if __name__ == "__main__":
    unittest.main()
