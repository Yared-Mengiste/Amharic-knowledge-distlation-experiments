"""GPU eligibility checks with mocked CUDA; actual GPU smoke remains required."""
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from students.modeling import gpu_precision


class GPUChecks(unittest.TestCase):
    def cuda(self, name="Tesla T4", gib=14.56, count=1, available=True, bf16=False, emulation=False, major=None):
        properties = SimpleNamespace(name=name, total_memory=int(gib * 1024**3), major=(8 if bf16 else 7) if major is None else major)
        return SimpleNamespace(cuda=SimpleNamespace(
            is_available=lambda: available,
            device_count=lambda: count,
            get_device_properties=lambda index: properties,
            is_bf16_supported=lambda including_emulation=True: bf16 or (including_emulation and emulation),
        ))

    def test_kaggle_t4_reserved_memory_and_actual_vram_recorded(self):
        with patch.dict(sys.modules, torch=self.cuda()):
            precision, gpu = gpu_precision()
        self.assertEqual(precision, "fp16")
        self.assertEqual(gpu["total_vram_bytes"], int(14.56 * 1024**3))
        self.assertEqual(gpu["name"], "Tesla T4")

    def test_small_t4_and_other_small_gpus_still_rejected(self):
        for name, gib in (("Tesla T4", 12), ("Tesla T4", 13.99), ("Other GPU", 14.56)):
            with self.subTest(name=name, gib=gib), patch.dict(sys.modules, torch=self.cuda(name, gib)):
                with self.assertRaisesRegex(ValueError, "nominal >=16 GB"):
                    gpu_precision()
        with patch.dict(sys.modules, torch=self.cuda("Tesla P100", 15.89)):
            self.assertEqual(gpu_precision()[0], "fp16")

    def test_exactly_one_cuda_gpu_and_precision_requirements(self):
        for count, available in ((2, True), (0, False)):
            with self.subTest(count=count), patch.dict(sys.modules, torch=self.cuda(count=count, available=available)):
                with self.assertRaisesRegex(ValueError, "exactly one"):
                    gpu_precision()
        with patch.dict(sys.modules, torch=self.cuda()):
            with self.assertRaisesRegex(ValueError, "BF16 is not supported"):
                gpu_precision("bf16")
        with patch.dict(sys.modules, torch=self.cuda("NVIDIA A100", 40, bf16=True)):
            self.assertEqual(gpu_precision()[0], "bf16")

    def test_emulated_bf16_never_authorizes_native_bf16_training(self):
        for name, major in (("Tesla T4", 7), ("Other GPU", 8)):
            with self.subTest(name=name), patch.dict(sys.modules, torch=self.cuda(name, 16, emulation=True, major=major)):
                self.assertEqual(gpu_precision()[0], "fp16")
                with self.assertRaisesRegex(ValueError, "BF16 is not supported"):
                    gpu_precision("bf16")


if __name__ == "__main__":
    unittest.main()
