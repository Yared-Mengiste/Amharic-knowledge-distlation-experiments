"""The fixed first-experiment protocol; model pins are GPU-smoke candidates."""
import json
from pathlib import Path

MODELS = {
    "llama": "rasyosef/Llama-3.2-1B-Amharic-Instruct",
    "gemma": "google/gemma-4-E2B-it",
}
PROJECTIONS = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}
SETTINGS = {
    "seed": 42, "validation_fraction": 0.1, "max_length": 2048,
    "rank": 16, "alpha": 32, "dropout": 0.05, "learning_rate": 1e-4,
    "epochs": 2, "microbatch": 1, "gradient_accumulation": 16,
    "optimizer": "paged_adamw_8bit", "schedule": "cosine", "warmup_ratio": 0.03,
    "weight_decay": 0.01, "max_grad_norm": 1.0, "checkpoint_steps": 100,
    "quantization": "nf4", "double_quantization": True,
    "gradient_checkpointing": True, "enable_thinking": False,
    "date_string": "26 Jul 2024", "max_new_tokens": 1000,
}
DEFAULT_LOCK = Path(__file__).with_name("model_lock.candidate.json")


def read_lock(path=DEFAULT_LOCK, verified=False):
    lock = json.loads(Path(path).read_text())
    for key, model in MODELS.items():
        spec = lock["models"][key]
        if spec["model_id"] != model:
            raise ValueError(f"Unexpected model for {key}")
        for field in ("revision", "tokenizer_revision"):
            revision = spec[field]
            if len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision):
                raise ValueError(f"{key}: {field} must be an exact commit SHA")
    if verified and not lock.get("gpu_smoke_verified"):
        raise ValueError("Use a lock promoted after BOTH GPU smoke/resume tests")
    return lock
