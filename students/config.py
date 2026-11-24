"""The fixed first-experiment protocol; model pins are GPU-smoke candidates."""
import json
import math
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
    "loss_chunk_size": 64,
}
DEFAULT_LOCK = Path(__file__).with_name("model_lock.candidate.json")


def training_steps(num_examples):
    """Optimizer updates for one GPU, including partial batches and epochs."""
    if num_examples <= 0:
        raise ValueError("Training needs at least one example")
    batches = math.ceil(num_examples / SETTINGS["microbatch"])
    updates_per_epoch = math.ceil(batches / SETTINGS["gradient_accumulation"])
    return math.ceil(updates_per_epoch * SETTINGS["epochs"])


def training_argument_values(output, precision, smoke_steps=None):
    """Values passed to Trainer, derived from the recorded protocol settings."""
    if precision not in {"fp16", "bf16"}:
        raise ValueError("Resolve GPU precision before constructing training arguments")
    return {
        "output_dir": str(output), "num_train_epochs": SETTINGS["epochs"],
        "learning_rate": SETTINGS["learning_rate"],
        "per_device_train_batch_size": SETTINGS["microbatch"],
        "per_device_eval_batch_size": SETTINGS["microbatch"],
        "gradient_accumulation_steps": SETTINGS["gradient_accumulation"],
        "gradient_checkpointing": SETTINGS["gradient_checkpointing"],
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
        "optim": SETTINGS["optimizer"], "lr_scheduler_type": SETTINGS["schedule"],
        "warmup_ratio": SETTINGS["warmup_ratio"], "weight_decay": SETTINGS["weight_decay"],
        "max_grad_norm": SETTINGS["max_grad_norm"],
        "seed": SETTINGS["seed"], "data_seed": SETTINGS["seed"],
        "fp16": precision == "fp16", "bf16": precision == "bf16",
        "eval_strategy": "steps", "eval_steps": SETTINGS["checkpoint_steps"],
        "save_strategy": "steps", "save_steps": SETTINGS["checkpoint_steps"],
        "load_best_model_at_end": True, "metric_for_best_model": "eval_loss",
        "greater_is_better": False, "save_total_limit": None,
        "logging_steps": 1, "report_to": [], "remove_unused_columns": False,
        "max_steps": smoke_steps if smoke_steps is not None else -1,
        "dataloader_num_workers": 0, "save_safetensors": True,
        "logging_nan_inf_filter": False,
    }


def quantization_label():
    return SETTINGS["quantization"].upper() + ("-double" if SETTINGS["double_quantization"] else "")


def read_lock(path=DEFAULT_LOCK, verified=False, model=None):
    lock = json.loads(Path(path).read_text())
    if "llama" not in lock["models"]:
        raise ValueError("A student lock must contain llama")
    for key, spec in lock["models"].items():
        if key not in MODELS or spec["model_id"] != MODELS[key]:
            raise ValueError(f"Unexpected model for {key}")
        for field in ("revision", "tokenizer_revision"):
            revision = spec[field]
            if not isinstance(revision, str) or len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision):
                raise ValueError(f"{key}: {field} must be an exact commit SHA")
    verified_models = lock.get("verified_models", [])
    if not isinstance(verified_models, list) or len(set(verified_models)) != len(verified_models) or set(verified_models) - set(lock["models"]):
        raise ValueError("Invalid verified_models in student lock")
    if model is not None and model not in lock["models"]:
        raise ValueError(f"{model}: model is absent from this lock; supply its candidate pins and pass GPU smoke/resume before use")
    if verified and (not lock.get("gpu_smoke_verified") or (model or "llama") not in verified_models):
        raise ValueError(f"{model or 'llama'}: model is not GPU smoke/resume verified in this lock")
    return lock
