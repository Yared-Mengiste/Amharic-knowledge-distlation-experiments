"""Single-GPU NF4 loading and text-decoder-only PEFT adapters."""
import importlib.metadata
import re

from students.config import PROJECTIONS, SETTINGS
from students.tokenization import ENDINGS


def package_versions():
    names = ("torch", "transformers", "peft", "bitsandbytes", "accelerate", "tokenizers", "huggingface-hub", "numpy", "safetensors", "jinja2")
    return {name: importlib.metadata.version(name) for name in names}


def gpu_precision(requested="auto", training=True):
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError("Select exactly one NVIDIA GPU; laptop/CPU training is unsupported")
    info = torch.cuda.get_device_properties(0)
    # Kaggle's nominal 16 GB T4 can report 14.56 GiB after reserved memory.
    # Keep the existing minimum for other GPUs and reject smaller T4 allocations.
    minimum_memory = (14 if re.search(r"\bT4\b", info.name) else 15) * 1024**3
    if training and info.total_memory < minimum_memory:
        raise ValueError(f"Need a nominal >=16 GB GPU; found {info.name} with {info.total_memory / 1024**3:.2f} GiB")
    # PyTorch's default includes emulation, which can report BF16 on a T4.
    # Select BF16 only with native hardware and runtime support.
    supported = info.major >= 8 and torch.cuda.is_bf16_supported(including_emulation=False)
    precision = ("bf16" if supported else "fp16") if requested == "auto" else requested
    if precision == "bf16" and not supported:
        raise ValueError("BF16 is not supported on this GPU; T4/P100 require FP16")
    return precision, {"name": info.name, "total_vram_bytes": info.total_memory, "precision": precision}


def load_tokenizer(spec):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(spec["model_id"], revision=spec["tokenizer_revision"], use_fast=True, trust_remote_code=False)
    if not tokenizer.is_fast:
        raise ValueError("A fast tokenizer with exact character offsets is required")
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_model(spec, precision, training=False):
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText, BitsAndBytesConfig
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    config = AutoConfig.from_pretrained(spec["model_id"], revision=spec["revision"], trust_remote_code=False)
    loader = AutoModelForImageTextToText if config.model_type == "gemma4" else AutoModelForCausalLM
    model = loader.from_pretrained(spec["model_id"], revision=spec["revision"], config=config,
                                  dtype=dtype, device_map={"": 0}, trust_remote_code=False,
                                  attn_implementation="eager",
                                  quantization_config=BitsAndBytesConfig(load_in_4bit=True,
                                      bnb_4bit_quant_type=SETTINGS["quantization"],
                                      bnb_4bit_use_double_quant=SETTINGS["double_quantization"],
                                      bnb_4bit_compute_dtype=dtype))
    model.config.use_cache = not training
    if hasattr(model.config, "text_config"):
        model.config.text_config.use_cache = not training
    return model


def adapter_targets(model, key):
    prefix = "model.layers." if key == "llama" else "model.language_model.layers."
    targets = [name for name, module in model.named_modules()
               if name.startswith(prefix) and name.split(".")[-1] in PROJECTIONS
               and hasattr(module, "in_features") and hasattr(module, "out_features")]
    found = {name.split(".")[-1] for name in targets}
    if found != PROJECTIONS:
        raise ValueError(f"Incomplete text-decoder projections: {sorted(found)}")
    return sorted(targets)


def attach_adapters(model, key):
    from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
    targets = adapter_targets(model, key)
    expected = sum(SETTINGS["rank"] * (model.get_submodule(name).in_features + model.get_submodule(name).out_features) for name in targets)
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=SETTINGS["gradient_checkpointing"],
                                           gradient_checkpointing_kwargs={"use_reentrant": False})
    # A full-name regex excludes the similarly named vision/audio projections.
    model = get_peft_model(model, LoraConfig(task_type=TaskType.CAUSAL_LM,
        r=SETTINGS["rank"], lora_alpha=SETTINGS["alpha"], lora_dropout=SETTINGS["dropout"],
        bias="none", target_modules="(?:" + "|".join(re.escape(name) for name in targets) + ")"))
    trainable = {name: p.numel() for name, p in model.named_parameters() if p.requires_grad}
    if any(".lora_A." not in name and ".lora_B." not in name for name in trainable):
        raise ValueError("Original parameters/embeddings are trainable")
    count = sum(trainable.values())
    if count != expected:
        raise ValueError(f"Adapter count mismatch: measured {count}, expected {expected}")
    if any("language_model.layers." not in name for name in trainable) and key == "gemma":
        raise ValueError("Gemma adapter attached outside its text decoder")
    return model, {"trainable_parameters": count, "targets": targets, "trainable_names": sorted(trainable)}


def configure_generation(model, tokenizer, key):
    ending = tokenizer.convert_tokens_to_ids(ENDINGS[key])
    configured = model.generation_config.eos_token_id
    stops = configured if isinstance(configured, list) else [configured]
    stops = sorted(set([ending] + [x for x in stops if x is not None]))
    model.generation_config.do_sample = False
    model.generation_config.eos_token_id = stops
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    return stops
