"""Fixed-protocol QLoRA; smoke, resume, and full training share the same settings."""
import argparse
import json
import math
import shutil
import time
from pathlib import Path

from students.config import DEFAULT_LOCK, SETTINGS, read_lock, training_argument_values, training_steps
from students.data import load_frozen, read_jsonl, sha256, write_json
from students.modeling import attach_adapters, gpu_precision, load_model, load_tokenizer, package_versions
from students.tokenization import AssistantCollator, encode_conversation, template_hash


def assistant_ce(logits, labels):
    """Shifted assistant CE; checkpoint chunks to avoid retained FP32 loss buffers.

    The model still produces full logits. Only supervised positions enter CE,
    and training recomputes one chunk's FP32 softmax during backward.
    """
    import torch
    import torch.nn.functional as F
    from torch.utils.checkpoint import checkpoint
    labels = labels[:, 1:]
    logits = logits[:, :-1, :]
    count = (labels != -100).sum()
    if count.item() == 0:
        raise ValueError("No assistant tokens in batch")
    chunk_size = SETTINGS["loss_chunk_size"]
    if not isinstance(chunk_size, int) or chunk_size <= 0:
        raise ValueError("loss_chunk_size must be a positive integer")

    def chunk_loss(chunk, target):
        supervised = target != -100
        return F.cross_entropy(chunk[supervised].float(), target[supervised], reduction="sum")

    total = torch.zeros((), device=logits.device, dtype=torch.float32)
    for start in range(0, labels.shape[1], chunk_size):
        chunk = logits[:, start:start + chunk_size, :]
        target = labels[:, start:start + chunk_size]
        if not (target != -100).any().item():
            continue
        if torch.is_grad_enabled() and chunk.requires_grad:
            loss = checkpoint(chunk_loss, chunk, target, use_reentrant=False, preserve_rng_state=False)
        else:
            loss = chunk_loss(chunk, target)
        total = total + loss
    return total, count


def verify_resume(checkpoint, identity):
    checkpoint = Path(checkpoint)
    for name in ("trainer_state.json", "optimizer.pt", "scheduler.pt", "rng_state.pth", "adapter_config.json", "adapter_model.safetensors", "run_identity.json"):
        if not (checkpoint / name).is_file():
            raise ValueError(f"Incomplete resumable checkpoint: missing {name}")
    if json.loads((checkpoint / "run_identity.json").read_text()) != identity:
        raise ValueError("Resume dataset/model/precision/packages/settings differ")
    state = json.loads((checkpoint / "trainer_state.json").read_text())
    if state["global_step"] <= 0:
        raise ValueError("Checkpoint has no completed optimizer steps")
    return state["global_step"]


def verify_preflight(preflight, lock, manifest_hash):
    if not preflight["passed"] or preflight["manifest_sha256"] != manifest_hash:
        raise ValueError("A successful preflight for this snapshot's locked models is required")
    if any(preflight["models"].get(k, {}).get("spec") != v for k, v in lock["models"].items()):
        raise ValueError("Preflight model/tokenizer revisions differ")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model", required=True, choices=["llama", "gemma"])
    parser.add_argument("--lock", default=str(DEFAULT_LOCK))
    parser.add_argument("--preflight", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--precision", choices=["auto", "fp16", "bf16"], default="auto")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke-steps", type=int, default=4)
    parser.add_argument("--stop-after", type=int, help="Smoke only: save and stop, then resume with identical arguments")
    parser.add_argument("--resume", help="Exact checkpoint directory; never auto-restart")
    args = parser.parse_args()
    if args.stop_after is not None and (not args.smoke or not 0 < args.stop_after < args.smoke_steps):
        parser.error("--stop-after must be a positive smoke step below --smoke-steps")
    lock = read_lock(args.lock, verified=not args.smoke, model=args.model)
    manifest, split = load_frozen(args.dataset)
    if not args.smoke and lock["manifest_sha256"] != sha256(Path(args.dataset) / "manifest.json"):
        raise ValueError("Full training must use the dataset used by the verified smoke tests")
    spec = lock["models"][args.model]
    preflight = json.loads(Path(args.preflight).read_text())
    verify_preflight(preflight, lock, sha256(Path(args.dataset) / "manifest.json"))
    import torch
    from transformers import Trainer, TrainerCallback, TrainingArguments, set_seed
    precision, gpu = gpu_precision(args.precision)
    versions = package_versions()
    if not args.smoke and versions != lock["packages"]:
        raise ValueError("Installed package versions differ from the GPU-verified lock")
    set_seed(SETTINGS["seed"])
    tokenizer = load_tokenizer(spec)
    if template_hash(tokenizer) != preflight["models"][args.model]["template_sha256"]:
        raise ValueError("Chat template changed after preflight")
    rows = read_jsonl(Path(args.dataset) / "sft_train.jsonl")
    features = {r["id"]: encode_conversation(tokenizer, r, args.model) for r in rows}
    if any(len(f["input_ids"]) > SETTINGS["max_length"] for f in features.values()):
        raise ValueError("Overlength conversation despite preflight")
    train_data = [features[rid] for rid in split["train_ids"]]
    val_data = [features[rid] for rid in split["validation_ids"]]
    identity = {"model": args.model, "spec": spec, "settings": SETTINGS,
                "manifest_sha256": sha256(Path(args.dataset) / "manifest.json"),
                "split_sha256": manifest["split_sha256"], "template_sha256": template_hash(tokenizer),
                "precision": precision, "packages": versions,
                "smoke_steps": args.smoke_steps if args.smoke else None}
    output = Path(args.out)
    completed = 0
    if args.resume:
        if Path(args.resume).resolve().parent != output.resolve():
            raise ValueError("Resume checkpoint must belong to this output directory")
        completed = verify_resume(args.resume, identity)
    elif output.exists() and any(output.iterdir()):
        raise ValueError("Output exists; use --resume or a new run directory")
    # Validate the pinned Trainer API before loading weights or writing run metadata.
    training_args = TrainingArguments(**training_argument_values(
        output, precision, args.smoke_steps if args.smoke else None))
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "run_identity.json", identity)
    write_json(output / "split.json", split)
    write_json(output / "package_versions.json", versions)
    shutil.copyfile(args.lock, output / "model_lock.json")
    shutil.copyfile(Path(args.dataset) / "manifest.json", output / "dataset_manifest.json")
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    model = None

    class Checks(TrainerCallback):
        def __init__(self):
            self.step_times = []
            self.step_started = None
            self.finite_gradients = 0
            self.resumed_step_observed = None

        def on_train_begin(self, args_, state, control, **kwargs):
            self.resumed_step_observed = state.global_step
            if state.global_step != completed:
                raise ValueError("Trainer did not restore completed optimizer steps")

        def on_step_begin(self, args_, state, control, **kwargs):
            torch.cuda.synchronize()
            self.step_started = time.perf_counter()

        def on_pre_optimizer_step(self, args_, state, control, model=None, **kwargs):
            grads = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
            if not grads or any(not torch.isfinite(g).all().item() for g in grads):
                raise ValueError("Missing/nonfinite adapter gradients; stop before optimizer update")
            self.finite_gradients += 1

        def on_step_end(self, args_, state, control, **kwargs):
            torch.cuda.synchronize()
            self.step_times.append(time.perf_counter() - self.step_started)
            if args.stop_after is not None and state.global_step >= args.stop_after:
                control.should_training_stop = True
                control.should_save = control.should_evaluate = True
            if args.smoke and state.global_step >= args.smoke_steps:
                control.should_save = control.should_evaluate = True
            return control

        def on_epoch_end(self, args_, state, control, **kwargs):
            control.should_save = control.should_evaluate = True
            return control

        def on_save(self, args_, state, control, **kwargs):
            write_json(output / f"checkpoint-{state.global_step}" / "run_identity.json", identity)

    class AssistantTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            labels = inputs["labels"]
            outputs = model(**{k: v for k, v in inputs.items() if k != "labels"}, use_cache=False)
            total, count = assistant_ce(outputs.logits, labels)
            loss = total / (num_items_in_batch if num_items_in_batch is not None else count)
            if not torch.isfinite(loss).item():
                raise ValueError("Nonfinite assistant-token loss")
            return (loss, outputs) if return_outputs else loss

        def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
            # Trainer's default example-mean loss weights short and long answers equally.
            # Compute the exact token-weighted validation criterion instead.
            was_training = self.model.training
            self.model.eval()
            total, tokens = 0.0, 0
            data = eval_dataset if eval_dataset is not None else self.eval_dataset
            if args.smoke:
                data = [max(data, key=lambda f: len(f["input_ids"]))]
            with torch.no_grad():
                for start in range(0, len(data), self.args.per_device_eval_batch_size):
                    features = data[start:start + self.args.per_device_eval_batch_size]
                    batch = self._prepare_inputs(self.data_collator(features))
                    with self.compute_loss_context_manager():
                        outputs = self.model(**{k: v for k, v in batch.items() if k != "labels"}, use_cache=False)
                        loss, count = assistant_ce(outputs.logits, batch["labels"])
                    total += loss.item()
                    tokens += count.item()
            metrics = {metric_key_prefix + "_loss": total / tokens, metric_key_prefix + "_assistant_tokens": tokens}
            if not math.isfinite(metrics[metric_key_prefix + "_loss"]):
                raise ValueError("Nonfinite validation loss")
            self.log(metrics)
            self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, metrics)
            if was_training:
                self.model.train()
            return metrics

    try:
        model = load_model(spec, precision, training=True)
        model, adapter_info = attach_adapters(model, args.model)
        write_json(output / "adapter_parameters.json", adapter_info)
        if args.resume:
            # Verify the adapter can load before Trainer restores optimizer/scheduler/RNG.
            from peft import set_peft_model_state_dict
            from safetensors.torch import load_file
            loaded = set_peft_model_state_dict(model, load_file(str(Path(args.resume) / "adapter_model.safetensors")))
            if loaded.unexpected_keys:
                raise ValueError(f"Unexpected adapter checkpoint keys: {loaded.unexpected_keys}")
        # The longest TRAIN example tests real forward/backward memory before full training.
        longest = max(train_data, key=lambda f: len(f["input_ids"]))
        collator = AssistantCollator(tokenizer)
        dtype = torch.bfloat16 if precision == "bf16" else torch.float16
        model.train()
        with torch.autocast("cuda", dtype=dtype):
            batch = {k: v.to("cuda") for k, v in collator([longest]).items()}
            prediction = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], use_cache=False)
            total, count = assistant_ce(prediction.logits, batch["labels"])
            loss = total / count
        if not torch.isfinite(loss).item():
            raise ValueError("Nonfinite longest-example loss")
        loss.backward()
        if any(not torch.isfinite(p.grad).all().item() for p in model.parameters() if p.grad is not None):
            raise ValueError("Nonfinite longest-example gradients")
        model.zero_grad(set_to_none=True)
        del prediction, batch, total, count, loss
        set_seed(SETTINGS["seed"])
        checks = Checks()
        trainer = AssistantTrainer(model=model, args=training_args,
            train_dataset=train_data, eval_dataset=val_data, data_collator=collator,
            callbacks=[checks])
        # compute_loss consumes Trainer's accumulation-window supervised token count.
        trainer.model_accepts_loss_kwargs = True
        result = trainer.train(resume_from_checkpoint=args.resume)
        expected_steps = args.smoke_steps if args.smoke else training_steps(len(train_data))
        stopped = args.stop_after is not None and trainer.state.global_step < expected_steps
        if not stopped and trainer.state.global_step < expected_steps:
            raise ValueError("Training ended before the fixed protocol completed")
        if not stopped:
            adapter_dir = output / "adapter"
            trainer.save_model(str(adapter_dir))
            tokenizer.save_pretrained(adapter_dir)
            write_json(adapter_dir / "run_identity.json", identity)
            # Reload the saved best adapter and compare fixed train-example logits.
            model.eval()
            inputs = {k: v.to("cuda") for k, v in collator([longest]).items() if k != "labels"}
            with torch.no_grad(), torch.autocast("cuda", dtype=dtype):
                before = model(**inputs, logits_to_keep=1).logits.detach().clone()
                model.load_adapter(str(adapter_dir), adapter_name="load_check")
                model.set_adapter("load_check")
                after = model(**inputs, logits_to_keep=1).logits
            if not torch.allclose(before, after, rtol=0.01, atol=0.01):
                raise ValueError("Saved best adapter reload changed logits")
            model.set_adapter("default")
            model.delete_adapter("load_check")
        steps = checks.step_times
        report = {"identity": identity, "gpu": gpu, "adapter_parameters": adapter_info["trainable_parameters"],
                  "completed_steps": trainer.state.global_step, "resumed_from_step": completed,
                  "resumed_step_observed": checks.resumed_step_observed,
                  "finite_gradient_steps": checks.finite_gradients, "stopped_for_resume": stopped,
                  "adapter_load_verified": not stopped,
                  "gpu_smoke_passed": args.smoke and not stopped and completed > 0 and checks.finite_gradients > 0,
                  "best_validation_loss": trainer.state.best_metric,
                  "best_checkpoint": trainer.state.best_model_checkpoint,
                  "wall_seconds": time.perf_counter() - started,
                  "mean_optimizer_step_seconds": sum(steps) / len(steps) if steps else None,
                  "estimated_full_training_seconds": (sum(steps) / len(steps)) * training_steps(len(train_data)) if steps else None,
                  "runtime_estimate_note": "Optimizer-step estimate excludes validation, checkpoint IO and startup.",
                  "peak_vram_allocated_bytes": torch.cuda.max_memory_allocated(),
                  "peak_vram_reserved_bytes": torch.cuda.max_memory_reserved(),
                  "metrics": result.metrics, "cost_usd": 0, "provider": "Kaggle private notebook"}
        write_json(output / "metrics.json", report)
    except torch.cuda.OutOfMemoryError as error:
        write_json(output / "memory_failure.json", {"identity": identity, "gpu": gpu,
            "peak_vram_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_vram_reserved_bytes": torch.cuda.max_memory_reserved(),
            "memory_summary": torch.cuda.memory_summary(), "error": str(error)})
        raise SystemExit("GPU memory exhausted. Settings/data unchanged; see memory_failure.json") from error


if __name__ == "__main__":
    main()
