"""Merge a full-run adapter on CPU, convert to GGUF, and quantize to Q4_K_M."""
import argparse
import json
import subprocess
import sys
from pathlib import Path

from students.config import SETTINGS, read_lock
from students.data import sha256, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=["llama", "gemma"])
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--lock", required=True)
    parser.add_argument("--convert", required=True, help="llama.cpp convert_hf_to_gguf.py")
    parser.add_argument("--quantize", required=True, help="llama-quantize executable")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    lock = read_lock(args.lock, verified=True)
    spec = lock["models"][args.model]
    identity = json.loads((Path(args.adapter) / "run_identity.json").read_text())
    if identity["spec"] != spec or identity["settings"] != SETTINGS or identity["smoke_steps"] is not None:
        raise ValueError("Only a full-run adapter with matching protocol/revision may be exported")
    output = Path(args.out)
    if output.exists():
        raise ValueError("Export output exists; choose a new directory")
    output.mkdir(parents=True)
    import torch
    from peft import PeftModel
    from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText
    from students.modeling import load_tokenizer
    config = AutoConfig.from_pretrained(spec["model_id"], revision=spec["revision"], trust_remote_code=False)
    loader = AutoModelForImageTextToText if args.model == "gemma" else AutoModelForCausalLM
    model = loader.from_pretrained(spec["model_id"], revision=spec["revision"], config=config,
                                  dtype=torch.float16, device_map={"": "cpu"}, trust_remote_code=False)
    model = PeftModel.from_pretrained(model, args.adapter, is_trainable=False).merge_and_unload(safe_merge=True)
    merged = output / "merged_hf"
    model.save_pretrained(merged, safe_serialization=True)
    load_tokenizer(spec).save_pretrained(merged)
    del model
    unquantized, quantized = output / "merged-f16.gguf", output / "adapted-Q4_K_M.gguf"
    commands = [[sys.executable, args.convert, str(merged), "--outfile", str(unquantized), "--outtype", "f16"],
                [args.quantize, str(unquantized), str(quantized), "Q4_K_M"]]
    with (output / "conversion.log").open("w") as log:
        for command in commands:
            subprocess.run(command, check=True, stdout=log, stderr=log)
    with quantized.open("rb") as handle:
        if handle.read(4) != b"GGUF":
            raise ValueError("Quantization did not produce a GGUF")
    write_json(output / "export_manifest.json", {"model_id": spec["model_id"], "revision": spec["revision"],
        "adapter_sha256": sha256(Path(args.adapter) / "adapter_model.safetensors"),
        "gguf_sha256": sha256(quantized), "quantization": "Q4_K_M", "commands": commands,
        "merge_command": sys.argv, "converter_sha256": sha256(args.convert), "quantizer_sha256": sha256(args.quantize)})


if __name__ == "__main__":
    main()
