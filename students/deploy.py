"""Local Q4_K_M llama.cpp benchmark on identical deployment prompts (Linux)."""
import argparse
import json
import socket
import shutil
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

from eval.human import answer_hash
from students.config import SETTINGS, read_lock
from students.data import load_frozen, read_jsonl, sha256, write_csv, write_json
from students.modeling import load_tokenizer
from students.tokenization import ENDINGS, render_prompt


def process_ram(pid):
    try:
        lines = Path(f"/proc/{pid}/status").read_text().splitlines()
        values = [int(line.split()[1]) * 1024 for line in lines if line.startswith(("VmHWM:", "VmRSS:"))]
        return max(values, default=0)
    except FileNotFoundError:
        return 0


def call_json(url, payload=None, timeout=120):
    data = json.dumps(payload).encode() if payload is not None else None
    request = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--prompts", required=True, help="Shared deployment prompt JSONL from the frozen eval set")
    parser.add_argument("--lock", required=True)
    parser.add_argument("--model", required=True, choices=["llama", "gemma"])
    parser.add_argument("--gguf", required=True)
    parser.add_argument("--adapter", required=True, help="Full-run adapter used for this GGUF export")
    parser.add_argument("--server", required=True, help="llama-server executable")
    parser.add_argument("--export-manifest", required=True, help="Records conversion/merge/quantization commands and GGUF provenance")
    parser.add_argument("--gpu-layers", type=int, default=0)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    load_frozen(args.dataset)
    lock = read_lock(args.lock, verified=True, model=args.model)
    export = json.loads(Path(args.export_manifest).read_text())
    if export["quantization"] != "Q4_K_M" or export["gguf_sha256"] != sha256(args.gguf) or export["adapter_sha256"] != sha256(Path(args.adapter) / "adapter_model.safetensors"):
        raise ValueError("GGUF export provenance differs from the requested Q4_K_M adapter")
    spec = lock["models"][args.model]
    if export["model_id"] != spec["model_id"] or export["revision"] != spec["revision"] or not export["commands"]:
        raise ValueError("GGUF export is missing model revision/conversion commands")
    prompts = read_jsonl(args.prompts)
    frozen = {r["id"]: r for r in read_jsonl(Path(args.dataset) / "pilot_eval.jsonl")}
    if not prompts or any(r != frozen.get(r["id"]) for r in prompts):
        raise ValueError("Deployment prompts must be unchanged frozen eval rows")
    tokenizer = load_tokenizer(spec)
    output = Path(args.out)
    if output.exists():
        raise ValueError("Deployment output already exists; choose a new run")
    output.mkdir(parents=True)
    shutil.copyfile(args.prompts, output / "deployment_prompts.jsonl")
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    version = subprocess.run([args.server, "--version"], capture_output=True, text=True, check=True)
    command = [args.server, "-m", args.gguf, "--host", "127.0.0.1", "--port", str(port), "-c", "4096", "-ngl", str(args.gpu_layers)]
    answers, peak = [], 0
    started = time.perf_counter()
    with (output / "server.log").open("w") as log:
        process = subprocess.Popen(command, stdout=log, stderr=log)
        try:
            url = f"http://127.0.0.1:{port}"
            deadline = time.perf_counter() + 180
            while True:
                if process.poll() is not None:
                    raise ValueError("llama-server exited; inspect server.log for GGUF support/memory errors")
                peak = max(peak, process_ram(process.pid))
                try:
                    call_json(url + "/health", timeout=2)
                    break
                except (urllib.error.URLError, TimeoutError):
                    if time.perf_counter() >= deadline:
                        raise ValueError("llama-server startup timed out")
                    time.sleep(0.25)
            for row in prompts:
                prompt = render_prompt(tokenizer, row["messages"])
                response = call_json(url + "/completion", {"prompt": prompt, "n_predict": SETTINGS["max_new_tokens"], "temperature": 0, "seed": SETTINGS["seed"], "cache_prompt": False, "stop": [ENDINGS[args.model]]}, timeout=900)
                peak = max(peak, process_ram(process.pid))
                timings = response["timings"]
                answers.append({"id": row["id"], "category": row["category"], "prompt": row["messages"][0]["content"], "answer": response["content"], "output_tokens": timings["predicted_n"], "tokens_per_second": timings["predicted_per_second"], "latency_seconds": timings["predicted_ms"] / 1000, "termination_reason": "length" if response.get("stopped_limit") else "eos"})
            (output / "answers.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in answers))
            write_csv(output / "quality_review.csv", [{"id": r["id"], "category": r["category"], "prompt": r["prompt"], "answer_sha256": answer_hash(r["answer"]), "answer": r["answer"], "correctness": "", "fluency": "", "instruction_following": "", "reviewer": "", "notes": ""} for r in answers], ["id", "category", "prompt", "answer_sha256", "answer", "correctness", "fluency", "instruction_following", "reviewer", "notes"])
            tokens = sum(r["output_tokens"] for r in answers)
            seconds = sum(r["latency_seconds"] for r in answers)
            write_json(output / "metrics.json", {"identity": {"model": args.model, "spec": spec, "adapter_sha256": export["adapter_sha256"], "gguf_sha256": export["gguf_sha256"], "deployment_prompts_sha256": sha256(args.prompts)}, "quantization": "Q4_K_M", "peak_process_ram_bytes": peak, "ram_measurement": "Linux process VmHWM/RSS; GPU allocations excluded", "tokens_per_second": tokens / seconds if seconds else None, "wall_seconds": time.perf_counter() - started, "server_command": command, "llama_cpp_version": version.stdout + version.stderr, "quality_review": "quality_review.csv (requires native-speaker completion)", "cost_usd": 0})
        finally:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


if __name__ == "__main__":
    main()
