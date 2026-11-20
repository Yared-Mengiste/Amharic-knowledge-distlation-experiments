"""Record a concrete artifact's provenance and command in the experiment log."""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from students.data import sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--command", required=True, help="Exact command; never include credentials")
    parser.add_argument("--kind", choices=["preflight", "training", "evaluation", "deployment"], required=True)
    parser.add_argument("--log", default="experiments/LOG.md")
    args = parser.parse_args()
    artifact = Path(args.artifact)
    result = json.loads(artifact.read_text())
    provenance = {k: result[k] for k in ("identity", "evaluation_metadata", "manifest_sha256", "gpu", "adapter_parameters", "completed_steps", "resumed_from_step", "best_validation_loss", "wall_seconds", "mean_optimizer_step_seconds", "peak_vram_allocated_bytes", "cost_usd", "comparison_ready", "reference_coverage", "peak_process_ram_bytes", "tokens_per_second", "quantization") if k in result}
    today = datetime.now(timezone.utc).date().isoformat()
    entry = f"\n### {today} — Student bakeoff {args.kind}\n\n- **Command:** `{args.command}`\n- **Artifact:** `{artifact}` (SHA256 `{sha256(artifact)}`).\n- **Recorded provenance/results:**\n\n```json\n{json.dumps(provenance, ensure_ascii=False, indent=2)}\n```\n"
    path = Path(args.log)
    existing = path.read_text()
    marker = "\n---\n"
    if marker in existing:
        before, after = existing.split(marker, 1)
        path.write_text(before + marker + entry + after)
    else:
        path.write_text(existing + entry)


if __name__ == "__main__":
    main()
