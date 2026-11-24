"""Regenerate the private Kaggle orchestration notebook from reviewable source."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--include-gemma", action="store_true", help="Generate with the optional second student enabled")
    args = parser.parse_args()
    cells = []

    def markdown(source):
        cells.append({"cell_type": "markdown", "metadata": {}, "source": source.strip().splitlines(keepends=True)})

    def code(source):
        source = source.replace("INCLUDE_GEMMA = False", f"INCLUDE_GEMMA = {args.include_gemma}")
        cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": source.strip().splitlines(keepends=True)})

    markdown("""
# Private Amharic student bakeoff

Attach the **private** allowlisted input bundle. Use Python 3.12 and exactly one
NVIDIA GPU with nominal >=16 GB VRAM. Dataset review, corrected-answer merge,
rejected-answer dispositions and 100 native-speaker spot checks must be complete.
This notebook defaults to Llama only. Review its results before enabling
`INCLUDE_GEMMA` for a separate second-stage run directory. Each selected model
needs tokenizer preflight and GPU smoke/checkpoint-resume tests before training.
Memory failures stop without changing any training settings or removing rows.

Keep this notebook and its outputs private. Before ending the session, save a
private notebook version **with outputs**, download the outputs, and verify the
artifact hash index. Restoring only an adapter cannot resume completed training.
""")
    code("""
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

if sys.version_info[:2] != (3, 12):
    raise RuntimeError('Use Python 3.12 for the pinned package candidates.')
BUNDLE = Path('/kaggle/input/amharic-bakeoff-private')  # Set your PRIVATE dataset path.
REPO = BUNDLE / 'repo'
DATASET = BUNDLE / 'frozen'
WORK = Path('/kaggle/working/bakeoff')
WORK.mkdir(parents=True, exist_ok=True)
LOG = WORK / 'experiments/LOG.md'
LOG.parent.mkdir(exist_ok=True)
if not LOG.exists():
    shutil.copyfile(REPO / 'experiments/LOG.md', LOG)
os.environ['HF_HOME'] = '/kaggle/temp/amharic-bakeoff-hf'  # Do not persist model cache.
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
USE_HF_SECRET = False  # Enable only if an HF_TOKEN Kaggle Secret is configured.
if USE_HF_SECRET:
    from kaggle_secrets import UserSecretsClient
    os.environ['HF_TOKEN'] = UserSecretsClient().get_secret('HF_TOKEN')
INCLUDE_GEMMA = False  # Enable only after reviewing the Llama results.
MODELS = ['llama'] + (['gemma'] if INCLUDE_GEMMA else [])
CANDIDATE = REPO / 'students' / ('model_lock.candidate.json' if INCLUDE_GEMMA else 'model_lock.llama.candidate.json')
FULL_RESUME = {key: None for key in MODELS}  # Restore private output dirs first.
sys.path.insert(0, str(REPO))

def run(module, *arguments):
    command = [sys.executable, '-m', module, *map(str, arguments)]
    subprocess.run(command, cwd=REPO, check=True)
    return shlex.join(command)

def record(kind, artifact, command):
    run('students.record', '--kind', kind, '--artifact', artifact,
        '--command', command, '--log', LOG)
""")
    code("""
LOCK = WORK / 'model_lock.json'
requirements = WORK / 'requirements.lock.txt' if LOCK.exists() else REPO / 'students/requirements.candidate.txt'
subprocess.run([sys.executable, '-m', 'pip', 'install', '-r', str(requirements)], check=True)
from students.config import SETTINGS, read_lock, training_steps
from students.data import load_frozen, sha256, write_json
from students.modeling import gpu_precision, package_versions
manifest, split = load_frozen(DATASET)
precision, gpu = gpu_precision()
print(gpu)
if LOCK.exists():
    for key in MODELS:
        read_lock(LOCK, verified=True, model=key)
PREFLIGHT = WORK / 'preflight.json'
command = run('students.preflight', '--dataset', DATASET, '--lock', CANDIDATE, '--out', PREFLIGHT)
record('preflight', PREFLIGHT, command)
""")
    markdown("""
## Smoke and resume checks

Each smoke keeps rank, sequence cap, optimizer and accumulation fixed. The longest
training conversation exercises forward/backward memory. Step 2 is persisted;
the next process restores optimizer/scheduler/RNG and completes step 4. Full
training requires each selected model to pass smoke/resume/loading checks.
""")
    code("""
candidate = read_lock(CANDIDATE)
for key in MODELS:
    directory = WORK / ('smoke-' + key)
    metrics_path = directory / 'metrics.json'
    if metrics_path.exists():
        existing = json.loads(metrics_path.read_text())
        if existing.get('gpu_smoke_passed'):
            identity = existing['identity']
            assert identity['spec'] == candidate['models'][key]
            assert identity['manifest_sha256'] == sha256(DATASET / 'manifest.json')
            assert identity['packages'] == package_versions()
            assert identity['precision'] == precision
            assert existing['adapter_load_verified'] and existing['resumed_from_step'] > 0
            continue
    common = ('--dataset', DATASET, '--model', key, '--lock', CANDIDATE, '--preflight', PREFLIGHT,
              '--out', directory, '--smoke', '--precision', precision)
    checkpoint = directory / 'checkpoint-2'
    if not checkpoint.exists():
        command = run('students.train', *common, '--stop-after', 2)
        record('training', metrics_path, command)
    command = run('students.train', *common, '--resume', checkpoint)
    record('training', metrics_path, command)
    report = json.loads(metrics_path.read_text())
    print(key, report['peak_vram_reserved_bytes'], report['mean_optimizer_step_seconds'],
          report['estimated_full_training_seconds'])
if not LOCK.exists():
    arguments = ['--candidate', CANDIDATE, '--llama-smoke', WORK / 'smoke-llama/metrics.json', '--out', LOCK]
    if 'gemma' in MODELS:
        arguments += ['--gemma-smoke', WORK / 'smoke-gemma/metrics.json']
    run('students.lock', *arguments)
verified = read_lock(LOCK, verified=True)
for key in MODELS:
    read_lock(LOCK, verified=True, model=key)
assert verified['packages'] == package_versions()
assert verified['manifest_sha256'] == sha256(DATASET / 'manifest.json')
""")
    markdown("""
## Full runs for selected models

Run only after reviewing the selected memory/runtime smoke reports. A full run uses the
unchanged frozen split and two epochs. Specify `FULL_RESUME` explicitly after
restoring a complete checkpoint to its working run path. An incomplete output
directory without an explicit resume raises an error instead of starting over.
""")
    code("""
expected_steps = training_steps(len(split['train_ids']))
for key in MODELS:
    directory = WORK / ('train-' + key)
    metrics_path = directory / 'metrics.json'
    if metrics_path.exists():
        report = json.loads(metrics_path.read_text())
        identity = report['identity']
        assert identity['spec'] == verified['models'][key]
        assert identity['manifest_sha256'] == sha256(DATASET / 'manifest.json')
        assert identity['packages'] == package_versions() and identity['precision'] == precision
        assert identity['settings'] == SETTINGS and identity['smoke_steps'] is None
        if report['completed_steps'] == expected_steps and report['adapter_load_verified']:
            print(key, 'completed run preserved')
            continue
    arguments = ['--dataset', DATASET, '--model', key, '--lock', LOCK,
                 '--preflight', PREFLIGHT, '--out', directory, '--precision', precision]
    if FULL_RESUME[key] is not None:
        arguments += ['--resume', FULL_RESUME[key]]
    command = run('students.train', *arguments)
    record('training', metrics_path, command)
""")
    code("""
EVALUATION = WORK / 'eval'
for key in MODELS:
    for adapted in (False, True):
        configuration = key + ('_adapted' if adapted else '_original')
        output = EVALUATION / (configuration + '.jsonl')
        arguments = ['--dataset', DATASET, '--lock', LOCK, '--model', key,
                     '--precision', precision, '--out', output]
        if adapted:
            arguments += ['--adapter', WORK / ('train-' + key) / 'adapter']
        if output.exists() or output.with_suffix('.meta.json').exists():
            arguments += ['--resume']
        command = run('students.evaluate', *arguments)
        record('evaluation', output.with_suffix('.references.json'), command)
HUMAN = WORK / 'human'
if not HUMAN.exists():
    run('eval.human', '--dataset', DATASET, '--outputs', EVALUATION, '--out', HUMAN)
COMPARISON = WORK / 'comparison.json'
command = run('eval.report', '--dataset', DATASET, '--outputs', EVALUATION,
              '--human', HUMAN, '--out', COMPARISON)
record('evaluation', COMPARISON, command)
print('Give native speakers only:', HUMAN / 'public')
print('Retain private mapping:', HUMAN / 'private/mapping.json')
""")
    markdown("""
## Persist and finish locally

Review `comparison.json` coverage and finish all blind ratings/preferences locally.
Rerun `eval.report --require-complete` after the sheets return. Use the run guide
for CPU adapter merge, Q4_K_M conversion, identical deployment prompts and local
RAM/speed/quality measurements. Bring the working experiment-log entries back
to the repository. Training settings, revisions, dataset hashes, runtime, cost,
results and deployment outcomes must be recorded before accepting the comparison.

Save a private notebook version with **all** working outputs before ending this
session. Download them and verify the index below. Preserve optimizer, scheduler,
RNG and Trainer state alongside adapters. This notebook does not upload or publish.
""")
    code("""
index = WORK / 'artifact_manifest.json'
files = {str(path.relative_to(WORK)): sha256(path)
         for path in sorted(WORK.rglob('*')) if path.is_file() and path != index}
write_json(index, {'dataset_manifest_sha256': sha256(DATASET / 'manifest.json'),
                  'files': files, 'visibility': 'private', 'packages': package_versions()})
print('Persist private outputs and verify:', index)
""")
    notebook = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                "language_info": {"name": "python", "version": "3.12"},
                "kaggle": {"isInternetEnabled": True, "isGpuEnabled": True}}, "nbformat": 4, "nbformat_minor": 5}
    # Stable cell IDs are part of nbformat 4.5.
    for i, cell in enumerate(cells):
        cell["id"] = f"bakeoff-{i:02d}"
    Path(__file__).with_name("kaggle_bakeoff.ipynb").write_text(json.dumps(notebook, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
