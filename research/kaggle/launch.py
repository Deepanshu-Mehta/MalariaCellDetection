"""Run research jobs on a Kaggle GPU from this machine.

    # 1. upload / update the code as a private Kaggle dataset (after any code or split change)
    python -m research.kaggle.launch code -m "message"
    # 2. push a GPU job (each job: config:split[:repeat:fold][:seed=..][:key=value ...])
    python -m research.kaggle.launch push --slug danet-phase0-timing \
        --job E1_danet:grouped:0:0 --job E1r_resnet18:grouped:0:0
    # 3. check status / download results into research/runs_kaggle/<slug>/
    python -m research.kaggle.launch status --slug danet-phase0-timing
    python -m research.kaggle.launch fetch --slug danet-phase0-timing

Uses the Kaggle CLI from the active environment and the token in ~/.kaggle/access_token.
"""
import argparse
import json
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]                 # research/
USER = "deepanshum27"
CODE_DATASET = f"{USER}/danet-audit-code"
NIH_DATASET = "iarunava/cell-images-for-detecting-malaria"
PACKED_DATASET = f"{USER}/nih-malaria-cells-packed"     # one-file copy of the images (scripts/pack_images.py)
STAGE = ROOT / "kaggle" / "_stage"
CODE_FILES = ["__init__.py", "train.py", "malaria/*.py", "configs/*.json", "splits/*",
              "experiments/__init__.py",
              "experiments/E9_foundation_model_probes/__init__.py",
              "experiments/E9_foundation_model_probes/backbones.py",
              "experiments/E9_foundation_model_probes/extract_and_probe.py"]


def kaggle(*args, check=True):
    exe = Path(sys.executable).with_name("kaggle")
    return subprocess.run([str(exe), *args], check=check, text=True, capture_output=True)


def stage_code():
    d = STAGE / "code"
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)
    with zipfile.ZipFile(d / "danet_code.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for pat in CODE_FILES:
            for f in sorted(ROOT.glob(pat)):
                z.write(f, Path("research") / f.relative_to(ROOT))
    json.dump({"title": "danet-audit-code", "id": CODE_DATASET, "licenses": [{"name": "other"}]},
              open(d / "dataset-metadata.json", "w"))
    return d


def cmd_code(a):
    d = stage_code()
    exists = kaggle("datasets", "status", CODE_DATASET, check=False).returncode == 0
    r = kaggle("datasets", "version", "-p", str(d), "-m", a.message, check=False) if exists else \
        kaggle("datasets", "create", "-p", str(d), check=False)
    print(r.stdout, r.stderr)


def parse_job(spec):
    parts = spec.split(":")
    job = {"config": parts[0], "split": parts[1], "set": []}
    pos = [p for p in parts[2:] if "=" not in p]
    if job["split"] == "grouped" and pos:
        job["repeat"], job["fold"] = int(pos[0]), int(pos[1])
    for p in parts[2:]:
        if p.startswith("seed="):
            job["seed"] = int(p.split("=", 1)[1])
        elif "=" in p:
            job["set"].append(p)
    return job


def cmd_push(a):
    d = STAGE / a.slug
    d.mkdir(parents=True, exist_ok=True)
    jobs = [parse_job(s) for s in a.job]
    src = (ROOT / "kaggle" / "runner.py").read_text()
    (d / "runner.py").write_text(src.replace("JOBS = []", f"JOBS = {json.dumps(jobs)}", 1))
    json.dump({"id": f"{USER}/{a.slug}", "title": a.slug, "code_file": "runner.py", "language": "python",
               "kernel_type": "script", "is_private": True, "enable_gpu": True, "enable_internet": True,
               "dataset_sources": [NIH_DATASET, PACKED_DATASET, CODE_DATASET], "competition_sources": [], "kernel_sources": [f"{USER}/{k}" for k in a.resume_from]},
              open(d / "kernel-metadata.json", "w"), indent=1)
    r = kaggle("kernels", "push", "-p", str(d), "--accelerator", a.accelerator, check=False)
    print(json.dumps(jobs, indent=1)); print(r.stdout, r.stderr)


def cmd_push_e9(a):
    d = STAGE / a.slug
    d.mkdir(parents=True, exist_ok=True)
    shutil.copy(ROOT / "kaggle" / "e9_runner.py", d / "e9_runner.py")
    json.dump({"id": f"{USER}/{a.slug}", "title": a.slug, "code_file": "e9_runner.py", "language": "python",
               "kernel_type": "script", "is_private": True, "enable_gpu": True, "enable_internet": True,
               "dataset_sources": [NIH_DATASET, PACKED_DATASET, CODE_DATASET],
               "competition_sources": [], "kernel_sources": []},
              open(d / "kernel-metadata.json", "w"), indent=1)
    r = kaggle("kernels", "push", "-p", str(d), "--accelerator", a.accelerator, check=False)
    print(r.stdout, r.stderr)


def cmd_status(a):
    r = kaggle("kernels", "status", f"{USER}/{a.slug}", check=False)
    print(r.stdout, r.stderr)


def cmd_fetch(a):
    out = ROOT / "runs_kaggle" / a.slug
    out.mkdir(parents=True, exist_ok=True)
    r = kaggle("kernels", "output", f"{USER}/{a.slug}", "-p", str(out), "--force", check=False)
    print(r.stdout[-2000:], r.stderr[-2000:])
    print("saved to", out)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("code"); c.add_argument("-m", "--message", default="update")
    p = sub.add_parser("push"); p.add_argument("--slug", required=True); p.add_argument("--job", action="append", required=True)
    p.add_argument("--accelerator", default="NvidiaTeslaT4")
    p.add_argument("--resume-from", action="append", default=[], help="earlier job slug whose unfinished runs resume")
    pe = sub.add_parser("push-e9"); pe.add_argument("--slug", required=True)
    pe.add_argument("--accelerator", default="NvidiaTeslaT4")
    for name in ["status", "fetch"]:
        s = sub.add_parser(name); s.add_argument("--slug", required=True)
    a = ap.parse_args()
    dict(code=cmd_code, push=cmd_push, status=cmd_status, fetch=cmd_fetch, **{"push-e9": cmd_push_e9})[a.cmd](a)


if __name__ == "__main__":
    main()
