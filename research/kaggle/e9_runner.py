"""Kaggle-side runner for E9 (foundation-model probes). Installs timm, then runs the same
extract_and_probe.main() used locally -- Kaggle's internet is fast and reliable for the HF/torch-hub
downloads that stalled locally (2026-10-29: RedDino and DINOv2 both hung on this machine's network path).

Code dataset layout expected (see launch.py's cmd_push_e9): research/malaria/, research/experiments/
E9_foundation_model_probes/{backbones.py, extract_and_probe.py}, research/splits/.
"""
import subprocess
import sys
import zipfile
from pathlib import Path

WORK = Path("/kaggle/working")
CODE = WORK / "code"


def locate_code():
    for p in Path("/kaggle/input").rglob("train.py"):
        if p.parent.name == "research" and (p.parent / "malaria").is_dir():
            return p.parent
    for z in Path("/kaggle/input").rglob("danet_code.zip"):
        zipfile.ZipFile(z).extractall(WORK / "unzipped")
        return WORK / "unzipped" / "research"
    raise FileNotFoundError("code dataset not attached")


def main():
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "timm"], check=True)
    src = locate_code()
    import shutil
    if CODE.exists():
        shutil.rmtree(CODE)
    shutil.copytree(src, CODE / "research")
    with open(WORK / "gpu.txt", "w") as f:
        subprocess.run(["nvidia-smi"], stdout=f, stderr=subprocess.STDOUT)
    rc = subprocess.run([sys.executable, "-m",
                        "research.experiments.E9_foundation_model_probes.extract_and_probe"], cwd=CODE).returncode
    # copy the experiment's outputs (results.json, features/, available_backbones.json) up to /kaggle/working
    out_dir = CODE / "research" / "experiments" / "E9_foundation_model_probes"
    dest = WORK / "E9_outputs"
    dest.mkdir(exist_ok=True)
    for f in ["results.json", "available_backbones.json"]:
        p = out_dir / f
        if p.exists():
            shutil.copy(p, dest / f)
    feat_dir = out_dir / "features"
    if feat_dir.exists():
        shutil.copytree(feat_dir, dest / "features", dirs_exist_ok=True)
    print(f"E9 runner exiting with code {rc}")
    sys.exit(rc)


if __name__ == "__main__":
    main()
