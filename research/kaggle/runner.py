"""Kaggle-side job runner. launch.py copies this file into a kernel folder and fills in JOBS.

Finds the code dataset under /kaggle/input, copies it to /kaggle/working/code, runs every job with
research.train, and leaves results in /kaggle/working/runs (downloaded later by `kaggle kernels output`).
"""
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import zipfile
from pathlib import Path

SESSION_HOURS = 11.0   # Kaggle stops sessions at 12 h; leave time to save outputs
MIN_START_HOURS = 0.75 # do not start a new run with less session time than this
T_SESSION = time.time()
JOBS = []          # filled in by launch.py: [{"config": ..., "split": ..., "repeat": .., "fold": .., "seed": .., "set": [...]}]

WORK = Path("/kaggle/working")
CODE, RUNS = WORK / "code", WORK / "runs"


def locate_code():
    for p in Path("/kaggle/input").rglob("train.py"):
        if p.parent.name == "research" and (p.parent / "malaria").is_dir():
            return p.parent
    for z in Path("/kaggle/input").rglob("danet_code.zip"):
        zipfile.ZipFile(z).extractall(WORK / "unzipped")
        return WORK / "unzipped" / "research"
    raise FileNotFoundError("code dataset not attached")


def restore_partial_runs():
    """Copy unfinished runs (last.pth, no metrics.json) from attached earlier job outputs so they resume."""
    for last in Path("/kaggle/input").rglob("runs/*/*/last.pth"):
        d = last.parent
        dest = RUNS / d.parent.name / d.name
        if not (d / "metrics.json").exists() and not dest.exists():
            shutil.copytree(d, dest)
            print("restored partial run", dest, flush=True)


def main():
    RUNS.mkdir(parents=True, exist_ok=True)
    restore_partial_runs()
    src = locate_code()
    if CODE.exists():
        shutil.rmtree(CODE)
    shutil.copytree(src, CODE / "research")
    with open(RUNS / "gpu.txt", "w") as f:
        subprocess.run(["nvidia-smi"], stdout=f, stderr=subprocess.STDOUT)
    n_gpu = max(1, len(subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True).stdout.splitlines()))
    (RUNS / "logs").mkdir(exist_ok=True)
    summary, lock, todo = [], threading.Lock(), list(enumerate(JOBS))

    def lane(gpu):                         # one job at a time per GPU; quota is charged per session, not per GPU
        while True:
            with lock:
                left = SESSION_HOURS - (time.time() - T_SESSION) / 3600
                if not todo or left < MIN_START_HOURS:
                    return
                i, job = todo.pop(0)
            cmd = [sys.executable, "-m", "research.train", "--config", f"research/configs/{job['config']}.json",
                   "--split", job["split"], "--repeat", str(job.get("repeat", 0)), "--fold", str(job.get("fold", 0)),
                   "--seed", str(job.get("seed", 42)), "--out-root", str(RUNS)]
            cmd += ["--set", *job.get("set", []), f"max_hours={left - 0.25:.2f}"]   # train.py saves last.pth and exits
            print(f"\n>>> [gpu {gpu}] job {i}:", " ".join(cmd[2:]), flush=True)
            t0 = time.time()
            with open(RUNS / "logs" / f"job{i:02d}.log", "w") as log:
                rc = subprocess.run(cmd, cwd=CODE, stdout=log, stderr=subprocess.STDOUT,
                                    env={**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)}).returncode
            with lock:
                summary.append(dict(i=i, gpu=gpu, job=job, returncode=rc, hours=(time.time() - t0) / 3600))
                json.dump(sorted(summary, key=lambda s: s["i"]), open(RUNS / "summary.json", "w"), indent=1)
            print(f"<<< [gpu {gpu}] job {i} rc={rc} {(time.time() - t0) / 60:.1f} min", flush=True)

    threads = [threading.Thread(target=lane, args=(g,)) for g in range(n_gpu)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    skipped = [j for _, j in todo]
    if skipped:
        json.dump(skipped, open(RUNS / "not_started.json", "w"), indent=1)
        print(f"session time budget reached: {len(skipped)} jobs not started (runs/not_started.json)")
    for p in RUNS.rglob("last.pth"):            # resume checkpoints are not needed once a run finished
        if (p.parent / "metrics.json").exists():
            p.unlink()
    shutil.rmtree(CODE, ignore_errors=True)       # keep the downloaded output small
    print(json.dumps(sorted(summary, key=lambda s: s["i"]), indent=1))


if __name__ == "__main__":
    main()
