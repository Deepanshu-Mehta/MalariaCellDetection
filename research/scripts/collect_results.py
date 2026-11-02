"""Gather every run into one table: research/results/all_runs.csv (+ per-experiment summary).

    arch -arm64 .venv/bin/python -m research.scripts.collect_results

Scans research/runs_kaggle/<job>/runs/** and research/runs/** . A run counts as
  finished  metrics.json exists
  crashed   listed in a job's summary.json with a non-zero return code (log path kept)
  partial   train_log.csv exists but no metrics.json (session ended before the run finished)
  failed_to_train  finished, but validation loss never fell below 0.6 (stuck at chance, ln 2 = 0.693)
Jobs whose slug contains 'smoke' are pipeline checks and are excluded.
"""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results"


def run_dirs():
    for base in [ROOT / "runs_kaggle", ROOT / "runs"]:
        if not base.exists():
            continue
        for cfg in base.rglob("config.json"):
            if "smoke" in str(cfg.relative_to(base)).split("/")[0]:
                continue
            yield base, cfg.parent


def main():
    rows = []
    for base, d in run_dirs():
        c = json.load(open(d / "config.json"))
        row = dict(experiment=c["cfg"]["name"].split("_")[0], config=c["cfg"]["name"], split=c["split"],
                   tag=c["tag"], train_seed=c["cfg"]["train_seed"], epochs=c["cfg"]["epochs"],
                   source=str(d.relative_to(ROOT)), device=c.get("device"))
        overrides = {k: v for k, v in c["cfg"].items() if k in ("colour_jitter", "label_mode", "label_draw",
                                                                 "gate", "dilation_mode", "norm", "img_size")}
        row.update({f"cfg_{k}": json.dumps(v) for k, v in overrides.items()})
        if (d / "metrics.json").exists():
            m = json.load(open(d / "metrics.json"))
            min_val = pd.read_csv(d / "train_log.csv").val_loss.min()
            # Near-chance val loss (~ln 2) is the EXPECTED outcome for a random-patient-label permutation
            # null (E3) -- it is not a training failure there, so only flag it for real-label runs.
            is_real_label = c["cfg"].get("label_mode", "real") == "real"
            status = "finished" if (min_val < 0.6 or not is_real_label) else "failed_to_train"
            row.update(status=status, min_val_loss=round(min_val, 4),
                       best_epoch=m["best_epoch"], train_hours=m["train_hours"], sec_per_epoch=m["sec_per_epoch"])
            row.update({k: m["cell"].get(k) for k in ("BACC", "SE", "SP", "ROC_AUC", "PR_AUC", "ECE", "ACC", "n")})
            row.update({k: m["patient"].get(k) for k in ("SE_patient_macro", "SP_patient_macro",
                                                          "SP_healthy_only", "SP_infected_patients")})
        else:
            row["status"] = "partial" if (d / "train_log.csv").exists() else "started"
        rows.append(row)
    for summ in (ROOT / "runs_kaggle").glob("*/runs/summary.json") if (ROOT / "runs_kaggle").exists() else []:
        job_slug = summ.parts[-3]
        if "smoke" in job_slug:
            continue
        for s in json.load(open(summ)):
            if s["returncode"] != 0:
                j = s["job"]
                rows.append(dict(experiment=j["config"].split("_")[0], config=j["config"], split=j["split"],
                                 tag=f"{j['split']}_r{j.get('repeat', 0)}f{j.get('fold', 0)}", status="crashed",
                                 source=str((summ.parent / "logs" / f"job{s['i']:02d}.log").relative_to(ROOT))))
    OUT.mkdir(exist_ok=True)
    t = pd.DataFrame(rows)
    if t.empty:
        print("no runs found"); return
    t = t.sort_values(["experiment", "config", "split", "tag", "train_seed"])
    t.to_csv(OUT / "all_runs.csv", index=False)
    fin = t[t.status == "finished"]
    summ = fin.groupby(["experiment", "config", "split"]).agg(
        runs=("tag", "size"), BACC=("BACC", "mean"), BACC_sd=("BACC", "std"), SE=("SE", "mean"),
        SP=("SP", "mean"), ROC_AUC=("ROC_AUC", "mean"), hours=("train_hours", "sum")).round(3)
    summ.to_csv(OUT / "experiment_summary.csv")
    print(t.status.value_counts().to_string()); print(summ.to_string())


if __name__ == "__main__":
    main()
