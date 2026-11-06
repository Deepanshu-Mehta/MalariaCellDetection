"""Every data figure in the IEEE paper, drawn from the saved experiment outputs. Nothing here fits or tunes anything.

    arch -arm64 .venv/bin/python -m research.paper.ieee.make_figures

Writes vector PDFs to research/paper/ieee/figures/. Sample images are drawn with fixed seeds, never hand-picked.
"""
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import roc_curve

from research.experiments.E11_bbbc041_holdout.calculate import candidates, load, logit
from research.experiments.E2_shortcut_evidence.extract_embeddings import ARMS, ckpt_path
from research.experiments.E6b_five_checkpoint_external_rescore.score_external import CELLS_DIR, scored_table
from research.malaria import dark_spot
from research.malaria.data import image_root

HERE = Path(__file__).resolve().parent
R = HERE.parents[1]
X = R / "experiments"
OUT = HERE / "figures"
CACHE = HERE / "_cache"
B41 = R / "data_external" / "bbbc041_crops"

# Reference palette, first three categorical slots (validated all-pairs, light mode). Aqua is under 3:1, so every
# series also carries a line style and a legend or direct label.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df"
SEQ = ["#fcfcfb", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
COL, FULL = 3.5, 7.16

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Times", "STIXGeneral"], "mathtext.fontset": "stix",
    "font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 6.8, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6, "axes.spines.top": False,
    "axes.spines.right": False, "pdf.fonttype": 42, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    "legend.frameon": False, "axes.titleweight": "bold", "axes.titlelocation": "left",
})


def J(p):
    return json.load(open(p))


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.pdf")
    plt.close(fig)
    print("wrote", name)


def grid(ax, axis="x"):
    ax.grid(axis=axis, color=GRID, lw=0.5, zorder=0)
    ax.set_axisbelow(True)


# ----------------------------------------------------------------------------------------------------------------
def fig_forest():
    """Paired NIH differences, each against the 0.29-point minimum detectable effect."""
    e1, e1r, e4 = J(X / "E1_danet_leaky_vs_grouped/results.json"), J(X / "E1r_resnet_18_reference/results.json"), \
        J(X / "E4_m3_colour_augmentation_pre_registered/results_m3_rows.json")
    e9, e10, e8r = J(X / "E9_foundation_model_probes/results_vs_danet.json"), J(X / "E10_leave_session_out/part_b_results.json"), \
        J(X / "E8r_norm_layer_leaky_replication/results.json")
    feat = J(R / "exploratory/2026-11-02_paper_readiness/dark_spot_stress.json")["A_NIH_vs_DANet_patient_bootstrap"]["BACC_diff"]
    d = lambda o: (o["diff"], o["lo"], o["hi"])
    rows = [
        ("DANet $-$ dark-spot feature", tuple(feat), "baseline"),
        ("DANet $-$ DINOv2 probe", d(e9["dinov2_s"]["danet_minus_probe_BACC"]), "baseline"),
        ("DANet $-$ DinoBloom probe", d(e9["dinobloom_s"]["danet_minus_probe_BACC"]), "baseline"),
        ("DANet $-$ RedDino probe", d(e9["reddino_s"]["danet_minus_probe_BACC"]), "baseline"),
        ("DANet $-$ ResNet-18", d(e1r["danet_minus_resnet_grouped_BACC"]), "baseline"),
        ("Patient leakage (leaky $-$ grouped)", (e1["gap_leaky_minus_grouped"]["BACC"], None, None), "leak"),
        ("Session leakage (grouped $-$ session)", d(e10["gap"]["BACC"]), "leak"),
        ("ColorJitter, M3 hypothesis (H $-$ B1)", d(e4["H_minus_B1_BACC"]), "hyp"),
        ("LayerNorm $-$ InstanceNorm leak gain", (e8r["mean_LN_minus_IN"], *e8r["ci_95"]), "leak"),
    ]
    fig, ax = plt.subplots(figsize=(COL, 2.5))
    fig.subplots_adjust(left=0.45, right=0.735, top=0.93, bottom=0.17)
    ax.axvspan(-0.29, 0.29, color="#f0efec", zorder=0)
    ax.axvline(0, color=INK2, lw=0.6, zorder=1)
    y = np.arange(len(rows))[::-1]
    for yi, (lab, (m, lo, hi), kind) in zip(y, rows):
        if lo is not None:
            ax.plot([lo, hi], [yi, yi], color=INK, lw=1.2, solid_capstyle="round", zorder=3)
            ax.plot(m, yi, "o", ms=4.2, mfc=INK, mec="#fcfcfb", mew=0.8, zorder=4)
        else:
            ax.plot(m, yi, "o", ms=4.2, mfc="#fcfcfb", mec=INK, mew=1.0, zorder=4)
        txt = f"{m:+.2f}" + (f" [{lo:+.2f}, {hi:+.2f}]" if lo is not None else "\n(no paired CI)")
        ax.text(1.04, yi, txt, va="center", ha="left", fontsize=6.2, color=INK2, transform=ax.get_yaxis_transform())
    ax.set_yticks(y, [r[0] for r in rows])
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(-0.5, 1.6)
    ax.set_xticks([-0.5, 0, 0.5, 1.0, 1.5])
    ax.set_xlabel("Difference in NIH BACC (pp)")
    ax.text(0, len(rows) - 0.35, "MDE $\\pm$0.29", ha="center", va="bottom", fontsize=6.2, color=INK2)
    ax.spines["left"].set_visible(False)
    grid(ax)
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "fig_forest.pdf", bbox_inches=None)
    plt.close(fig)
    print("wrote fig_forest")


# ----------------------------------------------------------------------------------------------------------------
def fig_seeds():
    """E12: the same ten checkpoints on NIH and on MP-IDB."""
    s = J(X / "E12_retraining_variance/results.json")["seeds"]
    fig, ax = plt.subplots(figsize=(COL, 2.2))
    for f, col, ls in (("2", BLUE, "-"), ("3", ORANGE, "--")):
        for r in s["per_seed"][f]:
            a, b = r["NIH_BACC"], r["MP"]["BACC"]
            ax.plot([0, 1], [a, b], color=col, lw=0.9, ls=ls, alpha=0.9, zorder=2)
            mk = "s" if r["seed"] == "t0" else "o"
            for x, v in ((0, a), (1, b)):
                ax.plot(x, v, mk, ms=4.5, mfc=col, mec="#fcfcfb", mew=0.8, zorder=3)
    ax.set_xticks([0, 1], ["NIH test fold\n(unseen patients)", "MP-IDB\n(external)"])
    ax.set_xlim(-0.55, 1.75)
    ax.set_ylim(55, 100)
    ax.set_ylabel("Balanced accuracy (%)")
    sd_n, sd_m = s["pooled_NIH_BACC"], s["pooled_MP_BACC"]
    ax.text(-0.08, 93.5, f"seed SD {sd_n['sd']:.2f}\n[{sd_n['ci'][0]:.2f}, {sd_n['ci'][1]:.2f}]", fontsize=6.6, color=INK, va="top", ha="right")
    ax.text(1.08, 74, f"seed SD {sd_m['sd']:.2f}\n[{sd_m['ci'][0]:.2f}, {sd_m['ci'][1]:.2f}]", fontsize=6.6, color=INK, va="center")
    ax.plot([], [], color=BLUE, lw=0.9, label="fold 2, seeds 0-4")
    ax.plot([], [], color=ORANGE, lw=0.9, ls="--", label="fold 3, seeds 0-4")
    ax.plot([], [], "s", color=INK2, ms=4, ls="none", label="seed 0 (original run)")
    ax.legend(loc="upper right", handlelength=2.0)
    grid(ax, "y")
    save(fig, "fig_seeds")


# ----------------------------------------------------------------------------------------------------------------
def nih_feature():
    p = CACHE / "nih_feature.npy"
    if p.exists():
        return np.load(p)
    root = image_root()
    cells = pd.read_csv(R / "splits/cells.csv")
    with ProcessPoolExecutor(4) as ex:
        f = np.array(list(ex.map(_feat_path, [str(root / x) for x in cells.rel_path], chunksize=200)), dtype=float)
    CACHE.mkdir(exist_ok=True)
    np.save(p, f)
    return f


def _feat_path(p):
    return dark_spot.score(np.array(Image.open(p).convert("RGB")))


def bbbc(name):
    s, _ = load(name)
    lab = pd.read_csv(X / "E11_bbbc041_holdout/labels.csv")
    return candidates(s), lab


def fig_bbbc041():
    """ROC on both crop rules with the frozen operating points, then the feature's score by source."""
    fig = plt.figure(figsize=(FULL, 2.3))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.15], wspace=0.34)
    axs = [fig.add_subplot(gs[0]), fig.add_subplot(gs[1])]
    style = {"3_colour_ensemble": (BLUE, "-", "Colour-transfer ensemble (cand. 3)"),
             "2_E1_ensemble": (AQUA, "--", "DANet E1 ensemble (cand. 2)"),
             "1_feature": (ORANGE, "-.", "Dark-spot feature (cand. 1)"),
             "4_dinov2_s": (MUTED, ":", "DINOv2 probe (descriptive)")}
    for ax, name, title in ((axs[0], "bbbc041_ellipse", "(a) ROC, ellipse crops"), (axs[1], "bbbc041_otsu", "(b) ROC, Otsu crops")):
        c, lab = bbbc(name)
        prim = lab.y.notna().values
        y = lab.y.values[prim] == 1
        ax.plot([0, 1], [0, 1], color=GRID, lw=0.6)
        for k, (col, ls, lbl) in style.items():
            if k not in c:
                continue
            sc, t = c[k][0][prim], c[k][1]
            fpr, tpr, _ = roc_curve(y, sc)
            ax.plot(fpr, tpr, color=col, ls=ls, lw=1.1, label=lbl)
            q = sc >= t
            ax.plot((q & ~y).sum() / (~y).sum(), (q & y).sum() / y.sum(), "o", ms=5, mfc=col, mec=INK, mew=0.6, zorder=5)
        ax.set_xlim(-0.01, 1)
        ax.set_ylim(0, 1.01)
        ax.set_xlabel("False-positive rate")
        ax.set_ylabel("Sensitivity")
        ax.set_title(title)
        ax.set_aspect("equal")
        grid(ax, "both")
    h, l = axs[0].get_legend_handles_labels()
    h.append(plt.Line2D([], [], marker="o", ls="none", mfc="#fcfcfb", mec=INK, ms=5))
    l.append("Operating point at the frozen NIH threshold")
    fig.legend(h, l, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.13), handlelength=2.4, columnspacing=1.4)

    c, lab = bbbc("bbbc041_ellipse")
    f = c["1_feature"][0]
    cells = pd.read_csv(R / "splits/cells.csv")
    nf = nih_feature()
    bins = np.linspace(0, 1.2, 49)
    groups = [("NIH (fitting data)", nf[(cells.label == 0).values], nf[(cells.label == 1).values]),
              ("BBBC041, JPG group", f[(lab.group == "jpg").values & (lab.y == 0).values], f[(lab.group == "jpg").values & (lab.y == 1).values]),
              ("BBBC041, PNG group", f[(lab.group == "png").values & (lab.y == 0).values], f[(lab.group == "png").values & (lab.y == 1).values])]
    sub = gs[2].subgridspec(3, 1, hspace=0.75)
    for i, (g, u, p) in enumerate(groups):
        ax = fig.add_subplot(sub[i])
        hu, _ = np.histogram(u[np.isfinite(u)], bins, density=True)
        hp, _ = np.histogram(p[np.isfinite(p)], bins, density=True)
        ax.fill_between(bins[:-1], 0, hu, step="post", color="#cfcec9", lw=0, label="Uninfected")
        ax.step(bins[:-1], hp, where="post", color=ORANGE, lw=1.0, label="Parasitized")
        ax.axvline(dark_spot.THRESHOLD, color=INK, lw=0.8, ls=(0, (3, 2)))
        share = 100 * np.mean(u[np.isfinite(u)] >= dark_spot.THRESHOLD)
        ax.set_title(f"{g}: {share:.0f}% of uninfected above", fontsize=6.6, fontweight="normal", pad=2)
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
        ax.set_xlim(0, 1.2)
        if i == 0:
            ax.text(0, 1.62, "(c) Dark-spot score, threshold .3047 dashed", transform=ax.transAxes, fontsize=8, fontweight="bold")
            ax.legend(loc="upper right", fontsize=6.2, handlelength=1.2, borderaxespad=0)
        if i < 2:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel("Dark-spot score (BBBC041 on ellipse crops)")
    save(fig, "fig_bbbc041")


# ----------------------------------------------------------------------------------------------------------------
def fig_confusion():
    """Row-normalised confusion matrices: DANet on NIH unseen patients, and the three candidates on BBBC041."""
    nih = pd.concat([pd.read_csv(ckpt_path(ARMS["E1"], f).parent / "test_predictions.csv") for f in range(5)])
    c, lab = bbbc("bbbc041_ellipse")
    prim = lab.y.notna().values
    y = lab.y.values[prim] == 1
    panels = [("(a) DANet E1, NIH\n(grouped, 5 folds)", nih.label.values == 1, nih.p.values >= 0.5),
              ("(b) Dark-spot feature,\nBBBC041", y, c["1_feature"][0][prim] >= c["1_feature"][1]),
              ("(c) DANet E1 ensemble,\nBBBC041", y, c["2_E1_ensemble"][0][prim] >= 0),
              ("(d) Colour-transfer ens.,\nBBBC041", y, c["3_colour_ensemble"][0][prim] >= 0)]
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("seq", SEQ)
    fig, axs = plt.subplots(1, 4, figsize=(FULL, 1.75), gridspec_kw=dict(wspace=0.55))
    for ax, (title, yt, yp) in zip(axs, panels):
        m = np.array([[(~yt & ~yp).sum(), (~yt & yp).sum()], [(yt & ~yp).sum(), (yt & yp).sum()]])
        pct = 100 * m / m.sum(1, keepdims=True)
        ax.imshow(pct, cmap=cmap, vmin=0, vmax=100)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{pct[i, j]:.1f}%\n({m[i, j]:,})", ha="center", va="center", fontsize=6.6,
                        color="#fcfcfb" if pct[i, j] > 55 else INK)
        ax.set_xticks([0, 1], ["Uninf.", "Paras."])
        ax.set_yticks([0, 1], ["Uninf.", "Paras."])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(title, fontsize=7, loc="center")
        ax.tick_params(length=0)
        for sp in ax.spines.values():
            sp.set_visible(False)
    save(fig, "fig_confusion")


# ----------------------------------------------------------------------------------------------------------------
def crop(path, size=96):
    return np.array(Image.open(path).convert("RGB").resize((size, size), Image.BILINEAR))


def fig_examples():
    """Two random crops per class and source."""
    rng = np.random.default_rng(20261103)
    root = image_root()
    cells = pd.read_csv(R / "splits/cells.csv")
    mp = scored_table()
    lab = pd.read_csv(X / "E11_bbbc041_holdout/labels.csv")
    cols = []
    for name, paths in (("NIH", lambda k: [root / p for p in cells[cells.label == k].rel_path.sample(2, random_state=int(rng.integers(1e9)))]),
                        ("MP-IDB", lambda k: [CELLS_DIR / p for p in mp[mp.binary_label == k].file.sample(2, random_state=int(rng.integers(1e9)))]),
                        ("BBBC041 JPG", lambda k: [B41 / "ellipse" / f"{p}.png" for p in lab[(lab.y == k) & (lab.group == "jpg")].crop_id.sample(2, random_state=int(rng.integers(1e9)))]),
                        ("BBBC041 PNG", lambda k: [B41 / "ellipse" / f"{p}.png" for p in lab[(lab.y == k) & (lab.group == "png")].crop_id.sample(2, random_state=int(rng.integers(1e9)))])):
        cols.append((name, paths(1), paths(0)))
    fig, axs = plt.subplots(2, 8, figsize=(FULL, 1.95), gridspec_kw=dict(wspace=0.06, hspace=0.08))
    for j, (name, par, uni) in enumerate(cols):
        for k in range(2):
            for i, (imgs, _) in enumerate(((par, "P"), (uni, "U"))):
                ax = axs[i, 2 * j + k]
                ax.imshow(crop(imgs[k]))
                ax.set_xticks([])
                ax.set_yticks([])
                for sp in ax.spines.values():
                    sp.set_visible(False)
        axs[0, 2 * j].set_title(name, loc="left", fontsize=7.5)
    axs[0, 0].set_ylabel("Parasitized", fontsize=7)
    axs[1, 0].set_ylabel("Uninfected", fontsize=7)
    save(fig, "fig_examples")


def fig_failures():
    """Random BBBC041 errors (ellipse crops), two per category."""
    rng = np.random.default_rng(7)
    c, lab = bbbc("bbbc041_ellipse")
    f = c["1_feature"][0]
    p3 = 1 / (1 + np.exp(-c["3_colour_ensemble"][0]))
    cats = [("Ring missed\nby cand. 3", (lab.stage == "ring").values & (p3 < 0.5)),
            ("Uninfected called\nby cand. 3", (lab.y == 0).values & (p3 >= 0.5)),
            ("PNG uninfected called\nby feature only", (lab.y == 0).values & (lab.group == "png").values & (f >= dark_spot.THRESHOLD) & (p3 < 0.5)),
            ("Leukocyte called\nby feature only", (lab.stage == "leukocyte").values & (f >= dark_spot.THRESHOLD) & (p3 < 0.5))]
    fig, axs = plt.subplots(1, 8, figsize=(FULL, 1.35), gridspec_kw=dict(wspace=0.08))
    for j, (title, mask) in enumerate(cats):
        idx = rng.choice(np.where(mask)[0], 2, replace=False)
        for k, i in enumerate(idx):
            ax = axs[2 * j + k]
            ax.imshow(crop(B41 / "ellipse" / f"{lab.crop_id[i]}.png"))
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            ax.set_xlabel(f"f={f[i]:.2f}  p={p3[i]:.2f}", fontsize=6.3, color=INK2, labelpad=2)
        axs[2 * j].set_title(f"{title}  (n={mask.sum():,})", loc="left", fontsize=6.8)
    save(fig, "fig_failures")


# ----------------------------------------------------------------------------------------------------------------
def fig_explain():
    """E5b: share of true positives where each pointer lands on the annotated parasite, five folds per arm."""
    e = J(X / "E5b_five_checkpoint_explanations/results.json")["arms"]
    names = {"E1": "DANet (code)", "E8b": "Paper-faithful, LN", "E8c": "Paper-faithful, IN"}
    fig, ax = plt.subplots(figsize=(COL, 1.85))
    rules = (("darkest", "Darkest pixel (no model)", ORANGE, "s"), ("cam", "Grad-CAM peak", BLUE, "o"), ("centre", "Crop centre (no model)", MUTED, "^"))
    for i, a in enumerate(("E1", "E8b", "E8c")):
        for j, (k, lbl, col, mk) in enumerate(rules):
            v = [r[k] for r in e[a]["per_fold"]]
            x = i + (j - 1) * 0.24
            ax.plot([x] * 5, v, mk, ms=3.6, mfc=col, mec="#fcfcfb", mew=0.5, alpha=0.95, label=lbl if i == 0 else None, zorder=3)
            ax.plot([x - 0.09, x + 0.09], [np.mean(v)] * 2, color=INK, lw=1.2, zorder=4)
    ax.set_xticks(range(3), [names[a] for a in ("E1", "E8b", "E8c")])
    ax.set_ylabel("Hits on parasite (%)")
    ax.set_ylim(0, 104)
    ax.legend(loc="lower left", ncol=3, bbox_to_anchor=(0, 1.0), handletextpad=0.2, columnspacing=0.9)
    grid(ax, "y")
    save(fig, "fig_explain")


if __name__ == "__main__":
    for fn in (fig_forest, fig_seeds, fig_explain, fig_confusion, fig_examples, fig_failures, fig_bbbc041):
        fn()
