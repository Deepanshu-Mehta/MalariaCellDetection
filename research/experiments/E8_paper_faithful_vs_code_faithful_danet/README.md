# E8: Paper-faithful vs code-faithful DANet

**Status:** draft - team to confirm  ·  **Outcome:** — (positive / negative / invalid)

> **Correction, 2026-11-02 (E8r):** the LN-IN leakage result below did not replicate. On new seeds 45-47, mean paired LN-IN leaky BACC was +.15 [-.05, .36], against +.54 here, and the registered rule returned "not replicated". The claim stays exploratory, and
> the verdict below should be read with that in mind. See `../E8r_norm_layer_leaky_replication/README.md`.

## Question
Do the paper's sigmoid gate / parallel dilations / LayerNorm change shortcut reliance?

## Prediction (written before any run; date it)
**No explicit prediction was written before launch (2026-10-27).** The closest registered reasoning is M3's
"why we might be wrong" (2026-10-05): InstanceNorm in every block strips per-image colour, so DANet may already be
largely stain-invariant. Report E8 as exploratory, not confirmatory.

## Design change (2026-10-27, before any E8 run)
Split into two arms so the norm layer is isolated:
- **E8a** `configs/E8a_layernorm.json`: code-faithful DANet, InstanceNorm -> LayerNorm only.
- **E8b** `configs/E8_paper_faithful.json`: paper text (sigmoid gate, parallel dilations, LayerNorm).
Each: grouped repeat 0 folds 0-4 (paired with E1 r0 on identical cells) + leaky seeds 42-44, 50 epochs.

## Protocol
- Splits / runs: grouped 5 folds x 1 seed (screen)
- Config: `research/configs/E8_paper_faithful.json`
- Primary metric: leaky-grouped gap, stain-swap flip rate, BACC
- Estimated T4 GPU-h: 8.5

## Runs
Kaggle job slugs and dates (results are aggregated by `research/scripts/collect_results.py` into
`research/results/all_runs.csv`):

| Date | Job slug | Runs | Notes |
|---|---|---|---|
| 2026-10-27 | e8-norm-layer | E8a and E8b: grouped r0 f0-4 + leaky s42-44 each | 16/16 finished (~1.0 h each). **E8a invalid**: 7/8 runs stuck at chance |
| 2026-10-27 | e8c-paper-instancenorm | E8c: grouped r0 f0-4 + leaky s42-44 | added after E8a failed, to isolate the norm layer against E8b |

## Result (`analysis.py` -> `results.json`)

**E8a (LayerNorm only) is invalid.** 7 of 8 runs never left chance: validation loss stayed at 0.693 (= ln 2) and
test BACC was exactly 50 %; only grouped fold 3 trained (BACC 96.9). This is the same optimisation failure as the
M2 diagnostic (RMSprop leaves DANet at chance in some settings): LayerNorm combined with the code's channel-softmax
gate does not train reliably under the paper's optimiser. It says nothing about leakage. (Reproducibility note:
mixing the paper's LN with the code's softmax gate does not train.)

| Arm | Grouped r0 BACC | Leaky BACC (3 seeds) | Leakage gap [approx. 95 % CI] |
|---|---|---|---|
| E1, code-faithful (IN, softmax, cascade) | 96.78 | 96.96 ± 0.20 | +0.19 [-0.45, 0.87] |
| E8b, paper-faithful (LN, sigmoid, parallel) | 96.71 | 97.34 ± 0.06 | +0.63 [0.01, 1.28] |

- E8b - E1 on identical grouped cells: BACC -0.07 [-0.26, +0.12], SE -0.22 [-0.62, +0.12], SP +0.09 [-0.16, +0.36].
- Gap difference E8b - E1: +0.44 [+0.18, +0.70] (approximate: leaky and grouped bootstraps are independent).
- Reading so far: the paper's architecture scores higher on the leaky split (closer to the paper's 97.95) but no
  better on unseen patients. Three things change at once, so the norm layer is tested separately by E8c.

### E8c: the norm-layer contrast (E8b vs E8c differ only in LayerNorm vs InstanceNorm)

| Arm | Grouped r0 BACC | Leaky BACC (3 seeds) | Leakage gap [approx. 95 % CI] |
|---|---|---|---|
| E8b, paper-faithful, **LayerNorm** | 96.71 | 97.34 ± 0.06 | +0.63 [0.01, 1.28] |
| E8c, paper-faithful, **InstanceNorm** | 96.78 | 96.80 ± 0.18 | +0.02 [-0.57, 0.65] |

- LN - IN on identical grouped cells: BACC -0.07 [-0.30, +0.15], SE -0.04 [-0.46, +0.37], SP -0.10 [-0.35, +0.15].
- LN - IN gap difference: **+0.61 [+0.29, +0.90]** (approximate CI: independent bootstraps of leaky and grouped).
- E8c - E1 (sigmoid + parallel vs softmax + cascade, both IN): grouped +0.01 [-0.17, +0.19]; gap diff -0.17 [-0.43, +0.11].

## Verdict and what we learned
**Exploratory positive.** Holding the rest of the paper-faithful block fixed, swapping InstanceNorm for LayerNorm
lets DANet gain ~0.6 pp BACC from patient leakage while its accuracy on unseen patients is unchanged. The gate and
dilation choices do not matter (E8c ~= E1). This supports the M3 hunch that InstanceNorm acts as built-in stain
normalisation, and explains part of the gap between the paper's leaky headline and the code-faithful reproduction.

Limits: no prediction was written before E8; E8c was added after E8a failed; one grouped repeat and 3 leaky
seeds per arm; the gap CI is approximate. **To confirm:** (1) a pre-registered replication on grouped repeats
1-2 and new leaky seeds (45-47) for E8b and E8c; (2) the E2 stain-swap test on E8b vs E8c checkpoints (InstanceNorm
should flip far fewer predictions when a cell's colour is swapped to another patient's stain).

## Notes
LayerNorm = GroupNorm(1 C) - state in paper
