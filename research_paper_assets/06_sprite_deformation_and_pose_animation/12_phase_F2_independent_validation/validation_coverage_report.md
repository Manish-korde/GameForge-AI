# RUNG 2 — PHASE F2: VALIDATION COVERAGE REPORT
**Document ID:** `GF-AUDIT-RUNG2-F2-COVERAGE`  
**Execution Date:** 2026-11-01 23:28:51  
**Evaluation Scope:** 100% of Validation Cohort (40 Identities: `C_160` through `C_199`)  
**Evaluated Conditions:** 9 Poses (Idle + Walk F1–F8) $\times$ 16 Deterministic Seeds = **5,760 Sprites**  
**Sampler:** Pure Gaussian Reverse DDIM (50 Steps)  

---

## 1. INVESTIGATION OF THE EXPERIMENT E1 COVERAGE DISCREPANCY

During Experiment E1, the training configuration specified:
- **160 Training Identities** (`C_000` to `C_159`, 1,440 samples)
- **40 Validation Identities** (`C_160` to `C_199`, 360 samples)

However, the generation evaluation script (`run_phase_f_experiment_e1.py`, line 535) sub-selected only 16 test characters:
- 8 Training Identities: `C_000` through `C_007`
- 8 Validation Identities: `C_160` through `C_167`

### Root Cause of the Discrepancy:
1. **Computational Wall-Clock Constraint:** Generating 200 characters $\times$ 9 poses $\times$ 16 seeds (28,800 images) via 50-step DDIM on CPU required ~8 hours. To provide immediate feedback during training, evaluation was capped at 16 characters (2,304 generated images).
2. **Reporting Ambiguity:** The resulting metrics from the 8 validation characters (`C_160`–`C_167`) were labeled "Unseen Val Cohort" in the summary table, inadvertently giving the impression that all 40 validation characters had been sampled.
3. **Rectification in Phase F2:** In Phase F2, all **40 unseen validation characters** (`C_160` through `C_199`) have been evaluated across all 9 poses and 16 seeds (5,760 generated samples), completing 100% verification coverage.

---

## 2. FULL 40-CHARACTER VALIDATION COHORT SUMMARY

| Metric | E1 Reported (8 Val Chars) | F2 Measured (All 40 Val Chars) | Delta |
|---|---|---|---|
| **Identity PSNR** | 21.81 dB | **22.20 dB** | +0.39 dB |
| **Identity SSIM** | 0.869 | **0.8816** | +0.0126 |
| **Silhouette IoU** | 0.934 | **0.9490** | +0.0150 |
| **Walk Left-Leg Stride Error** | 0.17 px | **0.1692 px** | -0.0008 px |
| **Walk Right-Leg Stride Error** | 0.24 px | **0.2087 px** | -0.0313 px |
| **Pure Gaussian Stability** | 100% | **100% (5,760 / 5,760)** | 0.0% |

All 40 characters generate stable, identity-preserving sprites from pure Gaussian noise.
