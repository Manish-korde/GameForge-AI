# PHASE F2 METRIC RECONCILIATION AUDIT
**Document ID:** `GF-AUDIT-F2-RECONCILIATION`  
**Execution Date:** 2026-11-02  
**Investigation Topic:** Discrepancy between Main Report Silhouette IoU (0.9426) and Validation Coverage Report Silhouette IoU (0.9490).

---

## 1. RAW DATA AUDIT & MATHEMATICAL DISCOVERY

A programmatic audit of all 5,760 raw records in `phase_F2_validation/complete_validation_metrics.csv` reveals the exact mathematical origin of the discrepancy.

### Empirical Breakdown:
- **Total Evaluated Samples:** 5,760 sprites (40 unseen validation characters $\times$ 9 conditions $\times$ 16 seeds).
- **Idle Stance Samples ($a=0, p=0$):** 640 sprites.
  - Idle Mean Silhouette IoU: **1.000000**
- **Walk Pose Samples ($a=1, p \in \{0..7\}$):** 5,120 sprites.
  - Walk Frames (F1–F8) Mean Silhouette IoU: **0.942633** ($\approx \mathbf{0.9426}$)
- **Combined Cohort Mean (All 9 Poses):**
  $$\text{Mean IoU}_{\text{combined}} = \frac{1 \times 1.000000 + 8 \times 0.942633}{9} = \mathbf{0.949007} \approx \mathbf{0.9490}$$

---

## 2. ROOT CAUSE SUMMARY

The discrepancy was caused by **sample population stratification**, NOT data corruption, calculation errors, or multiple evaluation runs:

1. **`validation_coverage_report.md` (Table 2):**
   Reported the unweighted average across **all 5,760 samples**, pooling the 640 Idle samples ($\text{IoU} = 1.000$) with the 5,120 Walk samples ($\text{IoU} = 0.9426$), yielding **0.9490**.
2. **`phase_F2_report.md` (Table 1):**
   Separated Idle from Walk, explicitly reporting:
   - *Idle Stance IoU:* **1.0000**
   - *Walk Poses IoU:* **0.9426** (mean of the 8 dynamic walking frames).

Both reports derived from the exact same CSV records without discrepancy in the underlying inference.

---

## 3. PRESERVED SCIENTIFIC LIMITATIONS (NO GLOSSING)

In accordance with strict Phase G academic standards, all documented limitations from Phase F2 remain fully acknowledged:

| Metric | Target Threshold | Measured Value (40 Val Cohort) | Compliance Status | Documented Academic Reason |
|---|---|---|---|---|
| **Walk Left-Leg Stride Error** | $\le 0.15\text{ px}$ | **0.1692 px** | **MARGINAL / UNMET** | Minor centroid variance across diverse clothing palettes. |
| **Walk Right-Leg Stride Error**| $\le 0.15\text{ px}$ | **0.2087 px** | **FAILED** | Driven by stride compression at right apex (F7). |
| **F7 Stride Compression** | $0.0\%$ | **38.9% Compression** | **LIMITATION** | Model synthesizes -0.349 px vs -0.561 px ground truth. |
| **Pose Differentiation Rate** | High ($>80\%$) | **83.75% Distinguishable** | **MET** | 938 / 1,120 pairwise comparisons distinctly separated. |
| **Stagnant Pose Pairs** | Zero ideal | **16.25% Stagnant** | **LIMITATION** | Confined to passing stance transitions (e.g. F1 vs F5). |

These limitations are permanently declared and preserved in the GameForge AI Phase G integration documentation.

