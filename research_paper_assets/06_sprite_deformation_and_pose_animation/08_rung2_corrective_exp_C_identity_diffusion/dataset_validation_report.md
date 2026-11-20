# PHASE 0 & 1: DATASET VALIDATION & MANIFEST REPORT
## Action-Conditioned Diffusion Dataset (Experiment C)

**Date:** November 1, 2026  
**Audited Source:** `GameForge_AI_Phase2_audit/gameforge_phase2_1_audit/data/candidate_crops/action`  
**Dataset Size:** 20 Characters × 2 Actions = 40 Total Samples  

---

## 1. Split Architecture (Zero Identity Leakage)

To rigorously assess held-out generalization, splitting is strictly enforced along the **character identity boundary**:

- **Training Cohort:** Characters `C_00` to `C_15` (16 unique identities, 32 samples total).
  - 16 Idle targets (`action_id = 0`)
  - 16 Walk targets (`action_id = 1`)
- **Validation / Held-Out Cohort:** Characters `C_16` to `C_19` (4 unique identities, 8 samples total).
  - 4 Idle targets (`action_id = 0`)
  - 4 Walk targets (`action_id = 1`)

> **Integrity Guarantee:** Zero character identity overlap between training and validation sets. Both actions (Idle and Walk) for any given character remain strictly inside the same partition.

---

## 2. Extraction & Tensor Validation

| Parameter | Specification | Verified Status |
| :--- | :---: | :--- |
| **Canvas Dimensions** | $64 	imes 64$ pixels | Confirmed across all 40 crops ($100\%$). |
| **Color Space & Channels**| 4-channel RGBA | Confirmed. |
| **Coordinate Normalization** | $[-1.0, 1.0]$ via $x / 127.5 - 1.0$ | Verified for both RGB and Alpha channels. |
| **Idle Target Specification** | Row 2, Col 0 ($[0, 128, 64, 192]$) | Verified identical to Neutral Stance Reference. |
| **Walk Target Specification** | Row 2, Col 3 ($[192, 128, 256, 192]$) | Verified front-facing (South) Apex Stride. |
| **Duplicate / Corrupt Crops** | 0 detected | All 20 characters feature unique palettes and clothing. |

---

## 3. Sample Inventory Excerpt

| Sample ID | Character | Action ID | Action Name | Partition | Foreground Body Pixels | Target Grid Coordinates |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `C_00_idle` | `C_00` | 0 | Idle | Train | 1,056 px | Row 2, Col 0 |
| `C_00_walk` | `C_00` | 1 | Walk | Train | 1,007 px | Row 2, Col 3 |
| `C_01_idle` | `C_01` | 0 | Idle | Train | 1,186 px | Row 2, Col 0 |
| `C_01_walk` | `C_01` | 1 | Walk | Train | 1,170 px | Row 2, Col 3 |
| ... | ... | ... | ... | ... | ... | ... |
| `C_16_idle` | `C_16` | 0 | Idle | **Val (Held-out)** | 1,148 px | Row 2, Col 0 |
| `C_16_walk` | `C_16` | 1 | Walk | **Val (Held-out)** | 1,133 px | Row 2, Col 3 |
| `C_19_walk` | `C_19` | 1 | Walk | **Val (Held-out)** | 966 px | Row 2, Col 3 |

Dataset cached at: `corrective_experiment_C/dataset_c_tensors.pt`.
