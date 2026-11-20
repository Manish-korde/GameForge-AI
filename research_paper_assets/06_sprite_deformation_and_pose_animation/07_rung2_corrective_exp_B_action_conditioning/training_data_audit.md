# TEST 1: TRAINING DATA & PAIR AUDIT REPORT
## Rung 2 Diffusion Pipeline — Training Dataset Forensic Inventory

**Audit Date:** November 1, 2026  
**Audited Artifact:** `scripts/run_rung2_ddpm_overfit.py` & `research_paper_assets/06_sprite_deformation_and_pose_animation/rung0_forensic_metrics_report.csv`  
**Dataset Source:** `carlosuperb/lpc-action-pixel-art-diffusion`  

---

## 1. Quantitative Inventory

| Audit Criterion | Verified Value | Forensic Significance |
| :--- | :---: | :--- |
| **Number of Unique Reference Characters** | **20** (`C_00` to `C_19`) | Controlled 20-character prototype cohort. |
| **Total Reference-Target Pairs** | **20 Pairs** | Strictly 1 pair per character. |
| **Unique Target Poses in Training** | **1 Pose Type** | **South Walk Apex Stride (F3, Col 3, Row 2)** across all 20 samples. |
| **Animation Actions Represented** | **1 Action** | **South Walk Only**. Zero Idle, Slash, Run, Hurt, or Spellcast targets. |
| **Multi-Pose Reference Sharing** | **NO (0 instances)** | No character in the dataset has more than one target pose associated with it. |
| **Train vs Validation Split** | **100% Train / 0% Val** | Pure memorization / capacity overfit experiment. |
| **Pair Correspondence & Orientation** | **Strictly Validated** | Both ref and target are front-facing (South), matching character clothing, hair, and palette. |
| **Channel Layout & Normalization** | **Consistent RGBA [-1, 1]** | Both reference and target are 4-channel tensors scaled via `arr / 127.5 - 1.0`. |

---

## 2. Key Forensic Findings

### 2.1 The 1-to-1 Pose Degeneracy
In `scripts/run_rung2_ddpm_overfit.py` (lines 41-42):
```python
ref_crop = sheet.crop((0 * 64, 2 * 64, 1 * 64, 3 * 64))   # Row 2, Col 0: Neutral Rest Stance
walk_crop = sheet.crop((3 * 64, 2 * 64, 4 * 64, 3 * 64)) # Row 2, Col 3: South Walk Apex F3
```
Because every character in the training batch is paired with exactly **one** target action (Walk F3), the training distribution $p_{	ext{data}}(x_0 \mid x_{	ext{ref}})$ has support on only a single action mode. 

### 2.2 Absence of Action Supervision
The network was never shown:
1. $x_{	ext{ref}} 	o x_0^{	ext{idle}}$ (reconstructing the neutral stance)
2. $x_{	ext{ref}} 	o x_0^{	ext{slash}}$ (reconstructing weapon swing)
3. $x_{	ext{ref}} 	o x_0^{	ext{walk\_f1}}$ (subtle stride)

Consequently, **the dataset provides zero supervision to learn action differentiation.** The model cannot learn "how to walk" vs "how to swing"; it only experiences a static mapping from Rest Pose to Stride Pose.

---

## 3. Visual Contact Sheet
The complete 20-pair inventory is visualized in:
`corrective_experiment_B/reference_target_contact_sheet.png`
