# Alucard 280K DCGAN Label Review & Decision Record

## Summary of Final Decision

**DCGAN (Unconditional) is the finalized and trained GAN for this project.**

- Training: **50 epochs, 99,300 global steps** on 282,511 Alucard RGBA sprites.
- Artifacts: `models/DCGAN_280K_COMPLETE/`
- Evaluation: `Notebook/DCGAN_280K_Demonstration_and_Evaluation.ipynb`
- **Vanilla GAN baseline notebook has been removed** — DCGAN is the sole GAN implementation.

---

## Evidence Boundary & Dataset Scope

This document covers only the deduplicated `evilsocket/alucard-sprites` corpus (282,511 128×128 RGBA sprites).

VastTextures was a discarded concept — that entire direction (surface material texture GAN on a separate dataset) was abandoned. This project uses only the Alucard sprite corpus for all visual models.

---

## Why Unconditional DCGAN Was Chosen

| GAN variant | Decision | Reason |
|---|---|---|
| **Unconditional DCGAN** | ✅ **Selected and trained** | Label-free; learns full sprite visual distribution from pixels. Adversarial convolutional architecture is suited to spatial pixel-art structure. |
| Vanilla GAN (MLP) | ❌ Removed | No spatial inductive bias; produces blurry, unstructured output. Used only as a deprecated historical baseline, now removed. |
| Conditional GAN (cGAN) | ❌ Not approved | Requires validated class labels — Alucard captions were never human-reviewed to the required standard (≥80% Wilson lower bound). Noisy labels cause class collapse. |
| Class-controlled GAN | ❌ Not approved | Same label validation dependency as cGAN. Severely imbalanced categories (65% Characters) complicate balanced sampling. |
| Attribute-conditioned GAN | ❌ Not approved | Color, size, view-angle attributes from captions are unvalidated. Cannot train reliably. |
| Text-conditioned GAN | ❌ Not approved | Full caption reliability unverified. Too high a risk of noisy training signal. |
| StyleGAN2-ADA | ❌ Out of scope | Excess complexity, requires strong GPU budget and extensive tuning beyond project scope. |
| Pix2Pix / CycleGAN | ❌ Wrong task | Requires paired data or meaningful unpaired domains — not applicable to unconditioned single sprite population. |
| Texture GAN (VastTextures) | ❌ Abandoned | Separate domain (surface materials ≠ game sprites), no VastTexture data downloaded, direction discarded entirely. |

---

## Conditional GAN Future Pathway (Not Current Scope)

Conditional or class-controlled GAN remains possible **only after** the human label review process:

```powershell
python scripts\review_alucard_captions.py sample --output-dir evaluation\alucard_label_review --per-label 100 --seed 42
```

This requires:
1. Completing the human caption review CSV (`evaluation/alucard_label_review/review_samples.csv`)
2. Running `finalize` to generate `label_reliability_report.md`
3. Every reviewed stratum with ≥30 samples must have a 95% Wilson lower bound of ≥80%
4. Ambiguity fraction must be ≤10%

Only if this passes is conditional GAN an option. It is **not the current project state**.

---

## Current Project State (DCGAN)

The unconditional DCGAN is complete:

- **Architecture**: `z ∈ ℝ^100 ~ N(0,I)` → Dense(4×4×512) → 4× Conv2DTranspose → **64×64×4 RGBA** (Tanh)
- **Discriminator**: 4× Conv2D(stride=2, LeakyReLU α=0.2) → Dense(1)
- **Loss**: Binary Cross-Entropy with Logits
- **Optimizer**: Adam (lr=0.0002, β1=0.5, β2=0.999) for both G and D
- **Batch size**: 128 global (2 Kaggle replicas)
- **Epochs**: 50 | **Steps**: 99,300 | **Duration**: ~5.67 hours

Adversarial stability confirmed: D-loss range 0.015–0.040, no mode collapse detected.
