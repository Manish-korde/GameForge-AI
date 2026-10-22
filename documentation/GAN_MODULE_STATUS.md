# GameForge DCGAN Module — Status & Architecture Document

**Module Name**: GameForge DCGAN — Unconditional Sprite Generator
**Status**: ✅ COMPLETE — 50 Epochs trained, 99,300 global steps, artifacts saved
**Last Updated**: October 2026

---

## Module Role in Pipeline

The DCGAN module occupies a distinct, non-overlapping role from the AE and VAE components:

| Model | Role | Dataset |
|---|---|---|
| Transformer | Semantic planning (NL → JSON spec) | ViGGO-derived |
| AE | Reconstruction-based asset screening (P95 outlier QA) | Alucard 282K sprites |
| VAE | Latent-space asset search & exploration | Alucard 282K sprites |
| **DCGAN** | **Unconditional novel 2D sprite generation (adversarial)** | **Alucard 282K sprites** |

The DCGAN generates genuinely new pixel-art sprites that were never seen in training. It does **not** reconstruct (AE's job) or search/morph (VAE's job). Its output is brand-new sprite content sampled from learned adversarial distribution.

---

## Why Unconditional DCGAN (not cGAN / Vanilla GAN / Texture GAN)

| Approach | Decision | Reason |
|---|---|---|
| **Unconditional DCGAN** | ✅ **SELECTED & TRAINED** | Convolutional inductive bias suits spatial sprite structure. Trains on pixel content directly—no label dependency. |
| Vanilla GAN (MLP-based) | ❌ Rejected | MLP GAN cannot learn spatial structure; produces blurry, unstructured output. Only used as a deprecated baseline comparison. |
| Conditional GAN (cGAN) | ❌ Not approved | Requires validated, reliable class labels. Alucard caption labels were not human-validated; noisy labels cause class collapse. |
| Texture GAN (VastTexture) | ❌ Removed | Separate concept (surface material textures) abandoned — no VastTexture data, separate domain, never implemented. |
| StyleGAN2 / CycleGAN / Pix2Pix | ❌ Out of scope | Require paired data, large GPU budgets, or domain-specific architecture beyond this project scope. |

---

## Dataset

- **Source**: `evilsocket/alucard-sprites` (HuggingFace)
- **Unique Images Used**: 282,511 deduplicated RGBA sprites
- **Training Resolution**: 64×64 RGBA (4 channels, downscaled for GAN stability)
- **Pixel Normalization**: [-1, 1] (tanh output compatible)
- **License**: CC-BY-NC-SA 4.0

### Dataset Split (same unique_indices.json as AE/VAE)

| Split | Count |
|---|---|
| Train | ~226,009 (≈80%) |
| Validation | ~28,251 (≈10%) |
| Test | ~28,251 (≈10%) |

---

## Architecture (DCGAN)

**Generator**:
- Input: `z ∈ ℝ^100 ~ N(0, I)` (100-dimensional noise vector)
- Dense projection → `4×4×512` feature map → BatchNorm → ReLU
- 4× `Conv2DTranspose` (stride=2, kernel=4×4): `4×4 → 8×8 → 16×16 → 32×32 → 64×64`
- Filters: 512 → 256 → 128 → 64 → 4 (RGBA)
- Final activation: **Tanh** (output range [-1, 1])
- **Total parameters**: ~3,579,648 (~13.66 MB)

**Discriminator**:
- Input: `x ∈ ℝ^{64×64×4}` (real or fake RGBA sprite)
- 4× `Conv2D` (stride=2, LeakyReLU α=0.2): `64×64 → 32×32 → 16×16 → 8×8 → 4×4`
- Filters: 64 → 128 → 256 → 512
- Flatten → Dense(1) → single logit
- **Total parameters**: ~2,768,385 (~10.56 MB)

**Training Setup**:
```
Loss:       Binary Cross-Entropy with Logits
Optimizer:  Adam (G_lr=0.0002, D_lr=0.0002, β1=0.5, β2=0.999)
Batch size: 128 (global, 2 replicas on Kaggle P100)
Epochs:     50
Steps:      99,300 global training steps
Hardware:   Kaggle P100 GPU
Duration:   ~5.67 hours (~409 seconds/epoch average)
```

---

## Training Results (50 Epochs)

| Metric | Value |
|---|---|
| Total global optimizer steps | 99,300 |
| Best Generator loss epoch | Tracked via checkpoint (best_g_loss saved) |
| Discriminator loss range | 0.015 – 0.040 (stable, not collapsed) |
| Generator loss range | ~8.0 – 8.9 (standard adversarial range) |
| Mode collapse detected | No (diversity_std monitored per evaluation) |
| Total training time | ~5.67 hours |

---

## Model Artifacts

```
models/DCGAN_280K_COMPLETE/
├── weights/
│   ├── generator_latest.keras          ← primary inference model
│   ├── generator_latest.weights.h5     ← weight-only backup
│   ├── discriminator_latest.keras      ← not needed for inference
│   └── discriminator_latest.weights.h5
├── checkpoints/                        ← TF resumable checkpoints (ckpt-47..49)
├── logs/
│   ├── training_history.csv
│   ├── training_history.json
│   └── loss_curves.png
├── samples/
│   ├── samples_epoch_0000.png ... samples_epoch_0050.png
│   └── samples_final.png
├── config/
│   ├── run_config.json
│   ├── dataset_info.json
│   ├── training_state.json
│   ├── generator_summary.txt
│   └── discriminator_summary.txt
└── README.md
```

---

## Inference Usage

```python
import tensorflow as tf
import numpy as np

generator = tf.keras.models.load_model(
    'models/DCGAN_280K_COMPLETE/weights/generator_latest.keras',
    compile=False
)

# Sample 16 new sprites
z = np.random.normal(0, 1, size=(16, 100)).astype(np.float32)
fake_sprites = generator.predict(z, verbose=0)
sprites = np.clip((fake_sprites + 1.0) / 2.0, 0.0, 1.0)  # → [0, 1] RGBA
```

---

## Evaluation Notebook

Comprehensive evaluation with quantitative metrics, loss curves, epoch progression grids, alpha channel decomposition, and live inference is documented in:

```
Notebook/DCGAN_280K_Demonstration_and_Evaluation.ipynb
```

---

## Comparison with AE and VAE

| Feature | Autoencoder (AE) | VAE | **DCGAN** |
|---|---|---|---|
| Nature | Deterministic | Probabilistic | **Adversarial** |
| Role | Reconstruction / QA | Latent search / morphing | **Novel sprite generation** |
| Requires input image? | Yes (reconstruction) | Yes (encode then sample) | **No (pure generation from noise)** |
| Label dependency | None | None | **None (unconditional)** |
| Output type | Reconstructed sprite | Latent interpolation / sample | **Brand-new sprite** |
| Trained on | 282K RGBA sprites | 282K RGBA sprites | **282K RGBA sprites** |

---

## What DCGAN is NOT

| Myth | Reality |
|---|---|
| "DCGAN generates textures from prompts" | No — it generates novel RGBA game sprites from random noise `z ~ N(0,I)`. Text-to-image is not its role. |
| "DCGAN replaces AE or VAE" | No — AE does QA, VAE does search/morphing, DCGAN generates new content. Complementary roles. |
| "Vanilla GAN is also implemented" | No — Vanilla GAN notebook was removed. DCGAN is the sole final GAN implementation. |
| "This uses the VastTextures dataset" | No — VastTextures / Texture GAN was a discarded concept. DCGAN trains on Alucard sprites only. |
| "Conditional GAN (cGAN) is used" | No — cGAN requires validated labels. Alucard labels are noisy. Unconditional DCGAN avoids label dependency entirely. |
