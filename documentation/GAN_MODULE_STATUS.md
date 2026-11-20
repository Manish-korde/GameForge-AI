# GameForge DCGAN Module — Status & Architecture Document

**Module Name**: GameForge DCGAN — Discrete-Palette Character Sprite Generator  
**Status**: ✅ TRAINED & DEPLOYED (50 Epochs, Notebook 02) | ✅ VISUAL QUALITY: CRISP RETRO PIXEL ART  
**Last Updated**: November 15, 2026 (Production Locked & Backend Integrated)  

---

## 1. Module Role in GameForge Multi-Model Pipeline

The DCGAN module occupies a distinct, non-overlapping adversarial role in the GameForge AI platform:

| Model | Architecture | Role in Pipeline | Output Artifact |
|---|---|---|---|
| **Transformer** | Flan-T5-small | Semantic Planning (Natural Language → Spec) | Structured JSON Game Spec |
| **Autoencoder (AE)** | Deep Conv AE (128×128) | Deterministic Reconstruction & P95 Outlier QA | High-PSNR (35.36 dB) Reconstruction |
| **VAE** | $\beta$-VAE (256-D Latent) | Probabilistic Latent Space Search & Interpolation | Vector Search (1ms), Latent Morphs |
| **Discrete DCGAN** | Deep Conv GAN + Discrete Palette Head (64×64) | **Unconditional Novel 2D Sprite Synthesis** | **Game-Ready 16-bit Pixel Art Sprites** |

Unlike the Autoencoder (which reconstructs existing assets) or the VAE (which interpolates between known sprites), the DCGAN synthesizes brand-new character sprites sampled from the learned adversarial distribution.

---

## 2. Core Architecture: Discrete Palette Classification Head

### Why Previous Continuous DCGAN Failed
Standard DCGAN architectures use continuous $\tanh$ activations producing floating-point RGB values in $[-1, 1]$. On 2D retro sprites, this creates:
1. Continuous color blending (>1,400 colors per sprite instead of indexed palettes).
2. Blurry, undefined outlines and low spatial coherence (only 7.0% vs. real 53.0%).
3. Confetti noise specks on torsos and faces.

### The Discrete Solution (Notebook 02)
We restructured the DCGAN Generator from an image regressor into a **spatial categorical classifier**:
- **Generator Output:** Conv2D with 33 channels ($K+1$ logits):
  - 32 channels representing classification logits over an empirical 32-color palette.
  - 1 channel representing alpha foreground/background logit.
- **Differentiable Palette Quantization:**
  - Training: Gumbel-Softmax with temperature annealing ($\tau = 1.5 \to 0.8$).
  - Inference: Categorical argmax (`hard=True`, `noise=False`), guaranteeing crisp indexed colors.
- **Empirical Palette ($K=32$):** Pre-extracted RGB lookup table (`global_palette_32.npy`) derived from 42,000 clean front-facing Alucard sprites.

```
Latent z (128-D) 
       │
       ▼
Dense Projection (4×4×512)
       │
       ▼
Conv2DTranspose / UpSampling (4×4 → 8×8 → 16×16 → 32×32 → 64×64)
       │
       ▼
64×64 Full-Resolution Refinement (64 filters, LeakyReLU 0.2)
       │
       ▼
Conv2D (33 filters) ──► Logits [32 Color Logits | 1 Alpha Logit]
       │
       ├── Softmax + Argmax ──► One-Hot (64×64×32) ──► Dot(Palette[32×3]) ──► RGB (64×64×3)
       └── Sigmoid + Threshold(0.5) ─────────────────────────────────────────► Alpha (64×64×1)
```

---

## 3. Training & Benchmark Results

### Defense Ground Truth Metrics
* **Spatial Coherence:**
  - Real Dataset: **53.0%**
  - Discrete DCGAN Epoch 10: **58.2%**
  - Discrete DCGAN Epoch 50: **48.8%**
  - Continuous Baseline DCGAN: **7.0%** (+41.8% to +51.2% absolute gain)
* **Color Palette Count:**
  - Continuous Baseline: **~1,400 colors**
  - Discrete DCGAN: **~25–27 colors** (Target: ~18 colors)
* **Alpha Purity:** 100% strictly binary `[0, 255]`
* **Hardware & Runtime:** Kaggle Dual T4 GPUs, 50 epochs, ~270s/epoch.

---

## 4. Production Inference Pipeline

The raw generator output is passed through three post-processing stages in `backend/dcgan_service.py`:

1. **Truncation ($\sigma=0.85$):** Scales latent noise vectors to $z \times 0.85$ (clipped to $[-2.0, 2.0]$). Avoids outer Gaussian floater tails while maintaining diverse palettes, skin tones, and outfits (avoiding repeated archer collapse seen at $\sigma=0.70$).
2. **Weapon-Safe LCC Floater Filter:** Discards isolated 1–3 pixel noise specks while preserving character weapons, staves, and bows ($\ge 6\text{ px}$ or $\ge 3\text{ px}$ within $14\text{ px}$ of the torso).
3. **Best-of-8 Candidate Rejection Sampling:** Generates 8 candidates per character seed, evaluates a composite score (orphans, target fill fraction, color count), and enforces a strict gate ($0.15 \le \text{FG} \le 0.55$) so blank blobs or shrouded boxes can never win.
4. **8× Nearest-Neighbor Upscaling:** Upscaled to $512\times 512$ with nearest-neighbor interpolation, ensuring crisp pixel art in the web GUI.

---

## 5. File & Directory Inventory

### Master Training Directory in Downloads
* **Path:** `C:\Users\manis\Downloads\dcgan_frontfacing_results`
  - `game_ready_assets/generator_weights_final.weights.h5` (10.9 MB — Final Epoch 50 Weights)
  - `game_ready_assets/discriminator_weights_final.weights.h5` (22.1 MB)
  - `config/global_palette_32.npy` (Empirical 32-color palette)
  - `config/fixed_latent_vectors.npy` (36×128 fixed latent vectors)
  - `config/alucard_frontfacing_42k_indices.json` (Training set indices)
  - `logs/training_history.csv` (Complete 50-epoch training logs)
  - `samples/samples_epoch_0001.png` – `samples_epoch_0050.png`
  - `checkpoints/` (`ckpt-23`, `ckpt-24`, `ckpt-25` for Epochs 46, 48, 50)

### Production Workspace Model Folder
* **Path:** `c:\Users\manis\OneDrive\Desktop\Prompt_to_game_asset_generator\models\3_DCGAN_42k_v1`

### Backend Service File
* **Path:** `c:\Users\manis\OneDrive\Desktop\Prompt_to_game_asset_generator\backend\dcgan_service.py`

### Evaluation Dossiers
* **Master 6-Panel Comparison Board:** `phase_dcgan_improvement/controlled_variant_comparison_master.png`
* **Zoomed 8× Weapon Audit:** `phase_dcgan_improvement/zoomed_8x_forensic_comparison.png`
* **GUI Live Verified Output:** `phase_dcgan_improvement/gui_service_verified_output.png`
