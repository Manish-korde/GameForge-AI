# GameForge AI — Master Project Context & Knowledge Base 🎮🤖

This document serves as the definitive architectural knowledge base for **GameForge AI**. It documents core concepts, model roles, dataset configurations, backend optimizations, and review presentation strategies.

---

## 🧠 1. Project Overview & Multi-Model Pipeline Strategy

**GameForge AI** is a multi-model Generative AI platform for 2D game asset creation, semantic planning, adversarial sprite synthesis, and latent space analysis.

```text
                  GAMEFORGE AI PIPELINE
                           │
                  Natural-Language Request
                           │
                           ▼
                      TRANSFORMER
                    Semantic Planning
                           │
                           ▼
            Structured Game Specification (JSON)
                           │
           ┌───────────────┼────────────────┐
           │               │                │
           ▼               ▼                ▼
       DCGAN         Visual Generation   Asset Library
    Novel Sprite           │
    Generation             ▼
                      Generated Asset
                           │
                     ┌─────┴─────┐
                     ▼           ▼
                    AE           VAE
                     │           │
               Reconstruction   Latent
               / Data Cleaning  Representation
                     │           │
                     └─────┬─────┘
                           ▼
                     Asset Analysis
```

### The Four Model Components:
1. **Transformer Model**: Semantic planning layer. Translates natural language game descriptions into structured game specification sheets (JSON) detailing themes, character types, environment features, weapon properties, and NPC specs.
2. **DCGAN**: Unconditional Deep Convolutional GAN trained on 282,511 Alucard RGBA sprites. Generates brand-new 64×64 RGBA pixel-art sprites from random noise `z ~ N(0, I)`. 50 epochs, 99,300 global steps completed. Label-free — avoids noisy Alucard annotation issues.
3. **Autoencoder (AE)**: Deterministic model trained on **280,000 sprites**. Used for high-precision deterministic reconstruction (`MSE: 0.000782`, `PSNR: 28.52 dB`, `SSIM: 0.9248`) and outlier sprite screening (flagging the highest-error 5% of assets exceeding P95 MSE threshold `0.001313`).
4. **Variational Autoencoder (VAE)**: Probabilistic model trained on **280,000 sprites** ($\beta$-VAE with $\beta=0.001$). Learns a continuous probabilistic latent space for 2D asset clustering, fast vector similarity search, and smooth real-sprite-to-real-sprite interpolation (A → B). Evaluated on a 5,000-sprite candidate pool, achieving **84.40% KNN category accuracy** (**+19.21% lift** over the **65.19% majority baseline**).

---

## 📊 2. Single Unified Dataset Strategy (280K Game Assets)

All visual models (AE, VAE, and DCGAN) are trained on **one single dataset** to enable controlled benchmarking:
- **Dataset**: Alucard 2D Pixel-Art Game Asset Dataset
- **Volume**: **282,511 unique images** (after deduplication)
- **Dimensions for AE/VAE**: $128 \times 128 \times 4$ (RGBA format with transparency alpha masks)
- **Dimensions for DCGAN**: $64 \times 64 \times 4$ (RGBA, downscaled for GAN training stability)
- **Preprocessing**: Normalized to $[0.0, 1.0]$ for AE/VAE; $[-1, 1]$ (Tanh) for DCGAN
- **Splits**: 90% Train / 5% Val / 5% Test (consistent across all three visual models)

---

## ⚡ 3. Backend Architecture & High-Speed Optimizations

FastAPI backend (`backend/app.py`) serves model inference and latent space operations:

1. **Non-Blocking Instant Server Startup**:
   - Environment variables `TF_ENABLE_ONEDNN_OPTS=0` and `CUDA_VISIBLE_DEVICES=-1` set at top of `app.py` to prevent Windows GIL deadlocks.
   - Startup event completes in **0.05 seconds**, allowing status endpoints to respond immediately.

2. **Pre-Indexed VAE Latent Vectors (`ALUCARD_INDEX`)**:
   - All 25,000 Alucard sprites are pre-indexed into 256-dimensional VAE latent space.
   - `/find_similar_vae` executes vector dot-product matching against `ALUCARD_INDEX` in **0.001 seconds (1ms)**.

3. **Status Polling Integration**:
   - `gui/src/components/TopBar.jsx` polls `/status` every 3s via `checkAEStatus()` in `api.js`.
   - UI status badge dynamically displays model load status for AE, VAE, Transformer, and DCGAN.

---

## 📁 4. Key Artifact Locations

| Model | Key Files |
|---|---|
| **Transformer** | `models/gameforge_t5/` |
| **DCGAN** | `models/DCGAN_280K_COMPLETE/weights/generator_latest.keras` |
| **AE** | `models/280k dataset model/AE_280K_best.keras` |
| **VAE** | `models/280k model VAE (VAE v2)/VAE_280K_Outputs/` |
| **VAE Index (25K)** | `models/latent_gallery_index_25k.npy` + `latent_gallery_manifest_25k.json` |
| **DCGAN Evaluation** | `Notebook/DCGAN_280K_Demonstration_and_Evaluation.ipynb` |

---

## ⚖️ 5. Ethical Deployment & Responsible AI

1. **Creator Attribution**: Credit to original 2D pixel-art asset creators.
2. **Human-in-the-Loop Workflow**: AI provides semantic planning, novel sprite generation, and asset organization to assist indie game developers rather than replacing human artists.
3. **Responsible Content Boundaries**: Prompt filtering to prevent copyright infringement and inappropriate content.
4. **License Compliance**: All datasets used under their stated Creative Commons licenses (CC-BY-NC-SA 4.0 for Alucard/PICO-8, CC-BY 4.0 for ViGGO).
