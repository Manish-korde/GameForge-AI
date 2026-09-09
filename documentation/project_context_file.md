# GameForge AI — Master Project Context & Knowledge Base 🎮🤖

This document serves as the definitive architectural knowledge base for **GameForge AI**. It documents core concepts, model roles, dataset configurations, backend optimizations, and review presentation strategies.

---

## 🧠 1. Project Overview & Multi-Model Pipeline Strategy

**GameForge AI** is a multi-model Generative AI platform for 2D game asset creation, semantic planning, and latent space analysis.

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
                   ┌───────┴───────┐
                   │               │
                   ▼               ▼
             Visual Generation   Asset Library
                   │
                   ▼
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

### The Three Review Components:
1. **Transformer Model**: Semantic planning layer. Translates natural language game descriptions into structured game specification sheets (JSON) detailing themes, character types, environment features, weapon properties, and NPC specs.
2. **Autoencoder (AE)**: Baseline model trained on **280,000 sprites**. Used for high-precision deterministic reconstruction (`MSE: 0.00166`, `PSNR: 28.52 dB`, `SSIM: 0.9248`) and outlier sprite screening (flagging the highest-error 5% of assets exceeding P95 MSE threshold `0.001313`).
3. **Variational Autoencoder (VAE)**: Probabilistic model trained on **280,000 sprites** ($\beta$-VAE with $\beta=0.001$). Learns a continuous probabilistic latent space for 2D asset clustering, fast vector similarity search, and smooth real-sprite-to-real-sprite interpolation (A → B). Evaluated on a 5,000-sprite candidate pool, achieving **84.40% KNN category accuracy** (**+19.21% lift** over the **65.19% majority baseline**). *Framed as probabilistic latent space analysis and experimental sampling (avoiding claims of production-ready sprite generation).*

---

## 📊 2. Single Unified Dataset Strategy (280K Game Assets)

Both AE and VAE are evaluated on **one single dataset** to enable controlled benchmarking:
- **Dataset**: Alucard 2D Pixel-Art Game Asset Dataset
- **Volume**: **280,000 images**
- **Dimensions**: $128 \times 128 \times 4$ (RGBA format with transparency alpha masks)
- **Preprocessing**: Normalized to $[0.0, 1.0]$
- **Splits**: 80% Train / 10% Val / 10% Test

---

## ⚡ 3. Backend Architecture & High-Speed Optimizations

FastAPI backend (`backend/app.py`) serves model inference and latent space operations:

1. **Non-Blocking Instant Server Startup**:
   - Environment variables `TF_ENABLE_ONEDNN_OPTS=0` and `CUDA_VISIBLE_DEVICES=-1` set at top of `app.py` to prevent Windows GIL deadlocks.
   - Startup event completes in **0.05 seconds**, allowing status endpoints to respond immediately.

2. **Pre-Indexed VAE Latent Vectors (`ALUCARD_INDEX`)**:
   - All 20 local Alucard 280k dataset sprites are pre-indexed into 256-dimensional VAE latent space in **1 single fast batch call** on startup (< 0.1s).
   - `/find_similar_vae` executes vector dot-product matching against `ALUCARD_INDEX` in **0.001 seconds (1ms)**.

3. **Status Polling Integration**:
   - `gui/src/components/TopBar.jsx` polls `/status` every 3s via `checkAEStatus()` in `api.js`.
   - UI status badge dynamically displays:
     <span style="color: #4ade80;">**Autoencoder: Loaded | VAE: Loaded**</span> in bright green.

---

## ⚖️ 4. Ethical Deployment & Responsible AI

1. **Creator Attribution**: Credit to original 2D pixel-art asset creators.
2. **Human-in-the-Loop Workflow**: AI provides semantic planning and asset organization to assist indie game developers rather than replacing human artists.
3. **Responsible Content Boundaries**: Prompt filtering to prevent copyright infringement and inappropriate content.
