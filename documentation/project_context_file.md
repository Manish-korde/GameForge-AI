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
     DISCRETE DCGAN   Visual Generation   Asset Library
    Novel Pixel Art        │
       Synthesis           ▼
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
2. **Discrete DCGAN**: Unconditional Deep Convolutional GAN with a discrete 32-color palette classification head (Notebook 02, 50 Epochs). Trained on 42,000 Alucard front-facing RGBA sprites. Generates brand-new 64×64 RGBA pixel-art sprites from random noise `z ~ N(0, I)`. Output has 48.8%–58.2% spatial coherence (vs real 53.0%) and ~26 unique colors, completely eliminating continuous RGB blur.
3. **Autoencoder (AE)**: Deterministic model trained on **280,000 sprites**. Used for high-precision deterministic reconstruction (`MSE: 0.000456`, `PSNR: 35.36 dB`, `SSIM: 0.961`) and outlier sprite screening (flagging the highest-error 5% of assets exceeding P95 MSE threshold `0.001313`).
4. **Variational Autoencoder (VAE)**: Probabilistic model trained on **280,000 sprites** ($\beta$-VAE with $\beta=0.001$). Learns a continuous probabilistic latent space for 2D asset clustering, fast vector similarity search, and smooth real-sprite-to-real-sprite interpolation (A → B). Evaluated on a 5,000-sprite candidate pool, achieving **84.40% KNN category accuracy** (**+19.21% lift** over the **65.19% majority baseline**).

---

## 📊 2. Single Unified Dataset Strategy (280K Game Assets)

All visual models (AE, VAE, and DCGAN) are trained on the Alucard 2D Pixel-Art Game Asset Dataset:
- **Dataset**: Alucard 2D Pixel-Art Game Asset Dataset (`evilsocket/alucard-sprites`)
- **Volume**: **282,511 unique images** (deduplicated corpus); DCGAN trained on clean front-facing subset of **42,000 character sprites**.
- **Dimensions for AE/VAE**: $128 \times 128 \times 4$ (RGBA format with transparency alpha masks)
- **Dimensions for DCGAN**: $64 \times 64 \times 4$ (RGBA, downscaled for GAN training stability)
- **Preprocessing**: Normalized to $[0.0, 1.0]$ for AE/VAE; Discrete 32-color palette classification for DCGAN
- **Splits**: Consistent splits across models.

---

## ⚡ 3. Backend Architecture & High-Speed Optimizations

FastAPI backend (`backend/app.py`) serves model inference and latent space operations:

1. **Non-Blocking Instant Server Startup**:
   - Environment variables `TF_ENABLE_ONEDNN_OPTS=0` and `CUDA_VISIBLE_DEVICES=-1` set at top of `app.py` to prevent Windows GIL deadlocks.
   - Startup event completes in **0.05 seconds**, allowing status endpoints to respond immediately.

2. **Pre-Indexed VAE Latent Vectors (`ALUCARD_INDEX`)**:
   - All 25,000 Alucard sprites are pre-indexed into 256-dimensional VAE latent space.
   - `/find_similar_vae` executes vector dot-product matching against `ALUCARD_INDEX` in **0.001 seconds (1ms)**.

3. **Discrete DCGAN Production Service (`backend/dcgan_service.py`)**:
   - Loads 50-epoch discrete generator (`generator_weights_final.weights.h5`) + empirical 32-color palette (`global_palette_32.npy`).
   - Executes truncation ($\sigma=0.85$), weapon-safe LCC floater filtering, and Best-of-8 candidate rejection sampling ($0.15 \le \text{FG} \le 0.55$).
   - Returns crisp 8× nearest-neighbor upscaled pixel art ($512\times 512$) for instant GUI display.

---

## 📁 4. Key Artifact Locations

| Model | Canonical Production Directory & Key Files |
|---|---|
| **1: AE** | `models/1_AE_280k_v1/AE_280K_best.keras` |
| **2: VAE** | `models/2_VAE_280k_v2/VAE_280K_Outputs/encoder_280k_final.keras` |
| **3: DCGAN** | `models/3_DCGAN_42k_v1/game_ready_assets/generator_weights_final.weights.h5` |
| **4: Transformer** | `models/4_Transformer_2k_v1/model.safetensors` |
| **5: Diffusion** | `models/5_Diffusion_LPC_v1/checkpoint_best.pt` |
| **VAE Index (25K)** | `models/latent_gallery_index_25k.npy` + `latent_gallery_manifest_25k.json` |
| **DCGAN Training Archive** | `C:\Users\manis\Downloads\dcgan_frontfacing_results` |
| **DCGAN Evaluation Sheets** | `phase_dcgan_improvement/controlled_variant_comparison_master.png` |

---

## ⚖️ 5. Ethical Deployment & Responsible AI

1. **Creator Attribution**: Credit to original 2D pixel-art asset creators.
2. **Human-in-the-Loop Workflow**: AI provides semantic planning, novel sprite generation, and asset organization to assist indie game developers rather than replacing human artists.
3. **Responsible Content Boundaries**: Prompt filtering to prevent copyright infringement and inappropriate content.
4. **License Compliance**: All datasets used under their stated Creative Commons licenses (CC-BY-NC-SA 4.0 for Alucard/PICO-8, CC-BY 4.0 for ViGGO).
