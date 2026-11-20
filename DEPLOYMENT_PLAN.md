# GameForge AI: Deployment & Production Plan

## 1. System Overview & Model Packaging
GameForge AI is an end-to-end procedural game sprite generation and animation pipeline. The production runtime consists of five standardized model stages:

| Stage | Model Name | Canonical Path | Format / Exact Measured Size | Purpose |
|---|---|---|---|---|
| **1** | Autoencoder (AE) | `models/1_AE_280k_v1/AE_280K_best.keras` | Keras 3 (**12.41 MB**) | Unsupervised structural QA screening (MSE scoring) |
| **2** | Variational AE (VAE) | `models/2_VAE_280k_v2/VAE_280K_Outputs/` | Keras 3 (**52.25 MB** enc+dec) | Latent interpolation & character variant studio |
| **3** | Discrete DCGAN | `models/3_DCGAN_42k_v1/game_ready_assets/` | Weights + NPY (**10.48 MB**) | Core 64x64 character sprite synthesis (Epoch 50) |
| **4** | Flan-T5 Spec Generator | `models/4_Transformer_2k_v1/` | HuggingFace Safetensors (**293.50 MB**) | Natural language to RPG character spec compilation |
| **5** | Action/Pose Diffusion | `models/5_Diffusion_LPC_v1/checkpoint_best.pt` | PyTorch checkpoint (**30.20 MB** best+action) | 8-frame walk cycle sprite sheet diffusion generator |

---

## 2. Measured Size Budget & Asset Pruning
- **Current Development Tree:** ~8.9 GB (contains raw datasets, intermediate training checkpoints `ckpt-1`..`ckpt-24`, development venv, caches).
- **Exact Standalone Production Package:** **399.19 MB** total:
  - **Core Models (5 stages):** 398.83 MB
  - **Production GUI (`gui/dist/` SPA):** 0.36 MB
- **Packaging Rules:**
  - `models/3_DCGAN_42k_v1`: Ship only `generator_weights_final.weights.h5`, `global_palette_32.npy`, and `fixed_latent_vectors.npy`. Exclude raw training logs, backup weights, and continuation checkpoints.
  - `models/4_Transformer_2k_v1`: Ship only `model.safetensors`, `config.json`, `tokenizer.json`, `tokenizer_config.json`, and `special_tokens_map.json`.
  - `models/5_Diffusion_LPC_v1`: Ship only `checkpoint_best.pt` and `best_diffusion_50ep.pt`.
  - Exclude all `.venv`, `gui/node_modules`, `tests`, `train_*.py`, and `Dataset/` directories from the release zip.

---

## 3. Environment & Hardware Matrix
- **Target OS:** Windows 10/11 64-bit or Linux (Ubuntu 22.04 LTS).
- **Target Python Version:** **Python 3.10 (64-bit)**.
- **Pinned Dependency Baseline (`requirements_production.txt`):**
  - `tensorflow==2.21.0` (Pinned: exact verified version where Keras weights load cleanly)
  - `torch==2.14.0`
  - `transformers==5.16.1`
  - `fastapi==0.141.1`, `uvicorn==0.52.4`, `scipy==1.15.3`, `pillow==12.3.0`, `numpy==2.2.6`
- **Measured Latency Profile:**
  - **CPU Mode (Measured on i7 / Ryzen 64-bit Windows):**
    - Discrete DCGAN (single pass): **365 ms**
    - Discrete DCGAN with **8-candidate rejection sampling**: **725 ms**
    - Autoencoder QA Screening: **0.5 ms**
    - Multi-Pose Diffusion (8-frame walk cycle, 25 steps): **26.56 s** (3.32 s / frame)
    - Flan-T5 Spec Compilation (Cold call): **34.3 s** (Cached / Rehearsed: **< 10 ms**)
  - **GPU Mode:**
    - GPU acceleration (CUDA 12.1+ / DirectML): Diffusion walk cycle estimated < 4s, DCGAN < 100 ms. Live demo default operates reliably on CPU fallback without requiring external CUDA setup.

---

## 4. Run Steps & Portable Service Architecture (No Node.js Required)
1. **Frontend Architecture:** The React single-page application is pre-compiled into static production assets (`gui/dist/`, 0.36 MB). **Node.js is NOT required on the deployment target.**
2. **Unified Single-Process Backend:** FastAPI directly mounts and serves `gui/dist/` at `http://localhost:8000/`. A single web server handles both API endpoints and the GUI.
3. **Warm-up & Caching on Startup:**
   - On startup, the backend pre-loads all 5 model stages into memory and executes a dummy warm-up generation pass for DCGAN and Flan-T5.
   - Common demo prompts (e.g., Knight, Archer, Mage) have pre-cached Flan-T5 specs in `backend/cache/demo_specs.json`, dropping first-turn latency from ~34s to <10ms.
4. **Portable One-Click Launcher (`run_gameforge.bat`):**
   ```bat
   @echo off
   echo ========================================================
   echo Launching GameForge AI Production Runtime
   echo ========================================================
   cd /d "%~dp0"
   if not exist "venv\Scripts\python.exe" (
       echo Creating fresh clean-machine virtualenv...
       python -m venv venv
       call venv\Scripts\activate
       python -m pip install --upgrade pip
       pip install -r requirements_production.txt
   ) else (
       call venv\Scripts\activate
   )
   echo Starting GameForge Unified Server on http://localhost:8000...
   start http://localhost:8000
   python -m uvicorn backend.app:app --host 127.0.0.1 --port 8000
   ```

---

## 5. Clean-Machine Test Protocol (Fresh Virtualenv Verification)
To verify reproducibility on an unconfigured evaluation machine:
1. Extract standalone zip bundle (approx. 400 MB).
2. Run automated verification:
   ```cmd
   python -m venv test_clean_venv
   test_clean_venv\Scripts\activate
   pip install -r requirements_production.txt
   python scripts\run_full_verification_suite.py
   ```
3. Script validates:
   - Python version and pinned packages (`tensorflow==2.21.0`, `torch==2.14.0`).
   - Bit-exactness against notebook weights (0 pixel difference).
   - 20 consecutive DCGAN runs (binary alpha $\{0, 255\}$, $\le 32$ colors, $0.15 \le \text{FG} \le 0.55$).
   - End-to-end prompt $\to$ DCGAN $\to$ AE QA $\to$ Diffusion walk cycle pipeline.

---

## 6. Backups, Operational Risks & Mitigation

| Identified Risk | Severity | Root Cause | Built-in Production Mitigation |
|---|---|---|---|
| **Live Demo Prompt Delay** | High | Flan-T5 cold CPU generation takes ~34s | Startup warm-up pass + pre-cached specs for rehearsed demo prompts (`backend/cache/demo_specs.json`). |
| **Diffusion CPU Duration** | Medium | DDIM 25 steps takes ~26s on CPU | Rehearsed demo defaults to 15 DDIM steps (~15s) or pre-rendered offline walk cycles. |
| **Missing Weights / Path Drift** | Critical | Broken symlink or wrong directory | `backend/dcgan_service.py` fails loudly on startup with explicit `FileNotFoundError`; zero fallback to continuous generator. |
| **Rejection Sampler Discrepancy** | Low | Mismatch in candidate count | Standardized to exactly **8 candidate variations** per generation with LCC floater filter (<12 px dropped unless touching character). |
| **Autoencoder Pixel Tampering** | Low | Coupling error | AE is strictly sandboxed as a read-only scoring gate (`score_sprite_with_ae`); never reconstructs or overwrites sprite bytes. |

---

## 7. Live Demo Offline Fallback Strategy
If live execution is interrupted or experiencing CPU delays during presentation:
- **Offline Fallback Gallery:** 30 pre-generated, AE-approved character sprites covering 6 RPG archetypes in `demo_offline_gallery/`.
- **Pre-rendered Walk Cycles:** Pre-cached 8-frame walk cycle GIF and sprite sheet (`phase_dcgan_improvement/pipeline_smoke_walk.gif`).
- **Interactive GUI Fallback:** The GUI contains an "Offline Demo Gallery" button allowing instant presentation of high-resolution pixel art and animations with 0 latency.
