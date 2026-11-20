# GameForge AI 🎮🤖

Welcome to **GameForge AI**! This project is a multi-model Generative AI pipeline designed for 2D game asset design, semantic planning, adversarial sprite synthesis, latent space analysis, high-fidelity reconstruction, and from-scratch action-conditioned diffusion animation.

---

## 🧠 Master Pipeline Architecture

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
           ┌───────────────┼───────────────┬────────────────┐
           │               │               │                │
           ▼               ▼               ▼                ▼
       DCGAN         Visual Generation   Asset Library   MULTI-POSE DIFFUSION
    Novel Sprite                                         Action & Pose Studio
    Generation                                           (Idle & Walk F1-F8)
                           │
                     ┌─────┴─────┐
                     ▼           ▼
                    AE           VAE
                     │           │
               Reconstruction   Latent
               / QA Screening  Representation
                     │           │
                     └─────┬─────┘
                           ▼
                     Asset Analysis
```

---

## 🔒 Production Model Registry & Integrity Lock

All five production models in the GameForge AI lineage are finalized, validated, and locked:

| Model ID | Module Name | Framework & Format | Parameter Count | Canonical Checkpoint Path | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **1: AE** | Convolutional AE | TensorFlow / Keras | 2.1M params | `models/1_AE_280k_v1/AE_280K_best.keras` | **PRODUCTION** |
| **2: VAE** | Variational Autoencoder (VAE) | TensorFlow / Keras | 4.3M params | `models/2_VAE_280k_v2/VAE_280K_Outputs/encoder_280k_final.keras` | **PRODUCTION** |
| **3: DCGAN** | Discrete-Palette DCGAN | TensorFlow / Keras H5 | 2.6M params | `models/3_DCGAN_42k_v1/game_ready_assets/generator_weights_final.weights.h5` | **PRODUCTION ACTIVE** |
| **4: Transformer** | Semantic Planner | HuggingFace / PyTorch | 60M params | `models/4_Transformer_2k_v1/model.safetensors` | **PRODUCTION** |
| **5: Diffusion** | Multi-Pose Velocity Diffusion | PyTorch (`ActionAndPoseConditionedUNet`) | 1.98M params | `models/5_Diffusion_LPC_v1/checkpoint_best.pt` | **PRODUCTION READY** |

---

## 🔬 Core Model Components & Technical Applications

Our project presents **five distinct model components**, each fulfilling a technically defensible, non-overlapping role:

### 1. Transformer Model (Semantic Planning)
* **Role**: Translates free-form natural language game requests into structured game specification sheets (identifying themes, character types, environment features, weapon properties, and NPC attributes).
* **Technical Value**: Provides the natural language interface and semantic structuring layer that grounds downstream asset requirements.

### 2. Multi-Pose Velocity Diffusion Studio (Module 4) ✅ Production Ready
* **Role**: From-scratch, reference-grounded velocity-prediction diffusion model synthesizing idle and complete 8-frame walk animation cycles from pure Gaussian noise ($\mathbf{x}_T \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$).
* **Architecture**: Custom `ActionAndPoseConditionedUNet` with 4-channel reference spatial conditioning, time sinusoidal embeddings, and categorical action/pose embeddings.
* **Diffusion Formulation**: 50-step reverse DDIM sampling with velocity parameterization ($v_\theta$) under a 1000-timestep cosine variance schedule.
* **Capabilities**:
  - Action 0: Idle Stance (Pose 0)
  - Action 1: South Walk Cycle (Frames F1 through F8), including Left Apex (F3) and Right Apex (F7)
  - Seamless 512×64 horizontal sprite sheet assembly and looping 8 FPS animated GIF export.
  - Bit-exact deterministic reproducibility ($L_\infty = 0.000000$ difference under identical seeds).
* **Artifacts**: Checkpoint `phase_F_experiment_E1/checkpoint_best.pt` | Verified via Phase G Technical Validation Suite.

### 3. DCGAN (Unconditional Novel Sprite Generator) ⚠️ Under Refinement
* **Dataset**: Trained on 169,094 candidate character crops ($64 \times 64$ RGBA).
* **Architecture**: Deep Convolutional GAN — $z \in \mathbb{R}^{128} \rightarrow \text{Dense} \rightarrow 4 \times \text{Conv2D} \rightarrow 64 \times 64 \times 4$ Tanh with Nearest-Neighbor upsampling.
* **Why Unconditional**: Alucard category labels are noisy/unvalidated. Unconditional DCGAN learns the full sprite visual distribution from pixels alone.
* **Current Limitation**: Training data contained mixed perspectives (front, side profiles, back-facing), causing mode conflict and morphological blurring. Refinement plan underway.

### 4. Autoencoder (AE)
* **Dataset**: Trained & evaluated on the **282,511 2D Game Asset Dataset (Alucard)** ($128 \times 128 \times 4$ RGBA).
* **Role**: Deterministic baseline compression and high-precision pixel-art reconstruction.
* **Performance**: Median MSE `0.000782`, PSNR `28.52 dB`, SSIM `0.9248`.
* **Defensible Application**: **Data Quality Verification & Outlier Screening**. Automatically surfaces the highest-error 5% of assets exceeding the empirical $95\text{th}$ percentile MSE threshold (`0.001313`) for review.

### 5. Variational Autoencoder (VAE)
* **Dataset**: Trained & evaluated on the **282,511 2D Game Asset Dataset (Alucard)**.
* **Active Search Index**: **25,000 Pre-Indexed Latent Vectors** (`latent_gallery_index_25k.npy`) serving distance matching in **< 1ms**.
* **Role**: Probabilistic latent representation learning ($z \sim \mathcal{N}(\mu, \sigma^2)$).
* **Performance**: Evaluated on a 5,000-sprite candidate pool, achieving **84.40% KNN category accuracy** (a **+19.21 percentage point lift** over the 65.19% majority-class baseline) and **62.80% Precision@5** retrieval precision.
* **Defensible Application**: **Probabilistic Latent Space Analysis, Visual Similarity Search, & Controlled Real-Sprite Morphing (A → B)**.

---

## 📊 The Single Dataset Story: 282K Game Assets

To ensure a controlled, fair scientific comparison, all visual models were grounded in the **282,511 2D Game Asset Dataset (Alucard)**:

```text
               282,511 Game Assets (Alucard Dataset)
                         │
          ┌──────────────┼──────────────┬──────────────┐
          ↓              ↓              ↓              ↓
         AE             VAE          DCGAN         DIFFUSION
          ↓              ↓              ↓              ↓
   Reconstruction   Continuous    Novel Sprite   Action & Pose
   MSE / QA (P95)  Latent Space  Synthesis      Transformation
(Median 0.000782) (84.4% KNN)   (169K Crops)   (Pure Gaussian DDIM)
```

---

## 🚀 How to Run the Application

The project consists of a Python FastAPI backend and a React/Vite web GUI.

### 1. AI Backend (FastAPI)
```bash
# Activate virtual environment
.\backend\venv\Scripts\Activate.ps1

# Start Uvicorn server on port 8000
python -m uvicorn backend.app:app --host 127.0.0.1 --port 8000 --reload
```
- Multi-Pose Diffusion Status: `http://127.0.0.1:8000/generate/diffusion/status`
- General Status: `http://127.0.0.1:8000/status`

### 2. React Web GUI
```bash
cd gui
npm install
npm run dev
```
Open [http://localhost:5173/action-studio](http://localhost:5173/action-studio) in your browser.

---

## ⚖️ Ethical Deployment & Responsible AI

1. **Dataset Licensing & Credit**: Attribution to original 2D pixel-art creators (`evilsocket/alucard-sprites`, CC-BY-NC-SA 4.0).
2. **Human-in-the-Loop Oversight**: AI serves as an assistive workflow tool for indie game developers rather than replacing human artists.
3. **Transparent Model Boundaries**: Discloses validated capabilities and known physical limitations (e.g., F7 right-leg stride compression: 38.9%).
