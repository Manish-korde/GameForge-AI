# GameForge AI — Project Roadmap 🗺️

Welcome to the roadmap for **GameForge AI**. This document tracks our progression across model components for our upcoming project review and future development milestones.

---

## 🎯 Phase 1: Semantic Planning — Transformer Model (🚧 ACTIVE)
- **Role**: Free-form natural language request parsing into structured game specifications (JSON) detailing themes, character roles, environment settings, weapon attributes, and NPC parameters.
- **Implementation**: Fine-tuned Transformer model (`google/flan-t5-small`) integrated via FastAPI endpoint `/generate_concept` and React GUI "Game Concept" tab.

## 🎯 Phase 2: High-Precision Reconstruction & Outlier Screening — Autoencoder (✅ COMPLETED)
- **Dataset**: Full **282,511 Unique 2D Game Asset Dataset (Alucard)** ($128 \times 128 \times 4$ RGBA).
- **Performance**: High reconstruction fidelity (Median MSE `0.000782`, Mean MSE `0.001614`, PSNR `28.52 dB`, SSIM `0.9248`).
- **Defensible Application**: **Reconstruction-Based Outlier Screening & Asset Quality Control** (surfacing the highest-error 5% of sprites exceeding the empirical $95\text{th}$ percentile MSE threshold `0.001313` for automated review).

## 🧬 Phase 3: Probabilistic Latent Analysis & Visual Search — VAE (✅ COMPLETED & EVALUATED)
- **Dataset**: Full **282,511 Unique 2D Game Asset Dataset (Alucard)** (Evaluation run on a **5,000-sprite candidate pool / indexed subsample** for computational feasibility).
- **Model**: $\beta$-VAE ($\beta=0.001$, 256-dim Gaussian latent prior).
- **Empirical Benchmarks**:
  - **KNN Supervised Category Accuracy**: **84.40%** vs. **65.19% majority-class baseline** (**+19.21% lift**; vs AE's `77.80%`, Macro F1 `0.7932`).
  - **K-Means Unsupervised Clustering**: Silhouette `0.1624`, ARI `0.3120`, NMI `0.3845`.
  - **Held-Out Retrieval Precision (5K Subsample)**: **Precision@1 = 66.00%**, **Precision@5 = 62.80%**.
- **Defensible Applications**:
  - **Primary**: **Visual Asset Search & Related Asset Discovery** (Nearest-neighbor retrieval in continuous latent space).
  - **Secondary**: **Latent Space Exploration, 2D Clustering, & Real-Sprite Interpolation (A → B)**.
- **Framing**: Demonstrates continuous latent space navigation and similarity search; unconditional sampling is presented accurately as an experimental generative capability.

## ⚡ Phase 4: Unconditional Novel Sprite Generation — DCGAN (✅ COMPLETED)
- **Dataset**: Full **282,511 Unique 2D Game Asset Dataset (Alucard)**, resolution downscaled to **64×64 RGBA** for GAN training stability.
- **Architecture**: Deep Convolutional GAN (DCGAN) — `z ∈ ℝ^{100} ~ N(0,I)` → 4× `Conv2DTranspose` → 64×64×4 RGBA sprite (Tanh).
- **Training**: 50 epochs, 99,300 global optimizer steps, ~5.67 training hours on Kaggle P100.
- **Adversarial Stability**: D-loss range 0.015–0.040 (stable), G-loss range 8.0–8.9 — no mode collapse detected.
- **Why Unconditional**: Alucard category labels are noisy/unvalidated. Unconditional DCGAN learns the full sprite visual distribution directly from pixels, with zero label dependency.
- **Artifacts**: `models/DCGAN_280K_COMPLETE/` — weights, checkpoints, logs, epoch sample grids.
- **Evaluation Notebook**: `Notebook/DCGAN_280K_Demonstration_and_Evaluation.ipynb`.
- **Defensible Application**: **Genuinely novel sprite synthesis** — brand-new RGBA game sprites sampled from an adversarially learned prior, distinct from AE reconstruction and VAE latent sampling.

---

## 📊 Single Unified Dataset Strategy (282,511 Unique Game Assets)
All deep learning visual models (AE, VAE, and DCGAN) are trained on the **same Alucard RGBA sprite dataset**, providing a controlled experimental environment where model differences are due to architecture and objective — not data.

- **Empirical Category Breakdown (100% Total)**: Characters (**65.19%**), Unknown / Miscellaneous (**15.84%**), Items (**6.39%**), Enemies (**5.75%**), Weapons (**4.38%**), Tiles (**1.92%**), Props (**0.35%**), Effects (**0.18%**).

---

## ⚖️ Ethics & Responsible Deployment
- Dataset attribution (`evilsocket/alucard-sprites`) and licensing compliance.
- Human-in-the-loop workflow positioning AI as an indie developer assistant.
- Prompt safety and copyright boundary enforcement.
