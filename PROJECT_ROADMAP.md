# GameForge AI — Project Roadmap 🗺️

Welcome to the roadmap for **GameForge AI**. This document tracks our progression across model components for our upcoming project review and future development milestones.

---

## 🎯 Phase 1: Semantic Planning — Transformer Model (🚧 ACTIVE - 2-DAY SPRINT)
- **Role**: Free-form natural language request parsing into structured game specifications (JSON) detailing themes, character roles, environment settings, weapon attributes, and NPC parameters.
- **Implementation**: Pretrained Transformer model (e.g., T5-small / DistilGPT2 / HuggingFace Pipeline) integrated via FastAPI endpoint `/generate_concept` and React GUI "Game Concept" tab.

## 🎯 Phase 2: High-Precision Reconstruction — Autoencoder (✅ COMPLETED)
- **Dataset**: Full **280,000 2D Game Asset Dataset (Alucard)**.
- **Performance**: High reconstruction fidelity (MSE `0.00166`, PSNR `28.52 dB`, SSIM `0.9248`).
- **Defensible Application**: Deterministic compression baseline & **outlier sprite data cleaning** (flagging corrupted assets via high reconstruction error).

## 🧬 Phase 3: Probabilistic Latent Analysis — VAE (✅ COMPLETED & EVALUATED)
- **Dataset**: Full **280,000 2D Game Asset Dataset (Alucard)**.
- **Model**: $\beta$-VAE ($\beta=0.001$).
- **Defensible Application**: **Probabilistic Latent Space Analysis, 2D Asset Clustering, & Controlled Real-Sprite Interpolation (A → B)**.
- **Framing**: Demonstrates continuous latent space navigation and similarity search; sampling and interpolation are presented as experimental generative capabilities.

---

## 📊 Single Unified Dataset Strategy (280K Game Assets)
Both AE and VAE are benchmarked on the **same 280,000 RGBA sprite dataset** ($128 \times 128 \times 4$), providing a controlled evaluation of deterministic vs. probabilistic representations.

---

## ⚖️ Ethics & Responsible Deployment
- Dataset attribution and licensing compliance.
- Human-in-the-loop workflow positioning AI as an indie developer assistant.
- Prompt safety and copyright boundary enforcement.
