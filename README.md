# GameForge AI 🎮🤖

Welcome to **GameForge AI**! This project is a multi-model Generative AI pipeline designed for 2D game asset design, semantic planning, latent space analysis, and high-fidelity reconstruction.

---

## 🧠 Master Pipeline Architecture (Review Framing)

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

---

## 🔬 Core Model Components & Technical Applications

Our project presents **three distinct model components**, each fulfilling a technically defensible, non-overlapping role:

### 1. Transformer Model (Semantic Planning)
* **Role**: Translates free-form natural language game requests into structured game specification sheets (identifying themes, character types, environment features, weapon properties, and NPC attributes).
* **Technical Value**: Provides the natural language interface and semantic structuring layer that grounds downstream asset requirements.

### 2. Autoencoder (AE)
* **Dataset**: Trained & evaluated on the **280,000 2D Game Asset Dataset (Alucard)**.
* **Role**: Deterministic baseline compression and high-precision pixel-art reconstruction.
* **Performance**: MSE `0.00166`, PSNR `28.52 dB`, SSIM `0.9248`.
* **Defensible Application**: **Data Cleaning & Outlier Detection**. High-reconstruction-error assets from the trained AE are automatically flagged as corrupted, misaligned, or outlier sprites.

### 3. Variational Autoencoder (VAE)
* **Dataset**: Trained & evaluated on the **280,000 2D Game Asset Dataset (Alucard)**.
* **Role**: Probabilistic latent representation learning ($z \sim \mathcal{N}(\mu, \sigma^2)$).
* **Defensible Application**: **Probabilistic Latent Space Analysis & Controlled Real-Sprite Interpolation (A → B)**.
* **Framing Note**: The VAE learns a continuous latent representation of game assets for 2D visual clustering, similarity search, and smooth real-sprite-to-real-sprite morphing. *Sampling and interpolation are presented as experimental generative capabilities rather than production-ready asset generators.*

---

## 📊 The Single Dataset Story: 280K Game Assets

To ensure a controlled, fair scientific comparison between AE and VAE, **both models were trained on the exact same dataset**:

```text
              280K Game Assets (Alucard Dataset)
                     │
             ┌───────┴───────┐
             ↓               ↓
            AE              VAE
             ↓               ↓
       Reconstruction   Latent Space
       / Data Cleaning  Analysis & Interpolation
```

### Dataset Specifications:
- **Source**: 280,000 2D Pixel-Art Game Sprites (Alucard Dataset).
- **Dimensions**: $128 \times 128 \times 4$ (RGBA with hard alpha transparency mask).
- **Preprocessing**: Pixel normalization to $[0.0, 1.0]$.
- **Splits**: 80% Train, 10% Validation, 10% Test.

---

## 🚀 How to Run the Project

The project consists of a Python FastAPI backend and a React/Vite web GUI.

### 1. AI Backend (FastAPI)
```bash
cd backend
# Activate virtual environment
.\venv\Scripts\activate
# Start Uvicorn server
uvicorn app:app --port 8000
```
- Status Endpoint: `http://localhost:8000/status`
- Similarity Search Endpoint: `http://localhost:8000/find_similar_vae`

### 2. React Web GUI
```bash
cd gui
npm install
npm run dev
```
Open `http://localhost:5173` in your browser.

---

## ⚖️ Ethical Deployment & Responsible AI

1. **Dataset Licensing & Credit**: Attribution to original 2D pixel-art creators.
2. **Human-in-the-Loop Oversight**: AI serves as an assistive workflow tool for indie game developers rather than replacing human artists.
3. **Responsible Content Boundaries**: Built-in prompt filtering to enforce safe, non-infringing asset generation.
