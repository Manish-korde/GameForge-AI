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
* **Dataset**: Trained & evaluated on the **282,511 2D Game Asset Dataset (Alucard)** ($128 \times 128 \times 4$ RGBA).
* **Role**: Deterministic baseline compression and high-precision pixel-art reconstruction.
* **Performance**: Median MSE `0.000782`, PSNR `28.52 dB`, SSIM `0.9248`.
* **Defensible Application**: **Data Quality Verification & Outlier Screening**. Automatically surfaces the highest-error 5% of assets exceeding the empirical $95\text{th}$ percentile MSE threshold (`0.001313`) for review.

### 3. Variational Autoencoder (VAE)
* **Dataset**: Trained & evaluated on the **282,511 2D Game Asset Dataset (Alucard)**.
* **Active Search Index**: **25,000 Pre-Indexed Latent Vectors** (`latent_gallery_index_25k.npy`) serving distance matching in **< 1ms**.
* **Role**: Probabilistic latent representation learning ($z \sim \mathcal{N}(\mu, \sigma^2)$).
* **Performance**: Evaluated on a 5,000-sprite candidate pool, achieving **84.40% KNN category accuracy** (a **+19.21 percentage point lift** over the 65.19% majority-class baseline) and **62.80% Precision@5** retrieval precision.
* **Defensible Application**: **Probabilistic Latent Space Analysis, Visual Similarity Search, & Controlled Real-Sprite Morphing (A → B)**.

---

## 📊 The Single Dataset Story: 280K Game Assets

To ensure a controlled, fair scientific comparison between AE and VAE, **both models were trained on the exact same 282,511 dataset**:

```text
               282,511 Game Assets (Alucard Dataset)
                         │
         ┌───────────────┴───────────────┐
         ↓                               ↓
        AE                              VAE
         ↓                               ↓
  Reconstruction MSE            Continuous Latent Space
 Quality Control (P95)       Visual Search (25K Index < 1ms)
(Median MSE: 0.000782)      (KNN Acc: 84.4% vs 65.2% baseline)
```

### Dataset Specifications:
- **Source**: 282,511 2D Pixel-Art Game Sprites (Alucard Dataset).
- **Dimensions**: $128 \times 128 \times 4$ (RGBA with transparent alpha mask).
- **100.00% Category Breakdown**: Characters (**65.19%**), Unknown/Misc (**15.84%**), Items (**6.39%**), Enemies (**5.75%**), Weapons (**4.38%**), Tiles (**1.92%**), Props (**0.35%**), Effects (**0.18%**).
- **Project Status & Ground Truth**: See [CURRENT_PROJECT_STATUS.md](file:///c:/Users/manis/OneDrive/Desktop/Prompt_to_game_asset_generator/documentation/CURRENT_PROJECT_STATUS.md) for the definitive status document.

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
