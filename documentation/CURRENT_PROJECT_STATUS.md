# 📌 GameForge AI — Current Project Status & Architectural Snapshot 🚀

**Last Updated**: September 8, 2026  
**Project Version**: 2.0 (280K Corpus Scale & Multi-Dataset Evaluation)  
**Definitive Purpose**: This document establishes the absolute ground truth regarding our dataset scale, model training status, vector index scale, evaluation benchmarks, and architectural state. **Any assumption that the project is still on the 10K dataset or VAE v1 stage is outdated.**

---

## 🚫 Myths vs. ✅ Actual Project Reality

| Topic | 🚫 Outdated / Incorrect Assumption | ✅ Actual Current Ground Truth |
| :--- | :--- | :--- |
| **Dataset Scale** | "Trained on 10,000 (10k) images" | **Fully trained & evaluated on the 282,511 Alucard Game Asset Dataset** ($128 \times 128 \times 4$ RGBA). |
| **AE vs VAE Corpus** | "AE and VAE were trained on different datasets" | **Single Unified Dataset Strategy**: Both AE and VAE were benchmarked on the exact same 282,511 RGBA sprite corpus. |
| **Vector Search Pool** | "Searching across 5,000 sprites" | **25,000-Sprite Pre-Indexed Vector Gallery** (`latent_gallery_index_25k.npy` & `latent_gallery_manifest_25k.json`) serving sub-millisecond searches (< 1ms). |
| **KNN Category Accuracy** | "84.40% KNN accuracy presented without baseline" | **84.40% KNN category accuracy** vs. **65.19% majority-class baseline** (*Characters*), yielding a **+19.21 percentage point lift**. |
| **Evaluation Candidate Pool** | "Metrics are full-corpus continuous metrics" | Evaluated on a **5,000-sprite candidate pool / indexed subsample** (Precision@5 = **62.80%**, Precision@1 = **66.00%**). |
| **Category Distribution Total** | "Categories sum to 84.16%" | **100.00% Total Population**: Characters (**65.19%**), Unknown/Misc (**15.84%**), Items (**6.39%**), Enemies (**5.75%**), Weapons (**4.38%**), Tiles (**1.92%**), Props (**0.35%**), Effects (**0.18%**). |
| **Quality Control (QA) Metric** | "Cuts review time by up to 90%" | **Mathematically Grounded P95 Outlier Threshold**: Automatically flags the highest-error 5% of sprites exceeding MSE `0.001313` for review. |
| **Cross-Dataset Validation** | "Only Alucard dataset evaluated" | Fully benchmarked across **3 Datasets**: Alucard (282,511 Sprites), ViGGO (6,900 Dialogue Samples), and PICO-8 (10,967 Game Cartridges). |

---

## 🧠 Master Architecture & Model Status

```text
                           GAMEFORGE AI PIPELINE
                                    │
                           Natural Language Request
                                    │
                                    ▼
                         [ TRANSFORMER MODEL ]
                      Semantic Planning (Flan-T5-Small)
                                    │
                                    ▼
                 Structured Game Specification Sheet (JSON)
                                    │
                 ┌──────────────────┴──────────────────┐
                 ▼                                     ▼
      [ Visual Asset Search & Explorer ]    [ Concept Asset Generator ]
         (25,000 Vector Index < 1ms)            (Alucard Sprite Generator)
                 │
                 ▼
          Generated / Retrieved Sprite ($128 \times 128 \times 4$ RGBA)
                 │
        ┌────────┴────────┐
        ▼                 ▼
     [ AE ]            [ VAE ]
  Deterministic      Probabilistic ($z \sim \mathcal{N}$)
   Reconstruction     Similarity Search &
  / Outlier P95       Smooth Morphing (A → B)
```

### 1. Transformer (Semantic Planner)
- **Model**: `google/flan-t5-small` with rule-enhanced NLP fallback.
- **Job**: Translates natural language prompts into structured JSON specs (genre, character roles, environment themes, hazards, enemy roster, asset tags).

### 2. Autoencoder (AE 280K - Deterministic Baseline)
- **Checkpoint**: `models/280k dataset model/AE_280K_best.keras`
- **Fidelity**: Median MSE `0.000782`, PSNR `28.52 dB`, SSIM `0.9248`.
- **Application**: Deterministic reconstruction & **Quality Control** (surfacing the top 5% highest-error assets exceeding P95 MSE threshold `0.001313`).

### 3. Variational Autoencoder (VAE 280K - Probabilistic Representation)
- **Checkpoints**: `models/280k model VAE (VAE v2)/VAE_280K_Outputs/` (`encoder_280k_final.keras`, `decoder_280k_final.keras`).
- **Latent Prior**: 256-dimensional Gaussian space ($\beta=0.001$).
- **Active Search Index**: **25,000 Sprites** (`latent_gallery_index_25k.npy`), serving distance matching in **< 1ms**.
- **Applications**: Sub-millisecond vector similarity search, 2D PCA clustering with dynamic keyword extraction, and continuous real-sprite morphing ($A \to B$).

---

## 📊 Cross-Dataset Benchmark Summary

| Benchmark Dimension | 1. Alucard Dataset (Visual Sprites) | 2. ViGGO Dataset (Dialogue NLP) | 3. PICO-8 Dataset (Retro Games) |
| :--- | :--- | :--- | :--- |
| **Data Modality** | 2D Pixel-Art RGBA Images | Natural Language Dialogue Text | Lua Game Code + Metadata |
| **Total Population** | **282,511** RGBA Sprites | **6,900** Samples | **10,967** Game Cartridges |
| **Primary Category** | Characters & Heroes (**65.19%**) | `inform` Intent (**43.48%**) | Items & Casual Games (**52.52%**) |
| **Majority Baseline** | **65.19%** (*Character*) | **29.92%** (*inform*) / **27.32%** (*action-adventure*) | **52.51%** (*Items & Casual*) |
| **KNN Category Accuracy** | **84.40%** | **61.31%** (Intent) / **98.11%** (Genre) | **59.25%** |
| **Net Accuracy Lift** | **+19.21 percentage points** | **+31.39%** (Intent) / **+70.79%** (Genre) | **+6.75 percentage points** |

---

## 💻 Tech Stack & Active Endpoints

- **Backend**: Python 3, FastAPI ([backend/app.py](file:///c:/Users/manis/OneDrive/Desktop/Prompt_to_game_asset_generator/backend/app.py)), TensorFlow/Keras, NumPy, Scikit-learn, HuggingFace `datasets`.
- **Frontend**: React + Vite ([gui/src/App.jsx](file:///c:/Users/manis/OneDrive/Desktop/Prompt_to_game_asset_generator/gui/src/App.jsx)).
- **Status Endpoint**: `GET /status` -> Serves active status of AE, VAE, and Transformer models.
- **Search Endpoint**: `POST /vae/search_similar` -> Searches 25,000 pre-indexed VAE latent vectors in **< 1ms**.
