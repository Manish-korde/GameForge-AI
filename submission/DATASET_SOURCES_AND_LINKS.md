# 📦 GameForge AI: Official Dataset Sources, Links & Partitioning Guide
### Practical Submission Requirement #2: Dataset Sources (30 Marks)

This document provides complete, verifiable links and download instructions for all datasets utilized across the GameForge AI platform.

---

## 1. Primary Visual Sprite Corpus: Alucard 282K
- **Dataset Name:** Alucard 2D Pixel-Art Sprite Dataset (`alucard-sprites`)
- **Original Author & Maintainer:** Simone Margaritelli (`evilsocket`)
- **Total Scale:** **282,511 Unique RGBA Sprites** ($128 	imes 128 	imes 4$ pixels with full 8-bit alpha mask)
- **License:** Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (`CC-BY-NC-SA 4.0`)
- **Official Source Links:**
  - **Hugging Face Datasets Repository:** [https://huggingface.co/datasets/evilsocket/alucard-sprites](https://huggingface.co/datasets/evilsocket/alucard-sprites)
  - **GitHub Source & Pipeline:** [https://github.com/evilsocket/alucard](https://github.com/evilsocket/alucard)
- **Data Partitioning (Strict 90 / 5 / 5 Split):**
  - 🏋️ **Training Set (90%):** 254,260 images (used for parameter optimization of AE, VAE, and DCGAN)
  - 🔍 **Validation Set (5%):** 14,126 images (used for early stopping and hyperparameter tuning)
  - 🧪 **Held-Out Test Set (5%):** 14,125 images (used for unbiased metric evaluation & P95 outlier threshold calculation)
- **Active Search Gallery Index:**
  - Pre-computed 25,000 latent embeddings stored locally in `latent_gallery_index_25k.npy` (< 1.0 ms cosine distance query matching).

---

## 2. Kinematic Walk Cycle Corpus: Liberated Pixel Cup (LPC)
- **Dataset Name:** Liberated Pixel Cup (LPC) Character & Walk-Cycle Dataset
- **Curator & Origin:** OpenGameArt.org Community & Stephen Challener (Redshrike)
- **Scale:** Over 1,200 character sprite sheets with standard 8-frame walk cycles across 4 cardinal directions (South, North, East, West).
- **License:** Creative Commons Attribution-ShareAlike 3.0 (`CC-BY-SA 3.0`) / GNU GPL 3.0
- **Official Source Links:**
  - **OpenGameArt Official Collection:** [https://opengameart.org/content/lpc-collection](https://opengameart.org/content/lpc-collection)
  - **Kaggle Mirror:** [https://www.kaggle.com/datasets/nikhilpandey360/liberated-pixel-cup](https://www.kaggle.com/datasets/nikhilpandey360/liberated-pixel-cup)
- **Usage in GameForge AI:**
  - Powers Module 5 Multi-Pose Velocity Diffusion UNet (`ActionAndPoseConditionedUNet`).
  - Evaluated on a strictly held-out cohort of 40 unseen characters (`C_160` through `C_199`) across 5,760 total generated samples.

---

## 3. Semantic Dialogue & Game Concept Corpus: ViGGO
- **Dataset Name:** ViGGO (Video Game Dialogue Corpus)
- **Creators:** Juraska et al., University of California, Santa Cruz (UCSC)
- **Scale:** 6,900 conversational game design descriptions and functional dialogue acts.
- **License:** Creative Commons Attribution 4.0 International (`CC-BY 4.0`)
- **Official Source Links:**
  - **Hugging Face Datasets Repository:** [https://huggingface.co/datasets/GEM/viggo](https://huggingface.co/datasets/GEM/viggo)
  - **ACL Anthology Paper:** [https://aclanthology.org/W19-8623/](https://aclanthology.org/W19-8623/)
- **Usage in GameForge AI:**
  - Fine-tuning dataset for Module 4 Transformer Sequence-to-Sequence Semantic Concept Planner.

---

## 4. Quick Download & Extraction Scripts

### Python Download via Hugging Face Hub:
```python
from datasets import load_dataset

# 1. Download Alucard Sprites
dataset = load_dataset("evilsocket/alucard-sprites", split="train")
print(f"Loaded {len(dataset)} sprites successfully.")

# 2. Download ViGGO Semantic Dialogue Dataset
viggo_data = load_dataset("GEM/viggo")
print(f"Loaded ViGGO train samples: {len(viggo_data['train'])}")
```