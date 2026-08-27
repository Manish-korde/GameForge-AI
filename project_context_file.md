# GameForge AI - Master Project Context & Knowledge Base 🎮🤖

This file serves as the definitive project context and architectural knowledge base for **GameForge AI**. It preserves core concepts, model roles, implementation details, environment setups, and critical mathematical resolutions discovered during development. 

**READ THIS FILE** upon starting any new session, resetting the chat, or making architectural decisions to avoid breaking the system structure or re-introducing solved bugs.

---

## 🧠 1. Project Overview & Core Philosophy
**GameForge AI** is a multi-model Generative AI pipeline designed to automate 2D game asset creation (sprites, textures, concepts, and environments). 

### The Multi-Model Strategy
Each model in the pipeline serves a unique, non-overlapping task. The models are sequential and complementary, not competing alternatives.
* **Transformer**: Translates natural language prompts into structured game specification sheets (character/enemy/weapon attributes, NPC dialogue/description text).
* **Autoencoder (AE)**: Baseline model. Compressed representation and high-fidelity reconstruction. Satisfied with performance on the full dataset, so it was trained on the full **280,000 sprites** dataset. Used for high-precision reconstruction and data cleaning (outlier detection via reconstruction error).
* **Variational Autoencoder (VAE)**: Learns a smooth, continuous latent space to generate variants of existing sprites and handle interpolation. **Currently in the 10K subset research phase.** It will be trained on the full 280K dataset once validated.
* **GAN (Generative Adversarial Network)**: Sharpens decoder outputs, recovering high-frequency visual details (like hard outlines and corners) that AE/VAE pixel-wise decoders naturally smooth away.
* **Diffusion Model**: The creative text-to-sprite generation core. Conceptually runs inside the VAE's compressed latent space to make inference computationally feasible (Latent Diffusion architecture, *Rombach et al., 2022*).

---

## 🎛️ 2. Training Workflow & Environment Constraints

> [!IMPORTANT]
> **LOCAL MACHINE IS NOT FOR TRAINING.** 
> The local machine and environment (`gpu_env`) are used **strictly for local model serving, evaluation, visualization, and comparison/notebook running purposes.** Do not run training pipelines locally.

### The Kaggle Training Cycle
1. **Notebook Generation:** We write and optimize model training notebooks locally, ensuring they are structured for compatibility with Kaggle (handling TPU/GPU selection, kaggle input paths, etc.).
2. **Kaggle Training Execution:** Notebooks are uploaded to Kaggle and run on Kaggle's cloud environment.
3. **Artifact Download:** The Kaggle training run exports a zip folder consisting of all results (trained models `.keras` / `.h5`, training history `.json`, plots, and evaluation configurations).
4. **Local Integration:** The exported zip files are downloaded and extracted into the local project structure under the `models/` and `evaluation/` directories for local comparison, visualization in Jupyter, and serving via the GUI.

---

## 🛠️ 3. Current Implementation & Development Status

### Phase 1: Autoencoder (AE) [COMPLETED & STABLE]
* **Dataset:** Trained on the full **280,000 Alucard sprites** dataset.
* **Performance:** High reconstruction fidelity (MSE ~0.0016, PSNR ~28.5 dB, SSIM ~0.92).
* **Integration:** Integrated into the FastAPI backend (`backend/app.py`). It loads `models/280k dataset model/AE_280K_best.keras` and serves the `/reconstruct` endpoint (providing MSE, MAE, PSNR, SSIM, and exact pixel match metrics).
* **Usage:** Serves as the high-fidelity reconstruction baseline and is used as a dataset data-cleaning tool.

### Phase 2: Variational Autoencoder (VAE) [ACTIVE EXPERIMENTATION]
* **Dataset:** Currently trained and evaluated on a **10K subset** of the Alucard dataset (for speed and rapid iteration).
* **Models/Weights:** VAE v2 (a $\beta$-VAE with $\beta=0.001$) is stored in `models/10k dataset model` / `models/10k_VAE`.
* **Notebooks:** 
  * `Notebook/experimentaion/10K_AE_vs_VAE_Comparison.ipynb`: Performs the evaluation of standard AE vs VAE v1 (KL Overloaded) vs VAE v2 ($\beta$-VAE).
  * `Notebook/280K_VAE_Training_Pipeline.ipynb`: The training notebook for the full 280K dataset. It has been fully updated to match the VAE v2 architecture ($\beta = 0.001$), save the final weights and full `.keras` models, generate the prior-scaled generalization evaluation, save all output plots (`loss_plot.png`, `variants_comparison_280k.png`, etc.) to `/kaggle/working/VAE_280K`, and bundle everything into a ZIP file for download.
* **CRITICAL CONSTRAINT:** **Do NOT integrate this VAE v2 model into the backend/app.py yet.** This model is strictly a research iteration on the 10K subset. Integration into the production pipeline will only occur after training a stronger VAE version on the full 280,000 dataset on Kaggle.

### Phase 3: GAN-Enhanced VAE (VAE-GAN) [ACTIVE EXPERIMENTATION]
* **Dataset:** 10K subset of the Alucard dataset.
* **Architecture:** Uses the exact continuous Beta-VAE (v2) architecture for the Generator (to preserve the `z_mean + scale * epsilon` continuous variant generation capability) and adds a PatchGAN Discriminator.
* **Notebooks:** 
  * `Notebook/experimentaion/10k_VAE/VAE_v3_10K_GAN_Enhanced.ipynb`: Implements the custom `train_step` computing Reconstruction MSE loss, KL Divergence loss, and the Adversarial loss.


---

## 🧬 4. Key Scientific Discoveries & Bug Fixes

### A. The VAE Invariance Paradox (Local Neighborhood Variation)
During testing, the single-asset local neighborhood variation test (sampling $z = \mu + \text{scale} \cdot \sigma \cdot \epsilon$ around a sprite) generated identical outputs across all scale levels (0.0 to 2.5). 

* **The Doubt:** Did the model suffer from posterior/prior variance collapse, meaning $\beta$ was too low and needed to be tuned up?
* **The Diagnostic:** We ran a raw evaluation on the latent stats. `sigma` was found to be healthy (~0.23 for random noise, ~0.83 for zeros), ruling out collapse and proving $\beta = 0.001$ was fine.
* **The Paradox:** During training, the VAE decoder is forced to map samples from $z \sim N(\mu, \sigma^2)$ back to the original image. Therefore, the decoder explicitly learns to be **invariant** to noise of scale $\sigma$ (the reconstruction "invariance bubble"). Multiplying the perturbation by `z_sigma` keeps it inside the bubble, resulting in no visual change.
* **The Resolution:** To generate distinct semantic variants of an asset, we must scale the perturbation directly by the global prior standard deviation of 1.0, rather than the local posterior scale:
  $$\text{z\_variant} = \mu + (\text{scale} \cdot \epsilon)$$
  Testing this on the VAE v2 weights showed a **16.5x increase in MSE visual variation**, producing clean, distinct asset variants at `scale = 1.0` to `1.5` while keeping the global structure intact.

### B. Fixed-Direction Latent Exploration
* **The Bug:** In the local neighborhood sampling code, `epsilon = tf.random.normal(...)` was generated *inside* the loop over scales. This caused each scale step to explore a different random orthogonal direction in the 256-dimensional space rather than scaling a single axis of variation.
* **The Fix:** We modified the code in `Notebook/experimentaion/10K_AE_vs_VAE_Comparison.ipynb` (Cell #15 and Cell #17) to initialize the `epsilon` direction vector **once, outside the loop**, ensuring scaling works continuously along a single, fixed latent direction.

### C. Generalization & Quantitative Check
* **Validation Gating:** We added Cell #19 in `Notebook/experimentaion/10K_AE_vs_VAE_Comparison.ipynb` to run the prior-scaled variants on **12 different random sprites** (at `scale = 1.2`) to confirm that this local neighborhood exploration generalizes robustly.
* **Outlier Hypothesis Verification:** We added Cell #21 to evaluate both AE and VAE reconstructions on these same 12 sprites. 
  * **Spearman's Rank Correlation:** We calculated a correlation coefficient of **$\rho = 0.8112$ (p-value = 0.0014)** between AE and VAE reconstruction errors. This strong positive correlation mathematically **confirms the outlier hypothesis** (reconstruction error is a property of the sprite's intrinsic difficulty/outlierness rather than model noise), validating its use for dataset cleaning.
  * **MSE Comparison:** Across these 12 sprites:
    * Average VAE Reconstruction MSE: `0.010772`
    * Average VAE Variant MSE (Scale = 1.2): `0.008428`

---

## 📂 5. Project Directory & Environment Mapping

* **Root Directory:** `c:\Users\manis\OneDrive\Desktop\Prompt_to_game_asset_generator`
* **Python Executable (Jupyter/FastAPI comparison/viz env):** 
  `C:\Users\manis\AppData\Local\Programs\Python\Python311\python.exe` (Jupyter kernel name: `gpu_env`).
* **Active Codebase Parts:**
  * `backend/app.py`: FastAPI server running on port 8000 using Python 3.11 conda/jupyter environment `gpu_env`.
  * `gui/`: React/Vite development server (port 5173).
  * `config/`: JSON files containing dataset index splits (train, val, test indices).
  * `Notebook/experimentaion/`: Contains comparison and training notebooks.
