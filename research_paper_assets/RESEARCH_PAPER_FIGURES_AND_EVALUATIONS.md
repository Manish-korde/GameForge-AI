# 📚 Research Paper Evaluation Assets: Complete Figure Catalog & Academic Guide

**Project:** GameForge AI (Prompt-to-Game-Asset Generator)  
**Authors:** Samriddhi Das (202301040276), Manish Korde (202301040206)  
**Guide:** Dr. Savita Mane | **Institute:** MIT Academy of Engineering, Pune  
**Scope:** Research Paper Visual Figures, Quantitative Tables, and Evaluation Artifacts  
**Model Focus:** Dedicated **4-Model Architecture** featuring our sole production GAN: **169K Character 50-Epoch DCGAN** (all experimental multi-run drafts excluded).

---

## 📑 Figure Directory Structure

All figures, contact sheets, and loss plots have been copied directly from their original sources and organized into:

```text
research_paper_assets/
├── 01_dataset_analysis/                # Dataset distribution & representative ground-truth sprites
├── 02_autoencoder_qa_outliers/         # AE reconstruction fidelity, loss curves & P95 outlier screening
├── 03_vae_latent_morphing/             # VAE continuous latent space, sampling & sprite morphing
├── 04_dcgan_169k_character_synthesis/  # DCGAN 169K 50-epoch results, SLERP, loss audit & palettes
└── 05_comparative_evaluations/         # Cross-model comparisons, retrieval benchmarks & KNN metrics
```

---

## 1. Section: Dataset Analysis & Ground Truth Distribution

### 1.1 Category Distribution
- **File:** `research_paper_assets/01_dataset_analysis/category_distribution.png`
- **Caption for Paper:** *Figure 1: Class frequency distribution across the 282,511 Alucard RGBA game sprite corpus. Characters constitute the primary majority class (65.19%), followed by Unknown/Miscellaneous (15.84%), Items (6.39%), Enemies (5.75%), Weapons (4.38%), Tiles (1.92%), Props (0.35%), and Effects (0.18%).*

### 1.2 Ground Truth Contact Sheets
Representative $128 \times 128 \times 4$ ground truth sprite sheets from the training corpus:
- `research_paper_assets/01_dataset_analysis/character.png` — *Ground truth character sprites (Knights, Mages, Warriors, Rogues).*
- `research_paper_assets/01_dataset_analysis/enemy_monster.png` — *Ground truth enemy and monster assets.*
- `research_paper_assets/01_dataset_analysis/item_consumable.png` — *Ground truth potions, scrolls, and collectible items.*
- `research_paper_assets/01_dataset_analysis/weapon.png` — *Ground truth swords, bows, staves, and shields.*
- `research_paper_assets/01_dataset_analysis/tile_environment.png` — *Ground truth environment tiles (dungeon bricks, terrain).*

---

## 2. Section: Autoencoder (AE) Reconstruction & Outlier Screening

### 2.1 Training Loss Curve
- **File:** `research_paper_assets/02_autoencoder_qa_outliers/ae_training_loss_curve.png`
- **Caption for Paper:** *Figure 2: Convolutional Autoencoder training and validation loss progression over 282,511 samples. Stable convergence achieves a final training MSE of 0.000782 and validation MSE of 0.000804.*

### 2.2 Reconstruction Fidelity
- **File:** `research_paper_assets/02_autoencoder_qa_outliers/ae_original_vs_reconstruction.png`
- **File:** `research_paper_assets/02_autoencoder_qa_outliers/ae_reconstruction_quality_grid.png`
- **Caption for Paper:** *Figure 3: Qualitative comparison of ground-truth input sprites (left) versus AE deterministic reconstructions (right). The unregularized bottleneck preserves high-frequency pixel boundaries and alpha transparency.*
- **Quantitative Benchmark:**
  - **Median MSE:** `0.000782`
  - **PSNR:** `28.52 dB`
  - **SSIM:** `0.9248`

### 2.3 Mathematically Grounded P95 QA Outlier Screening
- **File:** `research_paper_assets/02_autoencoder_qa_outliers/ae_p95_error_threshold_plot.png`
- **File:** `research_paper_assets/02_autoencoder_qa_outliers/ae_p95_outlier_reconstructions_contact_sheet.png`
- **Caption for Paper:** *Figure 4: Automated reconstruction-based quality assurance (QA). Distribution of per-sprite MSE across 5,000 test sprites with the empirical 95th-percentile error threshold (MSE = 0.001313). Sprites exceeding this threshold are automatically flagged for review, identifying corrupted or poorly formatted assets.*

---

## 3. Section: Variational Autoencoder (VAE) Latent Space & Morphing

### 3.1 Training Loss Curve
- **File:** `research_paper_assets/03_vae_latent_morphing/vae_training_loss_curve.png`
- **Caption for Paper:** *Figure 5: VAE joint loss convergence, balancing reconstruction MSE against the KL-divergence constraint ($D_{KL}(q(z|x) \parallel \mathcal{N}(0, I))$).*

### 3.2 Reconstruction vs. Gaussian Sampling
- **File:** `research_paper_assets/03_vae_latent_morphing/vae_original_vs_reconstruction.png` — *VAE reconstruction fidelity.*
- **File:** `research_paper_assets/03_vae_latent_morphing/vae_gaussian_sampling_generations.png` — *Unconditional generation by sampling pure latent noise $z \sim \mathcal{N}(0, I)$. Unlike standard AE (which decodes to static noise), VAE decodes into valid sprite silhouettes.*

### 3.3 Continuous Latent Morphing (Variants)
- **File:** `research_paper_assets/03_vae_latent_morphing/vae_sprite_variants_interpolation.png`
- **Caption for Paper:** *Figure 6: Continuous latent space traversal demonstrating smooth sprite style and archetype morphing without visual discontinuity.*

---

## 4. Section: DCGAN 169K Character Generative Synthesis (The Paper's Single GAN Model)

This is the **sole, definitive GAN model** presented in the research paper: a 50-epoch Deep Convolutional GAN trained on **169,420 clean character sprites** ($64 \times 64 \times 3$).

### 4.1 Adversarial Minimax Loss Audit
- **File:** `research_paper_assets/04_dcgan_169k_character_synthesis/adversarial_loss_audit.png`
- **Caption for Paper:** *Figure 7: 50-epoch adversarial loss audit of Generator ($G$) versus Discriminator ($D$). Generator loss stabilizes at 1.84 while Discriminator loss converges at 0.52 without mode collapse or gradient explosion.*

### 4.2 Longitudinal Training Evolution
- **File:** `research_paper_assets/04_dcgan_169k_character_synthesis/longitudinal_all_epochs_evolution.png`
- **Caption for Paper:** *Figure 8: Longitudinal visual evolution across all 50 training epochs. Early epochs (0–5) show rudimentary blobs; intermediate epochs (15–30) resolve torso, limbs, and headgear; final epochs (40–50) demonstrate crisp pixel art silhouettes, multi-tone shading, and anatomical consistency.*

### 4.3 64-Character Showcase & High-Res Game Assets
- **File:** `research_paper_assets/04_dcgan_169k_character_synthesis/showcase_64_characters.png` — *Figure 9: Grid of 64 novel character sprites generated unconditionally from random latent seeds $z \in \mathbb{R}^{100}$.*
- **File:** `research_paper_assets/04_dcgan_169k_character_synthesis/game_assets_showcase.png` — *Figure 10: Selected high-fidelity characters demonstrating distinct RPG archetypes (armored knights, sorcerers, beasts, rogues).*
- **File:** `research_paper_assets/04_dcgan_169k_character_synthesis/random_single_character_test.png` — *Single character inspection showing local contrast and silhouette definition.*

### 4.4 Latent Geodesic Spherical Interpolation (SLERP)
- **File:** `research_paper_assets/04_dcgan_169k_character_synthesis/slerp_knight_to_mage.png`
- **File:** `research_paper_assets/04_dcgan_169k_character_synthesis/slerp_rogue_to_paladin.png`
- **Caption for Paper:** *Figure 11: Spherical Linear Interpolation (SLERP) across DCGAN latent space along a constant-velocity geodesic trajectory. Smoothly transitions headgear, chest armor, and weaponry between distinct classes without semantic distortion.*

### 4.5 Quantization & Alpha Cutout Post-Processing
- **File:** `research_paper_assets/04_dcgan_169k_character_synthesis/raw_vs_quantized_comparison.png` — *Figure 12: Comparison between raw continuous Tanh output and post-processed discrete 16-color palette quantization.*
- **File:** `research_paper_assets/04_dcgan_169k_character_synthesis/transparent_cutouts_preview.png` — *Figure 13: Clean alpha transparency ($\alpha = 0$) cutouts ready for direct game engine injection.*

---

## 5. Section: Comparative Evaluations & Benchmark Metrics

### 5.1 AE vs. VAE Reconstruction & Objective Trade-Off
- **File:** `research_paper_assets/05_comparative_evaluations/ae_vs_vae_loss_comparison.png`
- **Caption for Paper:** *Figure 14: Objective function trade-off between deterministic AE and probabilistic VAE. AE achieves lower reconstruction MSE due to unconstrained latent space, whereas VAE trades reconstruction fidelity for a smooth, normal Gaussian prior ($z \sim \mathcal{N}(0, I)$) enabling generation and interpolation.*

### 5.2 Sub-Millisecond Vector Retrieval & Category Classification
- **File:** `research_paper_assets/05_comparative_evaluations/retrieval_comparison.png`
- **File:** `research_paper_assets/05_comparative_evaluations/retrieval_comparison_contact_sheet.png`
- **Caption for Paper:** *Figure 15: Top-k nearest-neighbor visual asset retrieval across the 25,000-sprite pre-indexed vector gallery ($< 1\text{ ms}$ latency).*
- **Quantitative Benchmark Table:**

| Metric | Evaluation Value | Academic Baseline | Lift / Significance |
| :--- | :---: | :---: | :---: |
| **KNN Category Classification Accuracy** | **84.40%** | 65.19% (Majority Class) | **+19.21%** lift |
| **Retrieval Precision@1** | **66.00%** | 20.00% (Random Chance) | **+46.00%** lift |
| **Retrieval Precision@5** | **62.80%** | 20.00% (Random Chance) | **+42.80%** lift |
| **Retrieval Mean Reciprocal Rank (MRR)** | **0.7512** | 0.4566 | Strong top-ranked relevance |
| **Vector Search Latency** | **< 1.0 ms** | Real-time threshold (<50ms) | Instantaneous in production GUI |

---

---

## 6. Section: Conditioned Action Generation via Diffusion (Module 4)

To bridge static novel character generation (from DCGAN) with animated multi-pose game mechanics, we implemented and evaluated the **Conditioned Pixel Diffusion Framework** trained on paired Liberated Pixel Cup (LPC) corpora (`carlosuperb/lpc-4view-pixel-art-diffusion` and `carlosuperb/lpc-action-pixel-art-diffusion`).

### 6.1 Action and Reference Corpus Exploration
- **File:** `research_paper_assets/06_sprite_deformation_and_pose_animation/lpc_action_dataset_preview.png` — *Figure 16: Multi-pose action sprite sheets ($576 \times 768$ RGBA) from the 50,000-row LPC action dataset containing walk, spellcast, thrust, slash, and hurt frame sequences.*
- **File:** `research_paper_assets/06_sprite_deformation_and_pose_animation/lpc_reference_dataset_preview.png` — *Figure 17: Canonical 4-view reference sprites ($128 \times 128$ RGBA).*
- **File:** `research_paper_assets/06_sprite_deformation_and_pose_animation/extracted_sprite_cells_grid.png` — *Figure 18: Uniform $9 \times 16$ grid decomposition yielding 31,000 isolated action cells for pose pairing.*

### 6.2 Conditional Diffusion UNet Architecture
- **Model:** `ConditionalDiffusionUNet` (1,972,675 trainable parameters).
- **Conditioning Mechanism:** Takes a 6-channel concatenation $[x_t, x_{\text{ref}}]$ combining the noisy target action sprite $x_t \in \mathbb{R}^{3 \times 64 \times 64}$ and canonical character reference $x_{\text{ref}} \in \mathbb{R}^{3 \times 64 \times 64}$.
- **Timestep Embedding:** 128-dimensional sinusoidal positional embeddings injected across multi-scale convolutional downsampling and upsampling skip connections.
- **Denoising Objective:** Trained on 10,000 conditioned action pairs across 50 strict epochs with dual-GPU acceleration using the standard $\epsilon$-prediction MSE loss:
  $$\mathcal{L}_{\text{diff}}(\theta) = \mathbb{E}_{t, x_0, \epsilon} \left[ \| \epsilon - \epsilon_\theta(x_t, t, x_{\text{ref}}) \|^2 \right]$$

### 6.3 Empirical Results & Convergence
- **File:** `research_paper_assets/06_sprite_deformation_and_pose_animation/diffusion_50ep_loss_curve.png` — *Figure 19: 50-Epoch Diffusion Loss Curve demonstrating smooth, monotonic convergence from $0.6096$ down to **0.03302**.*
- **File:** `research_paper_assets/06_sprite_deformation_and_pose_animation/diffusion_denoised_clean_showcase.png` — *Figure 20: Qualitative pose generation showcase: (Top) Canonical 4-view input reference sprites; (Bottom) Conditioned denoised action frames articulating directional poses and maintaining precise character color palettes.*
- **Historical Convergence Data:** `research_paper_assets/06_sprite_deformation_and_pose_animation/diffusion_50ep_history.csv` — Full epoch-by-epoch loss progression.

---

## 7. Model Summary Table for Research Paper Methodology

| Model Name | Training Corpus | Epochs / Steps | Primary Role in Paper | Key Academic Output Metric |
| :--- | :--- | :---: | :--- | :--- |
| **Transformer (Flan-T5)** | ViGGO Game Dialogue (6.9K) | Fine-tuned | Semantic Planning & Spec Sheet Structuring | Structured JSON schema generation |
| **Convolutional AE** | Alucard Corpus (282,511 RGBA) | 50 Epochs | Deterministic Reconstruction & QA Screening | Median MSE: **0.000782**, P95 Threshold: **0.001313** |
| **Probabilistic VAE** | Alucard Corpus (282,511 RGBA) | 50 Epochs | Continuous Latent Space & Geodesic Morphing | KNN Accuracy: **84.40%**, Precision@5: **62.80%** |
| **DCGAN (Characters)** | Alucard Characters (169,420 RGBA) | 50 Epochs | Novel Character & Monster Sprite Generation | Generator Loss: **1.84**, Discriminator Loss: **0.52** |
| **Conditional Diffusion UNet** | LPC 10,000 Paired Action Sprites | 50 Epochs | Character-Conditioned Action & Pose Synthesis | Denoising MSE Loss: **0.03302** (Convergence $0.61 \to 0.033$) |

*(Note: All other experimental multi-run drafts, 20K trials, and texture GAN attempts have been excluded to present a unified, publication-grade experimental architecture).*


