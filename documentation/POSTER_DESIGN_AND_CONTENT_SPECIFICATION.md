# GameForge AI: Academic Research Poster Specification

**Title:** GameForge AI: Intelligent Multimodal Game Asset & Character Action Synthesis  
**Institution:** MIT Academy of Engineering, Pune — Department of Computer Engineering  
**Project Guide:** Dr. Savita Mane  
**Project Group:**  
- Samriddhi Das (PRN: 202301040276)  
- Manish Korde (PRN: 202301040206)  
- Swayam Pagare (PRN: 202301040185)  

---

## 1. Global Layout and Visual Hierarchy

| Property | Specification |
| :--- | :--- |
| **Dimensions / Aspect** | Standard Academic Conference Poster (36" $\times$ 48" Portrait / 3:4 aspect ratio) |
| **Grid Structure** | Top full-width header banner, 3-column asymmetric layout (Col 1: 28%, Col 2: 36%, Col 3: 36%), multi-column bottom comparative section, and split footer |
| **Color Palette** | **Primary Dark Blue:** `#0F2042` / `#1E3A8A` (Header, left cards, table headers)<br>**Vibrant Cyan / Accent:** `#0284C7` / `#06B6D4` / `#38BDF8` (Team card, badges, borders)<br>**Light Cyan Tint:** `#F0F9FF` / `#E0F2FE` (Model card backgrounds, checklist container)<br>**Poster Background:** Clean White `#FFFFFF` with Slate `#F8FAFC` borders<br>**Text Colors:** High-contrast pure white on dark blue; dark slate `#0F172A` on light cyan |
| **Typography** | Strictly sans-serif (Inter, Roboto, Arial). Title: 32–36pt Bold; Section Headers: 18–20pt Bold; Body Text: 12–14pt Regular; Captions: 11pt |

---

## 2. Header Banner (Full Width)

* **Left Alignment:**
  * **Logo:** MIT Academy of Engineering (MITAOE) crest / logo
  * **Institutional Text:** "MIT Academy of Engineering, Pune (An Autonomous Institute Affiliated to Savitribai Phule Pune University)"
* **Center / Right Alignment:**
  * **Main Title:** `GameForge AI: Intelligent Multimodal Game Asset & Character Action Synthesis`
  * **Subtitle:** `A Multi-Model Generative Framework for Prompt-Driven 2D Sprite Synthesis, Latent Verification, and Kinematic Walk Animation`
* **Styling:** Solid deep royal blue (`#0A192F` to `#1E3A8A`) background, crisp white typography.

---

## 3. Left Column: Context, Dataset & System Highlights (Width: ~28%)

### Panel 1: Project Statement (Dark Blue Card, White Text)
* "2D video game development is fundamentally bottlenecked by the manual creation of coherent, transparent, multi-frame character sprite sheets."
* "Traditional procedural tools lack natural language semantic control, while general diffusion models fail on pixel-grid preservation, background transparency, and multi-pose kinematic walking cycles."
* "GameForge AI introduces a 4-tier generative pipeline integrating Transformer planning, DCGAN character synthesis, VAE quality assurance, and Action/Pose Velocity Diffusion."

### Panel 2: Dataset Corpus Description (Dark Blue Card, White Text)
* **Name:** GameForge Multimodal Asset Corpus (Alucard Sprites + Liberated Pixel Cup LPC + FLP).
* **Source:** Hugging Face (`evilsocket/alucard-sprites`, `carlosuperb/lpc`) & OpenGameArt.
* **Total Scale:** 280,000+ Raw Sprites | 169,094 Processed Character Sprites | 1,800 Audited Multi-Pose Frames.
* **Component Splits:**
  * **Transformer Planner (M1):** 10,000 Prompt-to-JSON Game Asset Specification pairs.
  * **DCGAN Characters (M2):** 169,094 Curated Transparent Character Sprites (50 Epochs).
  * **Autoencoder & VAE (M3):** 280,000 sprites (80% Train, 10% Val, 10% Test).
  * **Multi-Pose Diffusion (M4):** 160 Training Identities (1,440 pairs), 40 Unseen Validation Identities (360 pairs).
* **Preprocessing Pipeline:** $64 \times 64$ RGBA, 4-corner transparency bounding-box filtering, Tanh normalization $[-1.0, 1.0]$. Avoids background artifact memorization.

### Panel 3: Ethical, Style & Practical Considerations (Light Cyan Card, Dark Text)
* **Artistic Attribution & Licensing:** Trained exclusively on public open-source gaming repositories adhering to CC-BY-SA 3.0 and GPL v3 licenses, preserving indie artist lineage.
* **Style Consistency & Quality Gating:** Mitigates the risk of distorted generative outputs through automated VAE reconstruction anomaly scoring before game-engine export.
* **Computational Accessibility:** Operates on consumer hardware: lightweight inference allows instant sprite generation on commodity CPUs without cloud GPU dependency.
* **Limitations:** Generative resolution bounded to $64 \times 64$ retro pixel art; full 8-directional diagonal walking cycles currently require south-cycle projection.

### Panel 4: Overall Ranking / System Highlights (Light Cyan Card, Dark Text)
1. **Best Kinematic Motion:** Multi-Pose Velocity Diffusion (Limb IoU: $0.812$, 8-Frame cyclic consistency).
2. **Best Generation Speed:** DCGAN ($4.8\text{ ms}$ per sprite single-pass inference).
3. **Best Semantic Control:** Flan-T5 Sequence Planner ($100.0\%$ JSON validity, $97.5\%$ theme adherence).
4. **Best Anomaly Detection:** Convolutional Autoencoder (PSNR: $35.36\text{ dB}$, P95 Error Threshold: $1.31 \times 10^{-3}$).

---

## 4. Center and Right Columns: Main Body (Width: ~72%)

### Section A: Key Objectives (Top Center, Spanning Two Columns, Light Cyan Card)
* ✔ **Multimodal Specification Planning:** Translate unstructured natural language game briefs into structured JSON design assets using sequence-to-sequence Transformers.
* ✔ **Novel Character Synthesis:** Synthesize diverse, unconditioned $64 \times 64$ RGBA character sprites using Deep Convolutional GANs.
* ✔ **Automated Quality Assurance:** Detect corrupted or non-pixel-art sprites using reconstruction error manifolds in Autoencoders and VAEs.
* ✔ **Kinematic Pose Animation:** Generate complete 8-frame walk cycles from a single static front-facing sprite using reverse DDIM velocity diffusion.

---

### Section B: Model Overview (2x2 Grid of 4 Specialized Models)

#### Model 1: Autoencoder & VAE (Quality Assurance & Latent Morphing)
* **Architecture:** Convolutional Encoder-Decoder (3.24M / 13.68M params) with 256-D continuous bottleneck.
* **Role:** Latent Compression, Anomaly Detection & SLERP Asset Morphing.
* **Mathematical Formula:**
  $$\mathcal{L}_{\text{VAE}} = \mathbb{E}_{q_\phi(z|x)}\left[\|x - \hat{x}\|^2\right] + D_{\text{KL}}\left(q_\phi(z|x) \,\|\, p(z)\right)$$
* **Sample Outputs & Diagnostics:** Reconstructs high-fidelity assets (Mean PSNR: $35.36\text{ dB}$, SSIM: $0.961$). Pristine sprites exhibit near-zero MSE ($< 8.04 \times 10^{-4}$); corrupt or broken sprites trigger high reconstruction residuals ($> 1.31 \times 10^{-3}$) for automatic rejection.

#### Model 2: DCGAN (169K Character Sprite Generator)
* **Architecture:** 4-Stage Nearest-Neighbor Upsampling Generator with LeakyReLU Discriminator (5.40M params), $z \in \mathbb{R}^{128}$.
* **Role:** Fast, Diverse Front-Facing Character Synthesis.
* **Mathematical Formula:**
  $$\min_D \frac{1}{2}\mathbb{E}_{x}\left[(D(x) - 0.9)^2\right] + \frac{1}{2}\mathbb{E}_{z}\left[D(G(z))^2\right]$$
  $$\min_G \frac{1}{2}\mathbb{E}_{z}\left[(D(G(z)) - 1)^2\right] + \lambda_{\text{FM}}\|\mathbb{E}f(x) - \mathbb{E}f(G(z))\|_1$$
* **Sample Outputs & Previous Results:** Trained on 169K character corpus for 50 epochs (Gen/Disc Loss: $0.5212 / 0.0091$). Produces diverse archetypes with Pairwise $L_1$ Diversity of $12.15$, Laplacian Sharpness of $64.37$, and Alpha Mask Purity of $94.80\%$.

#### Model 3: Transformer Planner (Flan-T5-Small Sequence-to-Sequence)
* **Architecture:** 60.5M-parameter Encoder-Decoder Sequence-to-Sequence Transformer.
* **Role:** Multimodal Concept-to-Specification Translation.
* **Mathematical Formula:**
  $$\mathcal{L}_{\text{seq}}(\theta) = -\sum_{t=1}^T \log P_\theta(y_t \mid y_{<t}, X_{\text{prompt}})$$
* **Sample Outputs & Metrics:** Translates free-form text ("armored frost paladin with heavy warhammer") into strictly validated JSON schema. Achieves $100.0\%$ schema validity, $97.5\%$ theme classification accuracy, and $0.5228$ validation loss.

#### Model 4: Multi-Pose Velocity Diffusion U-Net (Delivered Production Model)
* **Architecture:** Action & Pose-Conditioned U-Net ($1.98\text{M}$ params) with velocity $v$-prediction objective and reverse DDIM sampling.
* **Role:** Reference-Conditioned Walk Cycle & Kinematic Stance Synthesis.
* **Mathematical Formula:**
  $$v_t \equiv \sqrt{\bar{\alpha}_t}\epsilon - \sqrt{1 - \bar{\alpha}_t}x_0, \quad \mathcal{L}_v = \mathbb{E}_{t, x_0, \epsilon, x_{\text{ref}}, c_{\text{act}}, c_{\text{pose}}}\left[ \| v_t - v_\theta(x_t, t, x_{\text{ref}}, c_{\text{act}}, c_{\text{pose}}) \|^2 \right]$$
* **Sample Outputs & Metrics:** Generates Idle stance + full 8-frame South Walk cycle (F1–F8) from pure Gaussian noise. Evaluated on 40 unseen characters ($C_{160}-C_{199}$) with Limb IoU of $0.812$ and CPU step latency of $24.2\text{ ms}$.

---

### Section C: Model Architecture Flowchart & Technical Learnings

#### Visual Pipeline Architecture (Center-Left)
```
[User Text Prompt] 
       │
       ▼
[M1: Flan-T5 Planner] ──► Generates Structured JSON Design Asset (Theme, Class, Palette)
       │
       ▼
[M2: DCGAN Generator] ──► Synthesizes Novel 64x64 RGBA Static Character Sprite
       │
       ▼
[M3: Autoencoder Gate] ──► Computes Reconstruction MSE (Rejects Noise/Corrupted Sprites)
       │
       ▼
[M4: Velocity Diffusion] ──► Reverse DDIM Samples Complete 8-Frame Walk Animation Cycle
       │
       ▼
[GameForge AI Studio] ──► Exports Verified 512x64 Animated Sprite Sheet to Game Engine
```

#### Challenges & Learnings (Center-Right, Light Blue Box)
* **Velocity Parameterization Dominance:** Velocity $v$-prediction completely eliminated color drift and structural degradation compared to standard $\epsilon$-prediction in pixel-art diffusion.
* **Bilateral Symmetry Constraints:** Unconditional GANs suffer from anatomical mode collapse when trained on mixed profiles; strict front-facing viewpoint curation is required.
* **Unsupervised Quality Assurance:** Convolutional Autoencoder reconstruction MSE reliably identifies and gates visual corruption without requiring labeled negative datasets.
* **Reverse DDIM Efficiency:** Deterministic DDIM sampling generates publication-grade 8-frame walk animation cycles in under 1.5 seconds on CPU.
* **Channel Preservation:** End-to-end 4-channel RGBA modeling prevents color fringing around alpha silhouettes.

---

## 5. Bottom Section: Comparative Performance Metrics & Footer

### Performance Metrics — Standardized Comparative Protocol (50 Epochs, Batch Size 32)

#### Table 1: Verification & Planning Models (AE vs VAE vs Transformer)
| Evaluation Metric | Convolutional AE | Variational Autoencoder (VAE) | Flan-T5-Small Planner |
| :--- | :---: | :---: | :---: |
| **Model Role** | Outlier QA Gating | Latent Morphing / Anomaly | Prompt Planning |
| **Parameters** | 3.24M | 13.68M | 60.5M |
| **Primary Evaluation Loss** | Test MSE: $4.56 \times 10^{-4}$ | Test MSE: $3.17 \times 10^{-3}$ | Val Loss: $0.5228$ |
| **Reconstruction PSNR** | **35.36 dB** | 26.59 dB | N/A |
| **Structural Similarity (SSIM)** | **0.961** | 0.843 | N/A |
| **KNN Latent Accuracy** | N/A | **84.40%** | N/A |
| **Schema / Theme Validity** | P95 Thresh: $1.31 \times 10^{-3}$ | Precision@5: 0.628 | **100.0% Valid JSON** (97.5% Theme) |

#### Table 2: Generative Synthesis Models (DCGAN vs Velocity Diffusion)
| Synthesis Metric | DCGAN (169K Character Run) | Multi-Pose Velocity Diffusion U-Net |
| :--- | :---: | :---: |
| **Comparative Epochs** | 50 Epochs | 50 Epochs (Extended 150ep / E1 Multi-Pose) |
| **Terminal Loss** | Gen/Disc: $0.5212 / 0.0091$ | Val MSE: $1.06 \times 10^{-1}$ (Velocity MSE: $3.12 \times 10^{-2}$) |
| **Pairwise Diversity / IoU** | $L_1$ Diversity: **12.15** | Limb Stride IoU: **0.812** (40 Unseen Identities) |
| **Edge Sharpness / PSNR** | Laplacian Sharpness: **64.37** | PSNR: **27.84 dB** (SSIM: 0.862) |
| **Alpha Mask Purity** | **94.80%** Pure Silhouette | **96.10%** Pure Silhouette |
| **Generation Latency** | **4.8 ms** (Single-Pass Forward) | **24.2 ms / step** (Reverse DDIM on CPU) |
| **Output Capability** | Random Static Novel Characters | Static Idle + Full 8-Frame South Walk Cycle |
| **Trainable Parameters** | 5.40M | 1.98M (1,987,044 parameters) |

---

### Conclusion (Bottom Left, Light Gray Card)
* **Unified Pipeline:** GameForge AI proves that multimodal asset synthesis is most robust when decomposed into specialized, cooperative generative tiers rather than an unconstrained monolithic model.
* **Instant Character Prototyping:** DCGAN provides near-instantaneous character synthesis across diverse fantasy archetypes.
* **Coherent Temporal Animation:** Multi-Pose Velocity Diffusion achieves kinematic limb articulation and 8-frame loop consistency from a single reference image.
* **Quality Assurance Reliability:** Autoencoder reconstruction error manifolds provide automated, human-free quality gating before assets are deployed to game engines.

### Team & Academic Details (Bottom Right, Solid Cyan Card)
* **Project Guide:** Dr. Savita Mane
* **Department:** Department of Computer Engineering
* **Institution:** MIT Academy of Engineering, Pune
* **Project Group Members:**
  1. Samriddhi Das (PRN: 202301040276)
  2. Manish Korde (PRN: 202301040206)
  3. Swayam Pagare (PRN: 202301040185)

