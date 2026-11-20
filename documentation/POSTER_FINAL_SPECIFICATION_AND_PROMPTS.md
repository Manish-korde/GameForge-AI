# GameForge AI: Final Academic Research Poster Specification & Asset Guide

**Project Title:** GameForge AI: Intelligent Multimodal Game Asset & Character Action Synthesis  
**Institution:** MIT Academy of Engineering, Pune (MITAOE) — Department of Computer Engineering  
**Project Guide:** Dr. Savita Mane  
**Presenters:**  
- Samriddhi Das (PRN: 202301040276)  
- Manish Korde (PRN: 202301040206)  
- Swayam Pagare (PRN: 202301040185)  

---

## 1. Poster Visual Hierarchy & Dimensions

| Layout Property | Specification |
| :--- | :--- |
| **Dimensions / Aspect** | Standard Academic Conference Poster: **36" $\times$ 48" Portrait** (3:4 ratio) or **48" $\times$ 36" Landscape** |
| **Grid Architecture** | Top full-width header banner, 3-column asymmetric body (Left: 28%, Center: 36%, Right: 36%), bottom benchmark tables, split footer |
| **Color Palette** | **Deep Royal Navy:** `#0B2545` / `#134074` (Header, card containers, table headers)<br>**Vibrant Sky Cyan:** `#0284C7` / `#38BDF8` (Badges, accents, active flow arrows)<br>**Dark Slate Text:** `#0F172A` on `#F8FAFC` light containers<br>**Pure White:** `#FFFFFF` on dark navy cards |
| **Typography** | Modern Clean Sans-Serif (`Inter` / `Roboto`). Title: 32–36pt Bold; Section Headers: 18–20pt Bold; Body Text: 12–14pt Regular; Captions: 11pt Regular Italic |

---

## 2. Poster Section Breakdown & Exact Text

### Top Header Banner (Full Width)
- **Top Left:** MIT Academy of Engineering (MITAOE) crest & "Department of Computer Engineering (Affiliated to SPPU)"
- **Center:**
  - **MAIN TITLE (34pt Bold):** `GameForge AI: Intelligent Multimodal Game Asset & Character Action Synthesis`
  - **SUBTITLE (16pt Medium):** `A Multi-Model Generative Framework for Prompt-Driven 2D Sprite Synthesis, Latent Verification & Kinematic Walk Cycles`
- **Right:** Team Names & Guide:
  - *Students:* Samriddhi Das • Manish Korde • Swayam Pagare
  - *Project Guide:* Dr. Savita Mane

---

### Column 1 (Left): Problem, Dataset & System Highlights

#### Panel 1.1: Project Statement & Motivation
- **The Bottleneck:** 2D indie video game development is bottlenecked by the manual creation of coherent, transparent, multi-frame character sprite sheets. A single animated character requires 8–16 hand-drawn frames per direction.
- **Why Existing General AI Fails:** General text-to-image models (Stable Diffusion, Midjourney) fail on discrete pixel art: they lack transparent background generation, blur sharp pixel boundaries, and cannot generate temporally coherent walking cycles.
- **The GameForge AI Solution:** A 4-tier generative framework that unites natural language planning, fast character synthesis, automated quality assurance, and temporal walk-cycle diffusion.

#### Panel 1.2: Multimodal Dataset Corpus
- **Scale:** 280,000+ Raw Sprites | 46,000 Audited Front-Facing Character Sprites | 1,800 Labeled Action Frames.
- **Sources:** Hugging Face (`evilsocket/alucard-sprites`, `carlosuperb/lpc-action-pixel-art-diffusion`) and OpenGameArt.
- **Preprocessing Pipeline:** $64 \times 64$ RGBA, 4-channel normalized $[-1.0, 1.0]$, 4-corner transparency bounding box verification, and canonical orientation curation.

#### Panel 1.3: Overall Ranking (Benchmark Highlights)
1. **Best Latent Representation:** Variational AE (**84.40%** k-NN accuracy, **256-D** smooth continuous manifold)
2. **Best Kinematic Motion:** Multi-Pose Velocity Diffusion (**0.812 Limb IoU**, **91.4%** cyclic consistency)
3. **Best Semantic Adherence:** Flan-T5 Planner (**100.0%** JSON validity, **97.5%** theme accuracy)
4. **Best Anomaly Detection:** Convolutional AE (**35.36 dB** PSNR, **0.961** SSIM, **94.2%** outlier accuracy)
5. **Fastest Baseline:** DCGAN (**4.8 ms** single-pass forward)

---

### Column 2 & 3 (Center & Right): The 4-Tier Generative Architecture

#### Section A: End-to-End System Flowchart
```
[User Text Prompt: "Armored Frost Paladin with Warhammer"]
       │
       ▼
[Tier 1: Flan-T5 Transformer Planner] ──► Structured JSON Spec (Theme: Frost, Class: Paladin, Palette: Cyan/Silver)
       │
       ▼
[Tier 2: DCGAN Character Generator]  ──► Synthesizes 64x64 RGBA Static Character Sprite
       │
       ▼
[Tier 3: Autoencoder Quality Gate]   ──► Reconstruction Error Test: MSE < 0.0013 ? [PASS] : [REJECT/REGENERATE]
       │
       ▼
[Tier 4: Velocity Diffusion U-Net]   ──► Conditioned on Static Sprite + Action Token ──► Reverse DDIM Sampling
       │
       ▼
[Game Engine Export: 512x64 Sprite Sheet] (Idle + Complete 8-Frame Kinematic Walk Cycle)
```

---

#### Section B: The Four Cooperative Models

##### Model 1: Flan-T5 Natural Language Planner
- **Architecture:** 60.5M-parameter Encoder-Decoder Sequence-to-Sequence Transformer.
- **Function:** Translates conversational prompts into structured JSON attribute vectors (class, armor, weapon, color palette).
- **Metric:** $100.0\%$ schema adherence, $97.5\%$ theme classification accuracy.

##### Model 2: DCGAN Character Synthesis Engine (LOCKED FINAL)
- **Architecture:** 5.40M Total Parameters (Generator: 2,633,732 params | Discriminator: 2,768,449 params), $z \in \mathbb{R}^{128}$.
- **Dataset:** 46,000 Audited Front-Facing Alucard Sprites (50 Epochs).
- **Key Feature:** High-speed latent synthesis capturing diverse fantasy archetypes (capes, broadswords, shields, and varied color themes).
- **Featured Visual:** Master 100-Character Showcase Grid (`final_master_showcase_100_characters.png`).

##### Model 3: Convolutional Autoencoder & VAE Quality Gate
- **Architecture:** Convolutional Encoder-Decoder (3.24M / 13.68M params) with 256-D continuous bottleneck.
- **Function:** Computes reconstruction error manifolds; automatically flags and discards noisy, malformed, or corrupt generated sprites before animation.
- **Metric:** Mean PSNR: $35.36\text{ dB}$, SSIM: $0.961$, Anomaly Threshold: $1.31 \times 10^{-3}$.

##### Model 4: Multi-Pose Velocity Diffusion U-Net (Animation Engine)
- **Architecture:** Action & Pose-Conditioned U-Net (1.98M params) with Velocity ($v$-prediction) parameterization and reverse DDIM sampling.
- **Function:** Takes the single static character sprite from Tier 2/3 and synthesizes a complete 8-frame walking cycle (F1 to F8).
- **Metric:** Limb Stride IoU: $0.812$ across 40 unseen test identities; $24.2\text{ ms}$ step latency on consumer CPU.
- **Featured Visual:** 8-Frame Kinematic Walk Cycle Filmstrip (`diffusion_generated_actions_showcase.png`).

---

### Column Bottom: Standardized Comparative Performance Tables

#### Table 1: Planning and Quality Assurance Tier
| Model | Role | Parameters | Primary Metric | Result |
| :--- | :--- | :---: | :--- | :---: |
| **Flan-T5 Planner** | Prompt-to-JSON Specification | 60.5M | JSON Schema Validity | **100.0%** (0.522 Val Loss) |
| **Convolutional AE** | Anomaly Gating & Noise Rejection | 3.24M | Reconstruction PSNR / SSIM | **35.36 dB / 0.961** |
| **Variational Autoencoder**| Latent Morphing & Interpolation | 13.68M | Latent k-NN Accuracy | **84.40%** (26.59 dB PSNR) |

#### Table 2: Generative Synthesis & Animation Tier
| Metric | Tier 2: DCGAN Character Generator | Tier 4: Velocity Diffusion Animation U-Net |
| :--- | :---: | :---: |
| **Target Task** | Static Character Prototyping | Reference-Conditioned 8-Frame Walk Cycle |
| **Dataset Scale** | 46,000 Audited Front Sprites | 1,800 Multi-Pose Character Pairs |
| **Parameter Count** | 5.40M | 1.98M |
| **Inference Latency** | **4.8 ms** (Instant Forward Pass) | **24.2 ms / step** (Reverse DDIM on CPU) |
| **Alpha Purity** | **94.8%** Transparent Background | **96.1%** Transparent Background |
| **Evaluation Score** | $L_1$ Pairwise Diversity: **12.15** | Limb Stride IoU: **0.812** (Cyclic Coherent) |

---

## 3. High-Resolution Poster Image Assets

| Visual Asset | Purpose on Poster | Path |
| :--- | :--- | :--- |
| **100-Character DCGAN Master Showcase** | Demonstrates character variety and fantasy archetypes in Tier 2 | [`final_master_showcase_100_characters.png`](file:///C:/Users/manis/Downloads/dcgan_frontfacing_46k_50ep_results/samples/final_master_showcase_100_characters.png) |
| **Diffusion 8-Frame Walk Cycle** | Shows kinematic motion from static reference in Tier 4 | [`diffusion_generated_actions_showcase.png`](file:///c:/Users/manis/OneDrive/Desktop/Prompt_to_game_asset_generator/research_paper_assets/06_sprite_deformation_and_pose_animation/14_historical_50ep_and_150ep_baseline/diffusion_generated_actions_showcase.png) |
| **Full Pipeline Showcase** | End-to-end prompt $\to$ static sprite $\to$ animated sheet | [`gameforge_diffusion_module4_empirical_showcase.png`](file:///c:/Users/manis/OneDrive/Desktop/Prompt_to_game_asset_generator/research_paper_assets/06_sprite_deformation_and_pose_animation/14_historical_50ep_and_150ep_baseline/gameforge_diffusion_module4_empirical_showcase.png) |
| **Autoencoder Quality Diagnostics** | Illustrates low error on real sprites vs high anomaly score on corrupt ones | [`gameforge_academic_poster_mockup.html`](file:///c:/Users/manis/OneDrive/Desktop/Prompt_to_game_asset_generator/research_paper_assets/gameforge_academic_poster_mockup.html) |

---

## 4. Final Poster Conclusion & Team Card

### Conclusion (Matched to Reference Poster Format - VAE Superiority)
- **• VAE provides superior latent representation & stability** — 84.40% k-NN latent accuracy & smooth 256-D continuous manifold eliminate adversarial mode collapse and enable seamless SLERP character morphing.
- **• Flan-T5 is superior planner** — 100.0% JSON schema validity & 97.5% semantic theme accuracy bridge natural language prompts directly to deterministic game asset specifications.
- **• Velocity Diffusion sets the motion bar** — 0.812 Limb IoU & 91.4% cyclic consistency achieve true 8-frame walk animation from a single 2D reference without identity drift.
- **• Integrated framework works best** — Combining Flan-T5 (planning), VAE (latent morphing & representation), Conv AE (QA gating), and Diffusion (animation) creates an end-to-end production framework.

### Team Card (Bottom Right)
- **Project Guide:** Prof. Dr. Savita Mane
- **Department:** Department of Computer Engineering, MITAOE, Pune
- **Group Members:**
  1. Samriddhi Das — 202301040276
  2. Manish Korde — 202301040206
  3. Swayam Pagare — 202301040185


