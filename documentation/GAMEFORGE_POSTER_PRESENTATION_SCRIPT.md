# GameForge AI: Academic Poster Defense Co-Presentation Script
**Target Duration:** 5–7 Minutes  
**Presenters:** Samriddhi Das & Manish Korde  
**Project:** GameForge AI — Intelligent Multimodal Game Asset & Character Action Synthesis  
**Institution:** MIT Academy of Engineering, Pune (MITAOE) — Department of Computer Engineering  
**Project Guide:** Dr. Savita Mane  

---

## Presentation Structure Overview

```
+---------------------------------------------------------------------------------------------------+
| TIME      | SPEAKER    | TOPIC COVERED                                 | POSTER PANEL TO POINT AT |
+---------------------------------------------------------------------------------------------------+
| 0:00-1:00 | Samriddhi  | Title, Hook & The Indie Game Asset Bottleneck | Header & Column 1 (Top)  |
| 1:00-2:15 | Manish     | 4-Tier Pipeline Architecture & Flan-T5 Planner| Center Flowchart & M1    |
| 2:15-3:30 | Samriddhi  | DCGAN Character Synthesis & VAE Quality Gate  | Model 2 & Model 3 Panels |
| 3:30-4:45 | Manish     | Velocity Diffusion & 8-Frame Walk Cycles      | Model 4 Panel & Filmstrip|
| 4:45-5:45 | Samriddhi  | Comparative Metrics & CPU Deployment          | Bottom Benchmark Tables  |
| 5:45-6:30 | Both       | Summary, Future Work & Opening for Questions  | Conclusion & Team Card   |
+---------------------------------------------------------------------------------------------------+
```

---

## Full Word-for-Word Co-Presentation Script

### [0:00 – 1:00] Part 1: Introduction & The Core Problem
*(Samriddhi begins with confidence and gestures to the poster header and left column)*

**Samriddhi:**  
"Respected evaluators and professors, good morning. I am **Samriddhi Das**, and along with my co-researcher **Manish Korde**, we are proud to present our final capstone project: **GameForge AI: Intelligent Multimodal Game Asset and Character Action Synthesis**, developed under the guidance of **Dr. Savita Mane** at the Department of Computer Engineering, MIT Academy of Engineering.

If you have ever talked to an indie game developer, you know that **2D asset creation is the single biggest bottleneck in game production**. To bring a single character to life, a pixel artist must manually illustrate between 8 to 16 individual frames per direction, ensuring consistent palettes, pixel grids, and transparent backgrounds.

Now, you might ask: *Why not just use general text-to-image AI like Stable Diffusion or Midjourney?*  
We tested them, and they fail on three critical game development requirements:
1. They cannot output **native transparent alpha channels**.
2. They destroy **discrete pixel-grid alignment** with blurry analog gradients.
3. Most importantly, they **cannot maintain temporal coherence across walking animation frames**.

To solve this, GameForge AI introduces a modular, 4-tier cooperative generative system. I will now hand over to Manish to walk you through our system architecture."

---

### [1:00 – 2:15] Part 2: System Architecture & The Transformer Planner
*(Manish steps forward, gestures to the central flowchart and Tier 1)*

**Manish:**  
"Thank you, Samriddhi. As you can see in our central pipeline flowchart:  
*[Manish points to Section A: End-to-End System Flowchart]*  
Rather than relying on an unconstrained, monolithic model, GameForge AI decomposes game asset synthesis into **four specialized, cooperating tiers**:

1. **Tier 1: Specification Planning (Flan-T5-Small):**  
   Game developers don't think in terms of mathematical seed vectors; they think in creative briefs, like *'armored frost paladin with a heavy warhammer'*.  
   We fine-tuned a 60.5-million parameter sequence-to-sequence Transformer on 10,000 prompt-specification pairs. It translates free-form natural language into a strictly validated JSON asset schema defining the character's class, equipment, and chromatic palette.  
   It achieved **100.0% JSON schema adherence** and **97.5% theme accuracy**, ensuring downstream models receive unambiguous generation parameters.

Once the specification is locked, it moves to Tier 2 for visual character generation. Samriddhi will explain how our character synthesis and quality gating operate."

---

### [2:15 – 3:30] Part 3: DCGAN Character Synthesis & Quality Gating
*(Samriddhi gestures to Model 2 and the 100-Character Showcase Grid)*

**Samriddhi:**  
"In **Tier 2**, we tackle rapid character prototyping using an audited **Deep Convolutional GAN (DCGAN)** with 5.4 million parameters:  
*[Samriddhi points to the 100-character master showcase]*  
Trained on a curated corpus of 46,000 front-facing character sprites for 50 epochs, our DCGAN maps a 128-dimensional latent space directly into $64 \times 64 \times 4$ RGBA sprites in just **4.8 milliseconds**—over 50 times faster than diffusion models. As you can see on our showcase grid, it successfully captures distinct fantasy archetypes, varying hair colors, shields, capes, and armor textures.

However, any generative practitioner knows that GANs can occasionally produce malformed or noisy artifacts. In a production game pipeline, broken sprites cannot be passed to the animator.

*[Samriddhi gestures to Model 3: Autoencoder Quality Gate]*  
This brings us to **Tier 3: Automated Quality Assurance**.  
We trained a Convolutional Autoencoder with a continuous latent bottleneck to act as an automated gatekeeper. Because the autoencoder was trained strictly on high-fidelity sprites, pristine generated characters reconstruct with near-zero error—achieving a **PSNR of 35.36 dB**.  
If the DCGAN outputs a corrupted or noisy sprite, the reconstruction error exceeds our 95th percentile threshold of $1.31 \times 10^{-3}$, and the system **automatically rejects and re-samples it without requiring human intervention**.

Once a character passes quality gating, it moves to our flagship animation engine. Manish will explain how we animate it."

---

### [3:30 – 4:45] Part 4: Kinematic Velocity Diffusion & Walk Animation
*(Manish points to Model 4 and the 8-frame walk filmstrip)*

**Manish:**  
"Now, the most technically challenging problem in game asset generation: **How do you take a single static character and turn it into a walking animation?**

*[Manish points to Model 4: Multi-Pose Velocity Diffusion]*  
In **Tier 4**, we developed a **Multi-Pose Velocity Diffusion U-Net** with 1.98 million parameters.  
Instead of predicting standard noise $\epsilon$, we formulated the training objective using **Velocity $v$-prediction parameterization**:
$$v_t \equiv \sqrt{\bar{\alpha}_t}\epsilon - \sqrt{1 - \bar{\alpha}_t}x_0$$
This formulation completely eliminated the color-drift and structural blurring that plagues traditional pixel-art diffusion.

The model is conditioned on two inputs:
1. The static reference sprite from Tier 2.
2. An action token specifying the animation frame (Frame 1 through Frame 8 of the South Walk cycle).

Using reverse **DDIM deterministic sampling**, our diffusion engine generates the full 8-frame kinematic walk cycle in under **1.5 seconds on a consumer CPU**.  
We evaluated this on **40 completely unseen character identities**, achieving a **Limb Stride IoU of 0.812**, proving that the limbs articulate realistically while preserving the exact hair, clothes, and weapon identity of the character."

---

### [4:45 – 5:45] Part 5: Empirical Benchmarks & Deployment
*(Samriddhi gestures to the comparative tables at the bottom)*

**Samriddhi:**  
"To validate our framework, we conducted standardized comparative evaluations across all tiers:  
*[Samriddhi points to Table 1 and Table 2]*

- In Table 1, our **Convolutional Autoencoder** establishes state-of-the-art anomaly gating with **0.961 SSIM**, while our **Flan-T5 Planner** achieves near-perfect semantic adherence.
- In Table 2, we contrast the strengths of **DCGAN vs. Velocity Diffusion**:  
  DCGAN provides **near-instant prototyping ($4.8\text{ ms}$)** for generating character rosters, while Velocity Diffusion provides **sub-pixel kinematic precision ($0.812\text{ IoU}$)** for animation.
- Both generative models natively output 4-channel RGBA, maintaining **over 94.8% to 96.1% pure alpha transparency**, ready for direct drag-and-drop into Unity, Unreal Engine, or Godot."

---

### [5:45 – 6:30] Part 6: Conclusion & Defense Hand-Off
*(Manish concludes)*

**Manish:**  
"In conclusion, GameForge AI demonstrates that production-grade generative game asset synthesis is best achieved not by a single monolithic model, but through a **cooperative multi-tier pipeline**:  
Transformers plan $\to$ DCGANs synthesize $\to$ Autoencoders verify $\to$ and Velocity Diffusion animates.

All components are optimized to run on commodity consumer hardware without requiring cloud GPUs, democratizing professional game development for indie creators.

Thank you very much for your time. We are now open to your questions."

---

## Anticipated Jury Questions & Bulletproof Answers

### Q1: "Why did you use DCGAN instead of just using Diffusion for everything?"
**Answer (Manish):**  
"That is a great engineering question. It comes down to **latency and compute economics**.  
DCGAN performs a single-pass forward inference in just **4.8 milliseconds**, which makes it ideal for rapid ideation and generating hundreds of character concepts in seconds. Diffusion, while superior for temporal animation, requires 20–50 iterative sampling steps ($24\text{ ms/step}$). By using DCGAN for the initial character exploration and reserving Diffusion for animating the chosen candidate, our pipeline gives the game developer the best of both worlds: instant prototyping and high-fidelity kinematic motion."

### Q2: "How does your Autoencoder know a sprite is 'corrupted' without having seen corrupted sprites during training?"
**Answer (Samriddhi):**  
"We leverage **unsupervised reconstruction error manifold analysis**.  
The Autoencoder was trained exclusively on valid, clean pixel art sprites. Consequently, its latent bottleneck learns the compact manifold of valid humanoid geometry and palettes. When a corrupted or noisy sprite is passed through, the decoder cannot accurately reconstruct the out-of-distribution noise, resulting in a large Mean Squared Error (MSE). Any sprite exceeding our calibrated 95th-percentile error threshold ($1.31 \times 10^{-3}$) is automatically flagged and rejected."

### Q3: "What is Velocity $v$-prediction in Diffusion, and why did you use it over standard $\epsilon$-prediction?"
**Answer (Manish):**  
"In standard diffusion, the neural network predicts the noise vector $\epsilon$. However, Salimans and Ho (2022) demonstrated that at low resolutions ($64 \times 64$) with high contrast, $\epsilon$-prediction causes severe color drift and contrast blowout during reverse sampling.  
By predicting the velocity vector $v_t = \sqrt{\bar{\alpha}_t}\epsilon - \sqrt{1 - \bar{\alpha}_t}x_0$, the model simultaneously interpolates between data and noise space. This completely stabilized our color palettes and prevented flickering across the 8-frame walk animation."

### Q4: "Where did you get your dataset, and are there copyright issues?"
**Answer (Samriddhi):**  
"Our dataset corpus comprises over 280,000 sprites sourced exclusively from open-source game art repositories, specifically the **Alucard Sprites corpus** and the **Liberated Pixel Cup (LPC)**. All assets are licensed under permissive open licenses, specifically **CC-BY-SA 3.0** and **GPL v3**, which explicitly permit AI research and derivative game asset creation with attribution."

