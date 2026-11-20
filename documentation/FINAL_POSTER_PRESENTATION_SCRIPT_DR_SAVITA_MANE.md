# GameForge AI: Final Poster Presentation Script
**Project Title:** GameForge AI: Multimodal Game Asset & Action Synthesis  
**A 5-Tier Generative Framework for Prompt-Driven 2D Sprite Synthesis, Latent Verification & Kinematic Walk Cycles**  
**Guide:** Prof. Dr. Savita Mane | Department of Computer Engineering, MITAOE, Pune  
**Presenters:** Samriddhi Das, Manish Korde, Swayam Pagare  
**Target Duration:** ~3 minutes 45 seconds (Academic Poster Evaluation Format)

---

## 🎙️ Spoken Presentation Script

### [0:00 – 0:35] Act 1: Introduction & The Core Problem
*(Pointing to Top Header Banner & Project Statement on the Left)*

**Samriddhi:**  
> *"Good morning respected judges and faculty. I am **Samriddhi Das**, along with my team members **Manish Korde** and **Swayam Pagare**. We are presenting our capstone project, **GameForge AI: Multimodal Game Asset & Action Synthesis**, developed under the guidance of **Prof. Dr. Savita Mane** at MIT Academy of Engineering.*  
>  
> *In indie game development, creating cohesive 2D character sprite sheets and multi-frame animations is one of the most labor-intensive and costly bottlenecks. While modern foundation models like Stable Diffusion excel at photorealistic art, they fail drastically on 2D pixel art: they produce blurry sub-pixel aliasing, lack alpha-channel background transparency, and suffer from severe kinematic jitter when generating sequential animation frames.*  
>  
> *To solve this, we engineered a coordinated 5-tier generative system specifically calibrated for 64×64 retro game assets."*

---

### [0:35 – 1:15] Act 2: Dataset, Key Objectives & System Flowchart
*(Pointing to Dataset Description, Key Objectives, and the Center Architecture Flowchart)*

**Manish:**  
> *"To train our models, we compiled the **GameForge Multimodal Corpus** of over **280,000 raw sprites** from Hugging Face and OpenGameArt, with **46,000 rigorously audited front-facing character sprites**. Every sprite was preprocessed with strict four-corner alpha gating and normalized into $[-1, 1]$ to avoid memorizing background noise.*  
>  
> *If you look at our **End-to-End Architecture Flowchart** right here in the center:*  
> *We realized that text-to-image networks shouldn't be monolithic. Instead, we built a **Decoupled Two-Stage Pipeline**:*  
> 1. *A natural language prompt first enters our **Flan-T5 Semantic Planner**, which decomposes the user's intent into a structured JSON game specification.*  
> 2. *These parsed attributes then steer our visual backbones: **DCGAN** for instant character synthesis, **VAE** for continuous latent representation and morphing, **Conv AE** as an unsupervised quality assurance gate, and a **Velocity Diffusion U-Net** for 8-frame kinematic walk cycles.*  
> 3. *Finally, everything integrates into a local **FastAPI + React GUI** for interactive indie game authoring."*

---

### [1:15 – 2:30] Act 3: The 5 Generative Models (Core Technical Walkthrough)
*(Pointing to the 5 Model Cards in the center grid)*

**Samriddhi:**  
> *"Let us walk you through the distinct roles of our five models:*  
>  
> * **Model 1 is our Convolutional Autoencoder (3.24M parameters):** *Trained with MSE reconstruction loss, it serves as an unsupervised QA gate. Any sprite exceeding our empirical P95 reconstruction error threshold of $1.31 \times 10^{-3}$ is flagged as an anomaly, achieving a **94.2% outlier rejection accuracy** with a high PSNR of **35.36 dB**.*  
>  
> * **Model 2 is our Variational Autoencoder (VAE), which is our standout theoretical model:** *Equipped with a probabilistic Conv2D encoder and 256-D continuous Gaussian bottleneck, it optimizes the Evidence Lower Bound (ELBO). Unlike GANs, the VAE enforces a smooth, continuous manifold via KL-divergence, achieving an **84.40% k-NN latent retrieval accuracy** and an **88.6% silhouette separability**. This eliminates mode collapse and allows game developers to perform smooth spherical linear interpolation (SLERP) to morph character armor and classes without tearing.*  
>  
> * **Model 3 is our DCGAN Character Generator (5.40M parameters):** *Using nearest-neighbor upsampling and feature-matching loss, it acts as our high-speed baseline. It generates 100 character candidates in just **0.8 seconds (4.8 ms per sprite)** with **94.80% alpha mask purity**, followed by a k-Means quantizer that locks continuous outputs to 16-color retro palettes."*

**Manish:**  
*(Pointing to Model 4 & Model 5)*  
> *"Moving to our higher-level intelligence tiers:*  
>  
> * **Model 4 is our Transformer Planner (Flan-T5-small):** *Standard T5 tokenizers map JSON curly braces to unknown `<unk>` tokens. We patched the tokenizer vocabulary with custom tokens 32100 and 32101, fine-tuning on 2,000 prompt-specification pairs. It achieved **100.0% JSON schema validity** and **97.5% semantic theme accuracy** on our held-out test set, running locally in just 42 milliseconds.*  
>  
> * **Model 5 is our Multi-Pose Velocity Diffusion U-Net (1.98M parameters):** *Generating multi-frame animations from static images usually suffers from identity drift. By switching from standard epsilon-noise prediction to a **velocity $v$-objective** with cosine scheduling and reverse DDIM sampling, our diffusion model synthesizes cohesive 8-frame walk cycles from a single reference character, achieving a **0.812 Limb Stride IoU** on completely unseen characters and **91.4% cyclic consistency**."*

---

### [2:30 – 3:15] Act 4: Performance Metrics & Scientific Insights
*(Pointing to the Comparative Benchmark Tables & Challenges box)*

**Samriddhi:**  
> *"Looking at our **Standardized Benchmark Protocol tables** below:*  
> *In our Verification tier, Conv AE delivers superior raw reconstruction fidelity at **35.36 dB PSNR and 0.961 SSIM**, while VAE provides the mathematically structured latent space needed for character similarity search and clustering.*  
>  
> *In our Generative tier, DCGAN gives ultra-fast exploratory throughput at **4.8 ms**, while Velocity Diffusion delivers state-of-the-art animation coherence with a minimal validation MSE of **0.0031**.*  
>  
> *Our major scientific learning was that continuous convolutional networks cannot produce sharp discrete pixel art out-of-the-box. Coupling continuous latent generation with downstream **k-Means palette snapping and alpha-thresholding** solved the sub-pixel blurring problem while keeping inference fast."*

---

### [3:15 – 3:45] Act 5: Conclusion & Closing
*(Pointing to Conclusion and Group Members)*

**Manish:**  
> *"To conclude:*  
> 1. ***VAE is our premier latent representation engine*** — *its 84.40% k-NN retrieval and smooth continuous manifold provide the mathematical foundation for asset management and morphing.*  
> 2. ***Flan-T5 proves that decoupled semantic planning works*** — *achieving 100% schema validity without the overhead of massive multi-billion parameter foundation models.*  
> 3. ***Velocity Diffusion sets the new benchmark for 2D kinematic walk cycles***, *eliminating animation jitter.*  
>  
> *By unifying these models into a single pipeline, GameForge AI provides an end-to-end, edge-viable AI asset studio for indie game creators.*  
>  
> *We would like to express our sincere gratitude to our guide, **Prof. Dr. Savita Mane**, for her invaluable mentorship throughout this research.*  
>  
> *Thank you, and we are now open to your questions."*

---

## 🎯 Quick Rehearsal Cheat-Sheet (Card Transitions)

| Slide / Section | Speaker | Visual Cue (Where to Point) | Key Numbers to Emphasize |
| :--- | :--- | :--- | :--- |
| **Intro & Problem** | **Samriddhi** | Top Header & Project Statement | 2D pixel art bottleneck, blur & jitter |
| **Dataset & Flowchart** | **Manish** | Dataset Card & Center Flowchart | 280k raw / 46k audited, Decoupled Pipeline |
| **Models 1, 2, 3** | **Samriddhi** | M1 (AE), M2 (VAE), M3 (DCGAN) | **AE:** 94.2% QA gate \| **VAE:** 84.40% k-NN, 256-D \| **DCGAN:** 4.8 ms |
| **Models 4 & 5** | **Manish** | M4 (Flan-T5), M5 (Diffusion) | **T5:** 100% schema, 97.5% theme \| **Diffusion:** 0.812 Limb IoU |
| **Benchmark Tables** | **Samriddhi** | Comparative Tables at the bottom | PSNR 35.36 dB vs 26.59 dB |
| **Conclusion & Thanks** | **Manish** | Bottom Conclusion & Guide box | VAE representation winner, thank Dr. Savita Mane |

