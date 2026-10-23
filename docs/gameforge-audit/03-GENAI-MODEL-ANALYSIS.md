# GameForge AI — Generative AI Model Deep-Dive & Training Analysis
**Document ID**: `03-GENAI-MODEL-ANALYSIS`  
**Audit Date**: October 23, 2026  
**Auditor Role**: Senior Generative AI Researcher & Model Evaluation Specialist  
**Audit Target**: Model Families, Checkpoints, Hyperparameters, Training Pipelines, and Inference Mechanics  

---

## 1. Executive Model Inventory & Taxonomy

GameForge AI does **not** rely on a single foundation model. Instead, it combines **four distinct neural models**, spanning sequence-to-sequence language modeling, adversarial generative synthesis, deterministic representation learning, and probabilistic latent autoencoding:

| Component | Exact Model ID / Architecture | Parameter Count | Precision / Format | Artifact Location in Repo | Primary Task |
|---|---|---|---|---|---|
| **Semantic Planner (Fine-Tuned)** | `google/flan-t5-small` (Seq2Seq Transformer) | 76,961,152 (~77M) | FP32 / `safetensors` | `models/gameforge_t5/` | Natural Language $\to$ Structured Game Spec (JSON) |
| **Semantic Planner (Cloud Fallback)** | `gemini-2.5-flash` (Google DeepMind API) | Undisclosed (Proprietary) | Cloud API Service | External (Google API Client in `app.py`) | Cloud-based fallback for Game Spec JSON generation |
| **Adversarial Sprite Synthesis** | Unconditional DCGAN (Deep Convolutional GAN) | 3,579,648 (Generator) + 2,768,385 (Discriminator) | FP32 / Keras SavedModel | `models/DCGAN_280K_COMPLETE/weights/generator_latest.keras` | Pure Noise $z \sim \mathcal{N}(0, I) \to$ Novel 64×64 RGBA Sprite |
| **Quality Control & Outlier QA** | Deep Convolutional Autoencoder (AE 280K) | 1,079,108 (~1.08M) | FP32 / Keras SavedModel | `models/280k dataset model/AE_280K_best.keras` | 128×128 RGBA Reconstruction & P95 Outlier Screening |
| **Probabilistic Representation & Search** | Deep Convolutional $\beta$-VAE ($\beta=0.001$) | ~2.16M (Encoder + Decoder) | FP32 / Keras SavedModel | `models/280k model VAE (VAE v2)/VAE_280K_Outputs/` | 256-d Latent Space, Sub-millisecond Vector Search, Morphing |

---

## 2. Investigation: Is the Project Training a Model?

A critical requirement of this audit is to resolve whether GameForge AI:
- Trains a foundation model from scratch.
- Fine-tunes an existing model.
- Uses retrieval-augmented generation (RAG).
- Calls a hosted model API.
- Runs an existing model locally.

### 2.1 Empirical Findings on Training vs. Inference

Based on rigorous source code and artifact inspection, **GameForge AI executes both model fine-tuning and full model training from scratch, complemented by local inference and hosted API fallbacks**:

1. **Transformer Fine-Tuning**:
   - The project **fine-tuned an existing open foundation model** (`google/flan-t5-small`).
   - *Training Script*: `transformer_training/train_transformer.py` using HuggingFace's `Seq2SeqTrainer`.
   - *Dataset Scale*: 1,600 training pairs, 200 validation pairs, 200 test pairs stored in `transformer_training/train.jsonl`, `validation.jsonl`, and `test.jsonl`.
   - *Vocabulary Augmentation*: The original T5 tokenizer maps `{` and `}` to `<unk>`. The script added `{` (token ID 32100) and `}` (token ID 32101) directly to the tokenizer vocabulary and resized model token embeddings before fine-tuning.
   - *Training Checkpoints*: Stored in `models/gameforge_t5/model.safetensors` (307 MB) with recorded final training loss of `0.5228` over 10 epochs.

2. **DCGAN Training from Scratch**:
   - The project **trained an unconditional DCGAN completely from scratch** on random Gaussian weights.
   - *Training Code*: `Notebook/02_DCGAN_280K_Training (1).ipynb` and `models/DCGAN_280K_COMPLETE/config/run_config.json`.
   - *Hardware & Compute*: Kaggle P100 GPU for ~5.67 hours (average 409 seconds/epoch).
   - *Scale*: 50 epochs, 99,300 global optimizer steps on 282,511 RGBA sprites from `evilsocket/alucard-sprites`.
   - *Weights*: Generator (`generator_latest.keras`, 14.38 MB) and Discriminator (`discriminator_latest.keras`, 11.13 MB).

3. **Autoencoder & VAE Training from Scratch**:
   - Both the AE and VAE were **trained from scratch** on the 282,511 Alucard sprite dataset.
   - *Training Notebooks*: `Notebook/280k_dataset_autoencoder.ipynb` and `Notebook/280k_dataset_VAE.ipynb`.
   - *AE Specs*: Trained for 50 epochs with MSE loss and Adam optimizer; best validation MSE reached `0.000491` at epoch 32.
   - *VAE Specs*: Trained with $\beta=0.001$ weighting on the Kullback-Leibler divergence term $\mathcal{D}_{KL}(q(z|x) \parallel \mathcal{N}(0, I))$.

4. **Retrieval-Augmented Latent Search (Latent RAG)**:
   - The project utilizes a vector-retrieval mechanism across **25,000 pre-computed latent vectors** (`latent_gallery_index_25k.npy`), matching user queries in latent space rather than relying purely on autoregressive generation.

5. **Hosted API Call**:
   - `backend/app.py` incorporates an optional hosted call to Google DeepMind's `gemini-2.5-flash` using `google-generativeai`.

---

## 3. Detailed Component Deep-Dive

### 3.1 Component 1: Transformer Semantic Planning Engine

#### Architecture & Configuration
- **Base Architecture**: Encoder-Decoder T5 (`T5ForConditionalGeneration`).
- **Layers**: 8 encoder layers, 8 decoder layers, 6 attention heads per layer ($d_{\text{model}} = 512$, $d_{\text{ff}} = 1024$, $d_{\text{kv}} = 64$).
- **Context Window**: $N_{\text{positions}} = 512$ tokens. Max input length: 256 tokens; max output length: 512 tokens.
- **Activation Function**: Gated-GELU (`gelu_new`).
- **Tokenizer**: Custom SentencePiece with 32,102 vocabulary items (including added `{` and `}`).

#### Prompt Construction & System Template
In `transformer_training/train_transformer.py`, training examples use the strict task prefix:
```text
game concept: <User Natural Language Prompt>
```
The target output is a strictly formatted, compact JSON string:
```json
{
  "theme": "Dark Fantasy",
  "environment": "Cursed Forest Village",
  "characters": [{"name": "Rogue Assassin", "desc": "Stealth shadow warrior"}],
  "weapons": [{"name": "Broadsword", "desc": "Heavy steel blade"}],
  "props": [{"name": "Acid Trap", "desc": "Corrosive chemical pool"}],
  "visualStyle": "16-bit Retro Pixel Art"
}
```

#### Inference Parameters
In `check_generation.py`:
- Generation Mode: Beam Search (`num_beams = 4`, `do_sample = False`).
- Max New Tokens: 512 tokens.

#### Implementation Disconnect in `backend/transformer_service.py`
In `backend/transformer_service.py` (lines 26–31), instead of loading the locally fine-tuned checkpoint (`models/gameforge_t5`), the service attempts to load base `google/flan-t5-small` from the Hugging Face Hub. It queries T5 for simple text completions (`num_beams = 2, max_new_tokens = 48`):
```python
t5_title = self._generate_with_t5(f"Give a title for a video game about {user_prompt}")
t5_summary = self._generate_with_t5(f"Describe the main character and setting for a game about {user_prompt}")
```
It then populates the structured JSON fields (genre, role, hazards, enemies, tags) using a **rule-based keyword heuristic dictionary** (matching words like "delivery", "pizza", "farm", "space", "knight"). If the model is not loaded, it falls back to the identical rule-based heuristic engine labeled `"Dynamic Semantic Engine"`.

---

### 3.2 Component 2: Unconditional DCGAN Sprite Generator

#### Generative Formulation
The DCGAN models the visual distribution $p_{\text{data}}(x)$ of 2D game sprites via an adversarial zero-sum game:
$$\min_G \max_D V(D, G) = \mathbb{E}_{x \sim p_{\text{data}}}[\log D(x)] + \mathbb{E}_{z \sim p_z}[\log(1 - D(G(z)))]$$

#### Generator Architecture
- **Input**: Latent noise vector $z \sim \mathcal{N}(0, I_{100})$ ($z \in \mathbb{R}^{100}$).
- **Dense Projection**: `Dense(4 * 4 * 512 = 8192)` $\to$ `Reshape((4, 4, 512))` $\to$ `BatchNormalization` $\to$ `ReLU`.
- **Transposed Convolutions**:
  - `Conv2DTranspose(256, kernel_size=4, strides=2, padding='same')` $\to 8 \times 8 \times 256 \to$ `BatchNorm` $\to$ `ReLU`.
  - `Conv2DTranspose(128, kernel_size=4, strides=2, padding='same')` $\to 16 \times 16 \times 128 \to$ `BatchNorm` $\to$ `ReLU`.
  - `Conv2DTranspose(64, kernel_size=4, strides=2, padding='same')` $\to 32 \times 32 \times 64 \to$ `BatchNorm` $\to$ `ReLU`.
  - `Conv2DTranspose(4, kernel_size=4, strides=2, padding='same')` $\to 64 \times 64 \times 4$ (RGBA).
- **Final Activation**: **Tanh** (maps pixels to $[-1.0, 1.0]$).
- **Parameters**: 3,579,648 trainable parameters (~13.66 MB).

#### Discriminator Architecture
- **Input**: Candidate sprite $x \in \mathbb{R}^{64 \times 64 \times 4}$.
- **Convolutions**:
  - `Conv2D(64, kernel_size=4, strides=2, padding='same')` $\to 32 \times 32 \times 64 \to$ `LeakyReLU(0.2)`.
  - `Conv2D(128, kernel_size=4, strides=2, padding='same')` $\to 16 \times 16 \times 128 \to$ `BatchNormalization` $\to$ `LeakyReLU(0.2)`.
  - `Conv2D(256, kernel_size=4, strides=2, padding='same')` $\to 8 \times 8 \times 256 \to$ `BatchNormalization` $\to$ `LeakyReLU(0.2)`.
  - `Conv2D(512, kernel_size=4, strides=2, padding='same')` $\to 4 \times 4 \times 512 \to$ `BatchNormalization` $\to$ `LeakyReLU(0.2)`.
- **Output Head**: `Flatten()` $\to$ `Dense(1)` (unscaled scalar logit).
- **Parameters**: 2,768,385 trainable parameters (~10.56 MB).

#### Training Dynamics & Stability
- **Optimizer**: Adam ($\text{lr} = 0.0002, \beta_1 = 0.5, \beta_2 = 0.999$) for both $G$ and $D$.
- **Loss**: Binary Cross-Entropy with Logits (`from_logits=True`).
- **Batch Size**: 128 (global across 2 replicas on Kaggle P100).
- **Duration**: 50 epochs (99,300 steps). Loss curves stabilized with $D_{\text{loss}} \in [0.015, 0.040]$ and $G_{\text{loss}} \in [8.0, 8.9]$ without catastrophic mode collapse.

---

### 3.3 Component 3: Autoencoder (AE 280K)

#### Architecture
- **Input**: $x \in \mathbb{R}^{128 \times 128 \times 4}$ (RGBA normalized to $[0.0, 1.0]$).
- **Encoder**: Strided `Conv2D` layers (downsampling $128 \times 128 \to 64 \times 64 \to 32 \times 32 \to 16 \times 16 \to 8 \times 8 \times 256$).
- **Bottleneck**: Compressed representation of shape $8 \times 8 \times 256 = 16,384$ values.
- **Decoder**: Symmetric `Conv2DTranspose` layers (upsampling back to $128 \times 128 \times 4$) with Sigmoid output.
- **Loss Function**: Mean Squared Error (MSE):
  $$\mathcal{L}_{\text{AE}} = \frac{1}{N} \sum_{i=1}^N \|x_i - \hat{x}_i\|^2$$
- **Parameters**: 1,079,108 parameters.

#### Quantitative Reconstruction Performance
Evaluated across test split (`evaluation/metrics.json`):
- **Median MSE**: `0.000256` (Test overall mean MSE: `0.000456`).
- **Mean PSNR**: `35.36 dB` (Median: `35.91 dB`).
- **Empirical Outlier Calibration**:
  - 50th Percentile (Median): `0.000254`
  - 90th Percentile: `0.000644`
  - **95th Percentile Outlier Threshold**: `0.001274 – 0.001313`
  - Maximum Recorded MSE: `0.033964`

#### Application: Automated Quality Assurance
Because the AE optimizes pure reconstruction without KL variance smoothing, assets that diverge from the learned sprite distribution (corrupted alpha channels, photo textures, non-pixel art) produce high reconstruction errors. The backend flags any sprite with $\text{MSE} > 0.001313$ as an outlier candidate for human review.

---

### 3.4 Component 4: Variational Autoencoder ($\beta$-VAE 280K)

#### Formulation & Latent Prior
The VAE optimizes the Evidence Lower Bound (ELBO) with a relaxation parameter $\beta = 0.001$:
$$\mathcal{L}_{\text{VAE}} = \mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)] - \beta \cdot \mathcal{D}_{\text{KL}}(q_\phi(z|x) \parallel \mathcal{N}(0, I))$$
Where the latent space is 256-dimensional ($z \in \mathbb{R}^{256}$).

#### Encoder & Reparameterization Trick
- The encoder outputs two heads: $\mu(x) \in \mathbb{R}^{256}$ and $\log \sigma^2(x) \in \mathbb{R}^{256}$.
- Stochastically sampled using the custom Keras `Sampling` layer:
  $$z = \mu + \exp(0.5 \cdot \log \sigma^2) \odot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I_{256})$$

#### Empirical Performance vs. Baseline
Evaluated on a 5,000-sprite candidate pool (`evaluation/retrieval_benchmark.json` and `CURRENT_PROJECT_STATUS.md`):
- **Supervised KNN Category Classification**:
  - Naive Majority-Class Baseline (*Characters*): **65.19%**
  - Autoencoder Latent Accuracy: 77.80% (+12.61% lift)
  - **VAE Latent Accuracy**: **84.40%** (**+19.21 percentage point lift** over baseline)
- **Top-5 Retrieval Precision (Precision@5)**: **62.80%**
- **K-Means Latent Silhouette Score**: VAE `0.1624` vs. AE `0.0912` (VAE demonstrates 78% tighter geometric clustering).

---

## 4. Why Were These Specific Models Chosen?

The repository contains explicit architectural rationales across `CURRENT_PROJECT_STATUS.md` and `purpose_of_model_in_our_project.md`. We classify each rationale into Verified, Inferred, or Unknown:

| Model Selection Question | Original Stated Rationale | Classification | Audit Verification & Findings |
|---|---|---|---|
| **Why Flan-T5 for Semantic Planning?** | Open weights, small memory footprint (80M params), sequence-to-sequence structure suitable for structured text generation. | **Verified** | Verified by training scripts (`train_transformer.py`). However, runtime memory constraints on Railway (1GB) led developers to implement aggressive model unloading and fallback keyword rules. |
| **Why Unconditional DCGAN instead of cGAN?** | Alucard category labels are noisy, unvalidated crowd tags. Conditional GANs suffer from class collapse and boundary confusion when trained on contradictory labels. | **Verified** | Verified in `documentation/ALUCARD_GAN_LABEL_REVIEW_AND_IDEATION.md`. Unconditional DCGAN models pixel distribution directly with zero label reliance. |
| **Why Autoencoder AND Variational Autoencoder?** | Mathematical trade-off: AE has no KL penalty, achieving superior pure reconstruction fidelity (median MSE 0.000782 vs VAE's higher MSE) for outlier QA; VAE enforces a smooth continuous Gaussian latent space for similarity search and morphing. | **Verified** | Verified in `evaluation/retrieval_benchmark.json`. AE achieves higher PSNR; VAE achieves +19.21% classification lift and smoother interpolation. |
| **Why 64×64 for DCGAN vs 128×128 for AE/VAE?** | GAN training stability on single-GPU Kaggle environment; 128×128 DCGAN suffers from gradient vanishing and checkerboard artifacts without complex multi-scale discriminators. | **Inferred** | Sound engineering decision. Training 64×64 DCGAN reached 99,300 steps stably without mode collapse. |
| **Why is Gemini 2.5 Flash in `app.py`?** | Cloud fallback for high-quality game concept generation when local PyTorch model runs out of memory or fails. | **Inferred** | Implemented as a late addition in `app.py` (lines 1112–1154), but accidentally duplicated route path and broke frontend JSON contract. |

---

## 5. Model Limitations & Failure Modes

1. **DCGAN Structural Blurriness**: As an unconditional 64×64 DCGAN, generated sprites possess plausible color palettes and silhouettes (swords, hats, bodies), but lack crisp, fine-grained single-pixel art outlines. High-frequency pixel borders require post-processing discretization.
2. **VAE Edge Smoothing**: Standard pixel MSE loss on the VAE decoder smooths high-contrast alpha transparency boundaries, making direct random sampling from the VAE decoder noticeably blurrier than DCGAN outputs.
3. **Flan-T5 Tokenizer Brace Limitation**: Base T5 cannot tokenize `{` or `}` without fine-tuning vocabulary expansion. In the live API, because the base model or regex fallback is invoked rather than the custom checkpoint, complex nested JSON structures cannot be reliably synthesized without hardcoded template wrapping.
