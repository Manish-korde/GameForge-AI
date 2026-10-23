# GameForge AI — Sources, Benchmark Citations & Repository Evidence Index
**Document ID**: `10-SOURCES-AND-EVIDENCE`  
**Audit Date**: October 23, 2026  
**Auditor Role**: Senior Technical Documentation Writer & AI Researcher  
**Scope**: External Literature, Official Vendor Documentation, and Internal Repository Evidence Mapping  

---

## 1. External Literature & Foundational Academic Papers

1. **Radford, A., Metz, L., & Chintala, S. (2015).** *Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks.* arXiv:1511.06434 [cs.LG].
   - *Relevance to GameForge AI*: Primary architectural basis for the unconditional 64×64 DCGAN generator and discriminator (`models/DCGAN_280K_COMPLETE`). Validates strided convolutions, batch normalization, Tanh output activation, and Adam hyperparameters ($\text{lr}=0.0002, \beta_1=0.5$).
   - *URL*: [https://arxiv.org/abs/1511.06434](https://arxiv.org/abs/1511.06434)

2. **Kingma, D. P., & Welling, M. (2013).** *Auto-Encoding Variational Bayes.* arXiv:1312.6114 [stat.ML].
   - *Relevance to GameForge AI*: Primary mathematical formulation for the 256-dimensional $\beta$-VAE (`models/280k model VAE (VAE v2)`), specifically the reparameterization trick ($z = \mu + \sigma \odot \epsilon$) and the Evidence Lower Bound (ELBO) objective with KL-divergence penalty.
   - *URL*: [https://arxiv.org/abs/1312.6114](https://arxiv.org/abs/1312.6114)

3. **Goodfellow, I., Pouget-Abadie, J., Mirza, M., Xu, B., Warde-Farley, D., Ozair, S., Courville, A., & Bengio, Y. (2014).** *Generative Adversarial Nets.* Advances in Neural Information Processing Systems (NeurIPS 2014).
   - *Relevance to GameForge AI*: Theoretical foundation of the minimax adversarial loss function optimized during DCGAN 50-epoch training.
   - *URL*: [https://arxiv.org/abs/1406.2661](https://arxiv.org/abs/1406.2661)

4. **Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, L., & Polosukhin, I. (2017).** *Attention Is All You Need.* Advances in Neural Information Processing Systems (NeurIPS 2017).
   - *Relevance to GameForge AI*: Architectural foundation for the sequence-to-sequence Transformer mechanism utilized by `google/flan-t5-small`.
   - *URL*: [https://arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762)

5. **Chung, H. W., Hou, L., Longpre, S., Zoph, B., Tay, Y., Fedus, W., Li, Y., Wang, X., Mostajabdaveh, M., et al. (2022).** *Scaling Instruction-Finetuned Language Models (Flan).* arXiv:2210.11416 [cs.CL].
   - *Relevance to GameForge AI*: Establishes the instruction-tuning methodology of the `google/flan-t5-small` foundation model.
   - *URL*: [https://arxiv.org/abs/2210.11416](https://arxiv.org/abs/2210.11416)

6. **Wang, Z., Bovik, A. C., Sheikh, H. R., & Simoncelli, E. P. (2004).** *Image Quality Assessment: From Error Visibility to Structural Similarity.* IEEE Transactions on Image Processing, 13(4), 600–612.
   - *Relevance to GameForge AI*: Defines the SSIM metric (`0.9248`) used to evaluate sprite structural fidelity in `backend/app.py` line 406.

---

## 2. Official Model Documentation & Pricing Sources (Retrieved October 2026)

| Model Family / Provider | Official Documentation Link | Verified Pricing & Context Specifications | Access Date |
|---|---|---|---|
| **OpenAI GPT-4o-mini** | [OpenAI Pricing & Models](https://openai.com/api/pricing/) | $0.150 / 1M input tokens; $0.600 / 1M output tokens; $0.075 / 1M cached input; 128,000 token context window. | October 2026 |
| **Google Gemini 1.5 Flash** | [Google DeepMind Gemini API Docs](https://ai.google.dev/pricing) | Free Tier (15 RPM / 1,500 RPD); Paid Tier: $0.075 / 1M input (<128K); $0.300 / 1M output (<128K); 1,048,576 token context window. | October 2026 |
| **Anthropic Claude 3.5 Haiku** | [Anthropic Claude API Pricing](https://www.anthropic.com/pricing) | $0.800 / 1M input tokens; $4.000 / 1M output tokens; $0.080 / 1M cache read; 200,000 token context window. | October 2026 |
| **Alibaba Qwen 2.5 Coder / 7B** | [Qwen 2.5 GitHub & HuggingFace](https://github.com/QwenLM/Qwen2.5) | Open Weights (Apache 2.0); 131,072 token context window; HumanEval >85%; zero token cost when self-hosted. | October 2026 |
| **Meta Llama 3.2 3B** | [Meta Llama 3.2 Release Docs](https://ai.meta.com/blog/llama-3-2-connect-2024/) | Open Weights (Llama 3.2 Community License); 128,000 token context window; optimized for mobile/edge/CPU inference. | October 2026 |
| **HuggingFace Datasets: Alucard** | [HuggingFace evilsocket/alucard-sprites](https://huggingface.co/datasets/evilsocket/alucard-sprites) | 282,511 deduplicated 2D game sprites (128×128 RGBA); Creative Commons CC-BY-NC-SA 4.0 license. | October 2026 |
| **HuggingFace Datasets: ViGGO** | [HuggingFace GEM/viggo](https://huggingface.co/datasets/GEM/viggo) | 6,900 video game dialogue act / intent samples; Creative Commons CC-BY 4.0 license. | October 2026 |
| **HuggingFace Datasets: PICO-8** | [HuggingFace Fraser/pico-8-games](https://huggingface.co/datasets/Fraser/pico-8-games) | 10,967 open-source retro Lua game cartridges; CC-BY-NC-SA 4.0 license. | October 2026 |

---

## 3. Comprehensive Repository Evidence Index

This index maps every finding, claim, and metric in the audit reports to exact file paths and line ranges within the GameForge AI repository:

| Report Finding / Topic | Repository File Path | Line Range | Verifying Code / Content Evidence |
|---|---|---|---|
| **Missing Module / Broken Import** | `backend/app.py` | Lines 200–202 | `from gan.gan_service import load_generator, generate_textures, get_status as get_gan_status` (Directory `gan/` does not exist). |
| **Duplicate Route Collision** | `backend/app.py` | Line 343 & Line 1112 | Definition 1: `@app.post("/generate_concept") def generate_concept(req: ConceptRequest)`<br/>Definition 2: `@app.post("/generate_concept") async def generate_concept(prompt: str = Form(...))` |
| **SSRF Vulnerability** | `backend/app.py` | Lines 381–385 | `if url_or_path.startswith("http"): response = requests.get(url_or_path, timeout=5)` (No private IP or loopback validation). |
| **Wildcard Permissive CORS** | `backend/app.py` | Lines 63–69 | `CORSMiddleware(allow_origins=["*"], allow_credentials=True, allow_methods=["*"])` |
| **Keras Deserialization Monkey-Patch** | `backend/app.py` | Lines 38–54 | `kwargs.pop('quantization_config', None)` monkey-patched onto `Dense`, `Conv2D`, `Conv2DTranspose`, etc. |
| **Memory-Mapped Gallery Index** | `backend/app.py` | Lines 635–644 | `GALLERY_INDEX = np.load(npy_path_25k, mmap_mode="r")` across 25,000 vectors. |
| **PyTorch Cooperative Memory Unloading** | `backend/transformer_service.py` | Lines 36–47 | `self.model = None; self.tokenizer = None; gc.collect()` |
| **Prompt Safety Boundary Filter** | `backend/transformer_service.py` | Lines 66–89 | `validate_prompt_safety()` filtering NSFW terms and flagging protected IP (`mario`, `pokemon`). |
| **Responsible AI Unit Tests** | `backend/test_ethical_guardrails.py` | Lines 10–40 | 4 automated unit test cases validating safe, unsafe, and IP-infringing prompts. |
| **Local Model Deserialization Test** | `backend/test_load.py` | Lines 43–53 | Test script loading `VAE_ENCODER_PATH` and `VAE_DECODER_PATH` with custom `Sampling` layer. |
| **Transformer Brace Vocabulary Hack** | `test_json_token_fix.py` | Lines 1–40 | Confirms `{` (ID 32100) and `}` (ID 32101) encode and decode in `models/gameforge_t5`. |
| **Flan-T5 Fine-Tuning Training Pipeline** | `transformer_training/train_transformer.py` | Lines 76–100 | `Seq2SeqTrainer` configuration: 10 epochs, $\text{lr}=10^{-4}$, warmup 100 steps, batch size 8. |
| **T5 Evaluation Benchmarks** | `models/gameforge_t5/evaluation_summary.json` | Lines 1–27 | 200 test examples: 100% valid JSON, 100% schema valid, 11% exact match, theme accuracy 97.5%. |
| **DCGAN Training Configuration** | `models/DCGAN_280K_COMPLETE/config/run_config.json` | Lines 1–42 | 50 epochs, global batch size 128, Adam ($\beta_1=0.5, \beta_2=0.999$), Tanh normalization $[-1, 1]$. |
| **DCGAN Training State & Steps** | `models/DCGAN_280K_COMPLETE/config/training_state.json` | Lines 1–27 | `global_step: 99300`, `epoch: 49`, `best_g_loss: 3.4059`. |
| **Autoencoder Evaluation Metrics** | `evaluation/metrics.json` | Lines 1–11 | Median MSE `0.000256`, Mean PSNR `35.36 dB`, 95th percentile MSE `0.001274`. |
| **AE vs VAE Benchmark Results** | `evaluation/retrieval_benchmark.json` | Lines 1–33 | VAE KNN accuracy 84.40% (+19.21% over baseline), Precision@5 62.80%, P95 threshold `0.001313`. |
| **Dataset Deduplication Audit** | `evaluation/dataset_analysis.json` | Lines 1–55 | 312,550 raw rows; 30,039 exact duplicates removed; 282,511 unique 128×128 RGBA sprites. |
| **Multi-Stage Docker Packaging** | `Dockerfile` | Lines 1–62 | Stage 1 `node:20-alpine` builds React SPA; Stage 2 `python:3.10-slim` serves unified API and SPA. |
| **Frontend Mock Diffusion Model** | `gui/src/services/api.js` | Lines 179–201 | `generateImage()` returns mock Pokemon sprites labeled `source: 'Diffusion', model: 'Diffusion Model'`. |
| **Frontend Client-Side Fallback Engine**| `gui/src/services/api.js` | Lines 83–151 | Catches API failure and executes regex string parsing to build client-side fallback spec. |
| **Vestigial Texture Studio Page** | `gui/src/pages/TextureStudio.jsx` | Lines 1–140 | Calls deleted endpoint `${API}/gan/generate_texture`; unrouted in `gui/src/App.jsx`. |

---

## 4. Claims That Could Not Be Verified

To maintain strict scientific integrity, the following items were identified in documentation or mockups but could **not** be verified from operational code:

1. **"Text-to-Image Diffusion Model"**: Referenced in `gui/src/pages/Dashboard.jsx`, `gui/src/services/api.js`, and `mockData.js`. No Stable Diffusion, Latent Diffusion, or UNet model weights exist in the repository. It is an unintegrated concept demonstrated via mock assets.
2. **"PICO-8 Cartridge Lua Code Generator"**: `documentation/PICO-8_INTEGRATION_BRAINSTORMING.md` discusses generating executable Lua carts. No Lua generation pipeline or runtime emulator exists in the active application.
3. **"VastTextures Seamless Material Generator"**: Referenced in early brainstorm documents. Confirmed deleted and abandoned in commit `cca64cc` in favor of unconditional sprite synthesis.
4. **"Production Game Export Pipeline"**: Packaging game assets into Godot scene trees or Unity bundles is described in presentation roadmaps but not implemented in code.
