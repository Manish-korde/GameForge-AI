# GameForge AI — Repository Structure & Git Branch Audit
**Document ID**: `07-REPOSITORY-AND-BRANCH-AUDIT`  
**Audit Date**: October 23, 2026  
**Auditor Role**: Senior Software Architect & Repository Auditor  
**Inspection Scope**: Git Metadata, Directory Hierarchy, Branch Lineage, Commit History, and Inspection Coverage  

---

## 1. Repository Inventory & Directory Map

The GameForge AI repository contains **15 top-level subdirectories and 11 root configuration/script files**:

```
f:\Project\GameForge-AI/
├── .dockerignore                            # Docker container build exclusions
├── .git/                                    # Git version control metadata
├── .gitattributes                           # Git LFS & line ending attributes
├── .gitignore                               # Tracked ignore patterns
├── .venv_transformer/                       # Python virtual environment for transformer
├── Dockerfile                               # Multi-stage production container build
├── Notebook/                                # Jupyter research & evaluation notebooks
│   ├── 02_DCGAN_280K_Training (1).ipynb     # 50-epoch DCGAN training notebook
│   ├── 280K_AE_vs_VAE_Comparison.ipynb      # Comparative evaluation notebook
│   ├── 280K_Autoencoder_Evaluation.ipynb    # Deterministic reconstruction analysis
│   ├── 280K_VAE_Evaluation.ipynb            # VAE latent space analysis
│   ├── 280k_dataset_VAE.ipynb               # VAE training notebook
│   ├── 280k_dataset_autoencoder.ipynb       # Autoencoder training notebook
│   ├── Alucard_Dataset_Analysis.ipynb       # Statistical audit of 282K sprites
│   ├── DCGAN_280K_Demonstration_and_Evaluation.ipynb # Complete DCGAN demonstration
│   ├── Denoising_Autoencoder_50K_Kaggle.ipynb # Dual-T4 Kaggle denoising run
│   └── experimentaion/                      # Scaffolding & early experiments
├── README.md                                # Master project documentation
├── backend/                                 # FastAPI application & ML inference
│   ├── alucard_samples/                     # 35 static reference RGBA sprites
│   ├── app.py                               # Monolithic FastAPI server & routes
│   ├── requirements_docker.txt              # Production Python package manifest
│   ├── test_ethical_guardrails.py           # Unit tests for prompt safety
│   ├── test_load.py                         # Smoke test for Keras model deserialization
│   └── transformer_service.py               # Sequence-to-sequence NLP service
├── build files/                             # Early project technical summaries & capsules
├── check_generation.py                      # Standalone T5 generation test script
├── check_tokenizer.py                       # Tokenizer vocabulary verification script
├── config/                                  # Train/val/test index splits & manifest
│   ├── dataset_manifest.json                # Summary metadata
│   ├── experiment_config.json               # Training hyperparameter configuration
│   ├── shuffled_unique_indices.json         # Full shuffled index array (2.1 MB)
│   ├── test_indices.json                    # 10% test split index array (107 KB)
│   ├── train_indices.json                   # 80% train split index array (1.9 MB)
│   ├── unique_indices.json                  # Deduplicated dataset index array
│   └── val_indices.json                     # 10% validation split index array
├── dataset_audit/                           # Dataset deduplication & quality inventory
│   ├── archive_inventory.csv                # Dataset file size & hash records
│   ├── duplicate_report.csv                 # Exact image duplicate report
│   ├── image_inventory.csv                  # 500-sample quality survey
│   └── quality_summary.json                 # Aggregate statistics
├── documentation/                           # Historical project docs & presentation notes
│   ├── CURRENT_PROJECT_STATUS.md            # Definitive project ground truth
│   ├── GAN_MODULE_STATUS.md                 # DCGAN training & architecture specs
│   ├── PROJECT_ROADMAP.md                   # Milestone progression
│   ├── purpose_of_model_in_our_project.md   # Architectural defense for reviews
│   └── ...                                  # Presentation slides and review notes
├── evaluation/                              # Quantitative evaluation outputs & plots
│   ├── ae_outlier_metrics.json              # Outlier quantile calibration
│   ├── dataset_analysis.json                # Category distributions & alpha stats
│   ├── metrics.json                         # Overall AE reconstruction metrics
│   ├── retrieval_benchmark.json             # 1,500-sample retrieval evaluation
│   ├── retrieval_metrics.json               # 5,000-sample retrieval evaluation
│   └── plots/                               # Loss curves and comparison figures
├── evaluation_vae/                          # VAE-specific evaluation plots & metrics
├── gui/                                     # React 18 / Vite Single Page Application
│   ├── package.json                         # Node dependency manifest
│   ├── vite.config.js                       # Vite bundler configuration
│   └── src/
│       ├── App.jsx                          # Main routing specification
│       ├── components/                      # TopBar, Sidebar, AppShell
│       ├── pages/                           # ConceptPlanner, CreateAsset, Experiments, etc.
│       └── services/                        # api.js, mockData.js
├── logs/                                    # Training history logs
├── models/                                  # Neural network weights & latent indices
│   ├── 10k dataset model/                   # Deprecated initial prototype model
│   ├── 280k dataset model/                  # Trained Autoencoder (AE_280K_best.keras)
│   ├── 280k model VAE (VAE v2)/             # Trained VAE (encoder & decoder .keras)
│   ├── DCGAN_280K_COMPLETE/                 # Trained DCGAN (generator, discriminator, logs)
│   ├── gameforge_t5/                        # Fine-tuned Flan-T5 model & tokenizer
│   ├── latent_gallery_index.npy             # 5,000-vector latent matrix
│   ├── latent_gallery_index_25k.npy         # 25,000-vector latent matrix (25.6 MB)
│   ├── latent_gallery_manifest.json         # 5,000-item metadata manifest
│   └── latent_gallery_manifest_25k.json     # 25,000-item metadata manifest (4.4 MB)
├── requirements.txt                         # Root Python dependencies
├── requirements_transformer.txt             # PyTorch & HuggingFace training dependencies
├── scripts/                                 # Data analysis, index building, and benchmarks
├── test_json_token_fix.py                   # Tokenizer brace fix verification
├── tmp_check.txt                            # Scratch status marker
└── transformer_training/                    # Transformer training pipeline & JSONL datasets
    ├── dataset_statistics.json              # Token distribution statistics
    ├── evaluate_transformer.py              # Automated JSON schema evaluator
    ├── test.jsonl                           # 200 test prompt-spec pairs
    ├── train.jsonl                          # 1,600 training prompt-spec pairs
    ├── train_transformer.py                 # HuggingFace Seq2Seq fine-tuning script
    └── validation.jsonl                     # 200 validation prompt-spec pairs
```

---

## 2. Git Branch & Commit History Audit

### 2.1 Current Working Tree Status
- **Current Branch**: `main`
- **Tracking Reference**: `origin/main`
- **Working Tree State at Audit Start**: **Clean** (`## main...origin/main`, zero untracked or modified files).
- **Latest Commit**:
  - Hash: `cca64cc7f5a8062551de876ec1a3f7b8cf797d13`
  - Author: `Manish-Korde <manishk230604@gmail.com>`
  - Date: `Thu Oct 22 21:20:30 2026 +0530`
  - Subject: `Remove Vanilla GAN/Texture GAN, add DCGAN complete (50ep/99300 steps), clean scripts, update all docs to 4-model pipeline`

### 2.2 Branch Topology & Relationship
Execution of `git branch --list -a` identified three local and two remote tracking branches:
- `* main`: The active trunk branch.
- `Transformer`: Feature branch used to develop the Flan-T5 fine-tuning pipeline.
- `Transformer-backup`: Safety snapshot of the feature branch.
- `remotes/origin/main`: Remote tracking pointer for `main`.
- `remotes/origin/Transformer`: Remote tracking pointer for `Transformer`.

### 2.3 Commit Lineage Analysis
Inspection of commit history shows that pull request #1 (`1943316 Merge pull request #1 from Manish-korde/Transformer`) merged the Transformer training code (`011221b`) into `main`. Subsequent commits on `main` executed:
1. `4fbb86a`: Optimized Docker memory usage with thread-safe Transformer lazy loading.
2. `a6db2d3`: Updated `Dockerfile` CMD to support dynamic `${PORT:-8000}` syntax.
3. `263e112`: Implemented `mmap_mode="r"` on `latent_gallery_index_25k.npy` to prevent OOM on 1GB hosts.
4. `523cfd2`: Added path diagnostics for Railway deployment.
5. `cca64cc`: Final commit removing abandoned Texture GAN and Vanilla GAN notebooks, committing the completed 50-epoch DCGAN model, and updating documentation across the repository.

---

## 3. Inspection Coverage & Uninspected Areas

### 3.1 Fully Inspected Subsystems (100% Source Code Coverage)
- All Python backend source code: `backend/app.py`, `backend/transformer_service.py`, test files.
- All frontend source code: `gui/src/App.jsx`, all pages, components, and `services/api.js`.
- All JSON/CSV configuration and benchmark files in `config/`, `evaluation/`, and `dataset_audit/`.
- Training scripts: `transformer_training/train_transformer.py`, `evaluate_transformer.py`, `validate_dataset.py`.
- Containerization: `Dockerfile`, `.dockerignore`, `requirements_docker.txt`.
- All markdown documentation files in `documentation/`, `build files/`, and root.

### 3.2 Opaque / Binary / Generated Areas (Inspected via Manifests & Metadata)
- **Neural Model Weights**: `models/gameforge_t5/model.safetensors` (307 MB), `models/DCGAN_280K_COMPLETE/weights/generator_latest.keras` (14.38 MB), and `models/280k dataset model/AE_280K_best.keras` were verified for structural presence, size, and header integrity, but raw binary float matrices were not decompiled.
- **Large Pre-computed Arrays**: `models/latent_gallery_index_25k.npy` (25.6 MB) was inspected via shape diagnostics (`(25000, 256)`) and manifest cross-referencing rather than printing raw float values.
- **Node Modules & Virtual Environments**: `gui/node_modules/` and `.venv_transformer/` were verified as standard third-party environments and excluded from manual code auditing.

---

## 4. Read-Only Verification

In strict compliance with the **Project Safety Rules**:
- No existing source files in `backend/`, `gui/`, `models/`, or `scripts/` were edited, refactored, or modified.
- No dependency manifests (`package.json`, `requirements.txt`) were updated or altered.
- No Git branch was switched, created, merged, reset, committed, pushed, or pulled.
- All documentation artifacts generated by this audit were written exclusively into the dedicated, non-conflicting directory: `docs/gameforge-audit/`.
- A final `git status` check confirms that the original tracked files remain 100% pristine.
