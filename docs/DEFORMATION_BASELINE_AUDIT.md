# GameForge AI — Character Pose Deformation Baseline Audit
**Document ID**: `DEFORMATION_BASELINE_AUDIT`  
**Date**: October 23, 2026  
**Branch**: `feature/gameforge-pose-deformation` (HEAD at `cca64cc`)  
**Auditor Role**: Senior ML Engineer, Computer Vision Researcher, FastAPI & React Architect  

---

## 1. Executive Summary & Objective

This baseline audit evaluates the active state of **GameForge AI** before implementing the **Character Pose Deformation** feature. The primary objective is to build a reliable, interactive character pose deformation pipeline that allows a user to select a 2D pixel-art sprite, define or auto-detect a 2D character skeleton, adjust joint positions or select a target pose template (e.g. idle, walk, run, crouch, jump, attack), preview the deformed sprite, inspect quality metrics, and export a clean transparent RGBA PNG.

Per our development strategy:
1. **Geometric Deformation** is the dependable first implementation (piecewise affine / skeletal bone-rotation mesh warping, nearest-neighbor pixel-art resampling, non-transparent bounding box detection, and alpha preservation).
2. **VAE Feasibility Experiments** using the existing 280K $\beta$-VAE to test whether the latent space exhibits any pose disentanglement versus identity destruction.
3. **Hybrid Geometric + VAE** only if empirically justified.

---

## 2. Verified Repository Architecture & File Paths

| System Component | Actual Checked-Out Path | Verification Status | Notes |
|---|---|---|---|
| **FastAPI Root Server** | `backend/app.py` | Verified | Monolithic API server handling model lifecycle, VAE similarity search, reconstruction, and static SPA serving. |
| **NLP Transformer Service** | `backend/transformer_service.py` | Verified | Manages `google/flan-t5-small`, thread-safe memory management, and prompt safety guardrails. |
| **VAE Encoder Checkpoint** | `models/280k model VAE (VAE v2)/VAE_280K_Outputs/encoder_280k_final.keras` | Verified (35.1 MB) | 256-dimensional Gaussian latent prior ($\mu, \sigma^2$). Verified loadable in `backend/test_load.py`. |
| **VAE Decoder Checkpoint** | `models/280k model VAE (VAE v2)/VAE_280K_Outputs/decoder_280k_final.keras` | Verified (19.6 MB) | 256-d latent vector $\to 128 \times 128 \times 4$ RGBA reconstruction. |
| **Autoencoder Checkpoint** | `models/280k dataset model/AE_280K_best.keras` | Verified | 128×128 RGBA deterministic reconstruction & P95 outlier screening. |
| **DCGAN Checkpoints** | `models/DCGAN_280K_COMPLETE/weights/generator_latest.keras` | Verified (14.4 MB) | 64×64 RGBA unconditional sprite synthesis (50 epochs, 99,300 steps). |
| **Pre-Indexed Gallery** | `models/latent_gallery_index_25k.npy` | Verified (25.6 MB) | 25,000 pre-indexed VAE vectors serving sub-millisecond retrieval. |
| **Reference Sample Sprites**| `backend/alucard_samples/alucard_{0..34}.png` | Verified | 35 authentic pixel-art RGBA character/enemy/item sprites. |
| **Frontend Root & Routing** | `gui/src/App.jsx` | Verified | React Router v6 declaring `/`, `/concept`, `/create`, `/library`, `/experiments`, `/about`. |
| **Asset Studio Component** | `gui/src/pages/CreateAsset.jsx` | Verified (56 KB) | Primary interactive workspace for AE reconstruction, VAE variants, and interpolation. |
| **Frontend API Service** | `gui/src/services/api.js` | Verified (18.9 KB) | Client-side fetch handlers, multipart form assembly, and fallback logic. |

---

## 3. Existing Capabilities vs. Proposed Pose Deformation

| Feature Dimension | Existing Implementation in Repository | Proposed Character Pose Deformation Feature |
|---|---|---|
| **Pose Control** | **None**. No pose-conditioned generation exists. | 2D character skeletal deformation with 13 key joints (head, torso, limbs). |
| **Image Manipulation** | Latent-space perturbation ($z_{\text{var}} = \mu + \text{scale} \cdot \epsilon$) and linear latent interpolation ($A \to B$). | Direct 2D geometric bone-weighted warping with constrained joint rotations and translation. |
| **Pixel Art Preservation**| Decoded via neural decoders with MSE/Sigmoid output, causing smoothing of high-frequency pixel borders. | Pure nearest-neighbor and transparency-preserving geometric resampling, retaining crisp 1-pixel borders. |
| **Joint Constraints** | None. | Biomechanically bounded angle limits (e.g., elbow/knee hinge limits $[-150^\circ, 0^\circ]$). |
| **Pose Templates** | None. | Standardized named templates: `idle`, `walk_pose_1`, `walk_pose_2`, `run`, `jump`, `crouch`, `sword_slash`. |
| **Validation & QA** | AE reconstruction MSE threshold (`P95 = 0.001313`). | Automated aspect ratio preservation, canvas clipping detection, and extreme stretching warnings. |

---

## 4. Dataset & Annotation Status

Inspection of `dataset_audit/`, `scripts/inspect_alucard_dataset.py`, and `evaluation/dataset_analysis.json` establishes:
1. **Dataset Source**: `evilsocket/alucard-sprites` (HuggingFace), consisting of 282,511 unique 2D sprites.
2. **Data Structure**: Records contain only two fields: `image` (PIL Image) and `text` (crowd-sourced caption, e.g., *"pixel art, armored warrior"*).
3. **Annotation Deficit**:
   - **Zero pose labels**: No joint coordinates, skeleton tags, or action annotations exist.
   - **Zero paired frames**: Sprites are isolated, static single-frame assets without animation sequences.
   - **No paired character poses**: Characters are not grouped by canonical identity ID across different poses.
4. **Implication**: Any claim that a neural model in this repository can perform pose-conditioned image-to-image synthesis from scratch is **empirically false**. Pose deformation must rely on a dependable geometric foundation.

---

## 5. Reproducible Blockers & Defect Diagnosis

### Blocker 1: Broken Import on Server Boot (`ModuleNotFoundError: No module named 'gan'`)
- **Root Cause**: In commit `cca64cc`, the developer deleted the obsolete `gan/` directory (Texture GAN / Vanilla GAN), but left line 201 in `backend/app.py`:
  ```python
  from gan.gan_service import load_generator, generate_textures, get_status as get_gan_status
  ```
- **Reproduction Command**:
  ```bash
  backend\venv\Scripts\python.exe -c "import sys; sys.path.insert(0, 'backend'); import app"
  ```
- **Observed Traceback**:
  ```text
  ModuleNotFoundError: No module named 'gan'
  ```
- **Required Minimal Safe Fix**: Guard the import with `try...except ImportError` so `app.py` boots cleanly without depending on the deleted module, while preserving all existing routes.

### Blocker 2: Duplicate Route Definition in `backend/app.py`
- Line 343 defines `@app.post("/generate_concept")` expecting JSON `ConceptRequest`.
- Line 1112 defines `@app.post("/generate_concept")` expecting `prompt: str = Form(...)`.
- The second route masks the first, causing frontend JSON requests to trigger 422 Unprocessable Entity errors.

---

## 6. Security Risks Relevant to Pose Deformation

1. **Unbounded Canvas Warping DoS**: If a user submits extreme joint coordinates or an oversized image, spatial transformation algorithms can induce high CPU loops or large array allocations. The endpoint must enforce image dimension limits (e.g. max $512 \times 512$, defaulting to $128 \times 128$) and parameter bounds.
2. **Local Path Exposure in Errors**: Ensure all exceptions during geometric deformation or model inference return sanitized JSON error messages rather than raw Python tracebacks or local directory paths.
3. **Input Validation**: Validate that uploaded files are valid RGBA images and reject corrupt binaries before executing NumPy/PIL processing.

---

## 7. Files Planned for Implementation & Modification

### New Backend Modules:
1. `backend/deformation/__init__.py`: Package root.
2. `backend/deformation/schemas.py`: Pydantic request and response models for joints, skeletons, poses, and metrics.
3. `backend/deformation/skeleton.py`: Standard 13-joint 2D humanoid skeleton hierarchy, default T-pose / idle pose, and named pose templates (`walk`, `run`, `jump`, `crouch`, `attack`).
4. `backend/deformation/geometric.py`: Core geometric transformation engine (bounding-box auto-detection, skeleton binding, piecewise-affine and skeletal bone-rotation mesh warping, nearest-neighbor pixel-art resampling, alpha mask preservation, and distortion metrics).
5. `backend/deformation/vae_adapter.py`: Isolated experimental adapter interfacing with `encoder_280k_final.keras` and `decoder_280k_final.keras` for latent-space feasibility testing.
6. `backend/deformation/service.py`: High-level orchestrator executing geometric deformation, validation checks, and optional VAE comparison.

### Modified Existing Modules:
1. `backend/app.py`:
   - Guard stale `gan.gan_service` import with `try...except ImportError` to unblock server startup.
   - Register dedicated pose deformation endpoints:
     - `POST /deformation/pose`
     - `GET /deformation/templates`
     - `POST /deformation/detect_skeleton`
2. `gui/src/services/api.js`:
   - Add frontend API client methods: `deformCharacterPose()`, `fetchPoseTemplates()`, `detectCharacterSkeleton()`.
3. `gui/src/pages/CreateAsset.jsx`:
   - Integrate an interactive **Character Pose Deformation Studio** tab/section into the existing Asset Studio, featuring:
     - Interactive SVG/Canvas skeleton overlay with draggable joint markers.
     - Pose template selector dropdown (Idle, Walk 1, Walk 2, Run, Jump, Crouch, Attack).
     - Live before/after side-by-side sprite preview with transparency grid.
     - Pixel-art nearest-neighbor rendering toggle.
     - Validation metadata (stretching warnings, bounding box, processing latency).
     - Export transparent PNG button.

### New Test Suites:
1. `backend/tests/test_deformation_geometric.py`: Unit tests for geometric transformations, joint constraints, alpha preservation, and boundary edge cases.
2. `backend/tests/test_deformation_api.py`: FastAPI integration tests for the `/deformation/pose` endpoint using `httpx` / `TestClient`.
3. `backend/tests/test_vae_feasibility.py`: Automated feasibility experiment test comparing geometric vs. VAE outputs.
