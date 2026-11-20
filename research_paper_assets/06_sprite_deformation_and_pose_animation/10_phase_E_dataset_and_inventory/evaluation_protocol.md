# RUNG 2 PHASE E: MULTI-POSE EVALUATION PROTOCOL
## Formal Evaluation Criteria for Multi-Pose Diffusion Synthesis

To evaluate future multi-pose diffusion models rigorously, GameForge AI establishes the following multi-dimensional benchmarking protocol:

---

## 1. EVALUATION METRICS SUITE

### A. Identity Preservation (Reference Consistency)
- **Reference SSIM:** Structural similarity between generated frame $x_{\text{pred}}(a, p)$ and reference neutral stance $x_{\text{ref}}$ on non-deforming regions (head, face, upper armor).
- **Target PSNR (dB):** Peak signal-to-noise ratio against ground-truth frame $x_{\text{GT}}(a, p)$ ($\ge 20.0\text{ dB}$).
- **Palette Drift ($\Delta E$):** Mean color difference on non-deformed torso and hair pixels to verify zero color mutation across animation frames.

### B. Kinematic & Pose Precision
- **Limb Centroid Displacement ($\Delta x, \Delta y$):** Horizontal and vertical centroid shift relative to reference stance.
- **Kinematic Error ($|\Delta x_{\text{pred}} - \Delta x_{\text{GT}}|$):** Target $\le 0.15\text{ px}$.
- **Silhouette IoU:** Intersection-over-Union of character alpha masks ($\ge 0.95$).

### C. Temporal & Animation Smoothness
- **Frame-to-Frame Trajectory Smoothness:**
  $$\mathcal{S} = \frac{1}{K-1} \sum_{p=1}^{K-1} \|x_{\text{pred}}(p) - x_{\text{pred}}(p-1)\|_1$$
  Ensures smooth pose progression without jerky discontinuities or frame flipping.
- **Centroid Path Monotonicity:** Verifies that limb stride follows an elliptical swing trajectory rather than random jumping.

---

## 2. SPLIT & GENERALIZATION HYGIENE

- **Strict Identity Splitting:** Split by character ID. All 23 frames of a character belong strictly to either Train or Val.
- **Recommended Split:**
  - 160 Training Characters (3,680 frames)
  - 40 Validation Characters (920 frames)
- **Zero Identity Leakage:** Val characters test whether the diffusion model can animate completely unseen outfits, hairstyles, and armor sets.
