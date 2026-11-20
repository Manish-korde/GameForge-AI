# RUNG 2 — CORRECTIVE EXPERIMENT A REPORT
## Reference-Grounded Warm-Start Feasibility Assessment

**Date:** November 1, 2026  
**Status:** COMPLETE — DECISIVE NEGATIVE RESULT FOR REFERENCE WARM-START  
**Artifact Directory:** `research_paper_assets/06_sprite_deformation_and_pose_animation/corrective_experiment_A/`  

---

## 1. Executive Summary & Verdict

| Experiment | PSNR vs Target | PSNR vs Ref | FG MSE vs Target | FG MSE vs Ref | Left Leg $\Delta x$ (GT = +0.40) | Decisive Visual & Physical Verdict |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Exp 1: Baseline** (Gaussian Init @ $t=999$) | 6.52 dB | 6.54 dB | 0.4723 | 0.4688 | -1.02 | **FAIL:** Pure TV static / unconstrained divergence. |
| **Exp 2: Control** (Target Warm-Start @ $t=500$) | 18.67 dB | 15.63 dB | 0.0900 | 0.2227 | +0.05 | **PASS (Control Only):** Reconstructs walking target with crisp legs. |
| **Exp 3: Ref Warm-Start** (Ref @ $t=500$, 16 Seeds) | 15.74 dB | **18.21 dB** | 0.2166 | **0.1136** | **-0.27** | **REJECTED AS POSE GENERATOR:** Reconstructs the *Neutral Reference*, not the Walk pose. |

### Decisive Verdict
**Reference-grounded warm-start at $t=500$ CANNOT be used as a pose generator.**
While it cleanly eliminates the high-noise TV static (producing crisp, valid pixel art at $18.21\text{ dB}$ PSNR with zero noise artifacts), it behaves strictly as an **identity-preserving auto-denoiser**: it denoises the noisy reference sprite back into the standing neutral reference ($\text{FG MSE} = 0.1136$), failing to introduce the target stride kinematics ($\text{FG MSE vs Target} = 0.2166$, leg $\Delta x = -0.27$ vs ground truth $+0.40$).

---

## 2. Experimental Setup

The experiment was executed with the production checkpoint `rung2_ddpm_overfit_best.pt` (Epoch 150, loss 0.217892) without retraining or modifying weights:
1. **Experiment 1 (Baseline):** 25-step DDIM starting from pure Gaussian noise $x_{999} \sim \mathcal{N}(0, \mathbf{I})$ conditioned on clean reference $C_{00}$.
2. **Experiment 2 (Target-Matched Reconstruction Control):** Injected cosine noise into ground-truth walk target $x_0^{\text{target}}$ at $t=500$:
   $$x_{500} = \sqrt{\bar{\alpha}_{500}} x_0^{\text{target}} + \sqrt{1 - \bar{\alpha}_{500}} \epsilon$$
   Conditioned on clean reference $C_{00}$, sampled 25 steps to $t=0$.
3. **Experiment 3 (Reference-Grounded Warm-Start):** Injected identical cosine noise into neutral reference $x_0^{\text{ref}}$ at $t=500$:
   $$x_{500} = \sqrt{\bar{\alpha}_{500}} x_0^{\text{ref}} + \sqrt{1 - \bar{\alpha}_{500}} \epsilon$$
   Conditioned on clean reference $C_{00}$, sampled 25 steps to $t=0$ across 16 deterministic seeds ($0$ to $15$).

---

## 3. Quantitative Results & Kinematic Validation

### 3.1 16-Seed Statistical Distribution (Experiment 3)

| Metric | Mean | Std Dev | Min | Max | Ground Truth Target Value |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **PSNR vs Reference** | **18.21 dB** | $\pm 0.09$ | 18.07 dB | 18.36 dB | — |
| **PSNR vs Target** | 15.74 dB | $\pm 0.11$ | 15.54 dB | 15.98 dB | $\infty$ |
| **SSIM vs Reference** | **0.8794** | $\pm 0.003$ | 0.8747 | 0.8831 | — |
| **SSIM vs Target** | 0.8035 | $\pm 0.004$ | 0.7937 | 0.8121 | 1.0000 |
| **FG MSE vs Reference** | **0.1136** | $\pm 0.004$ | 0.1056 | 0.1205 | — |
| **FG MSE vs Target** | 0.2166 | $\pm 0.005$ | 0.2065 | 0.2237 | 0.0000 |
| **Silhouette IoU vs Reference**| **0.8832** | $\pm 0.010$ | 0.8684 | 0.9033 | — |
| **Silhouette IoU vs Target** | 0.8117 | $\pm 0.009$ | 0.7968 | 0.8265 | 1.0000 |
| **Left Leg $\Delta x$ Displ.** | **-0.27** | $\pm 0.12$ | -0.41 | -0.04 | **+0.40** |
| **Leg Differing Pixels vs Ref**| **73.6** | $\pm 6.7$ | 64 | 86 | — |
| **Leg Differing Pixels vs Tgt**| 122.9 | $\pm 5.5$ | 114 | 132 | 0 |

---

## 4. Physical & Mathematical Diagnosis

### Why Warm-Starting at $t=500$ Reconstructs the Reference:
1. **Signal-to-Noise Ratio at $t=500$:**
   $$\bar{\alpha}_{500} = 0.5055, \quad \frac{\bar{\alpha}_{500}}{1 - \bar{\alpha}_{500}} \approx 1.02$$
   At $t=500$, exactly $50.55\%$ of the initial state $x_{500}$ is deterministic pixel signal and $49.45\%$ is noise.
2. **Score-Based Velocity Vector:**
   The reverse diffusion operator $p_\theta(x_{t-1} \mid x_t, x_{\text{ref}})$ follows the local density gradient. When $x_{500}$ contains $50\%$ standing sprite features (legs together), the learned score function denoises towards the nearest manifold mode seeded by that signal — which is the standing pose itself.
3. **Conditioning vs Initialization Conflict:**
   Because both the initial noisy tensor $x_{500}$ and the conditioning tensor $x_{\text{ref}}$ contain the standing reference, the network receives zero gradient pressure to shift pixels across 4–8 spatial positions to form a walking stride.
4. **Contrast with Experiment 2:**
   When $x_{500}$ is seeded with the walking target, it reconstructs the walking target with $\text{PSNR} = 18.67\text{ dB}$ and $\text{dx} = +0.05$. This proves that at $t \le 500$, DDIM functions as a local reconstructor/denoiser, not a long-range kinematic deformer.

---

## 5. Artifacts Produced
- Summary CSV: `research_paper_assets/06_sprite_deformation_and_pose_animation/corrective_experiment_A/corrective_experiment_a_summary.csv`
- 16-Seed Per-Run CSV: `research_paper_assets/06_sprite_deformation_and_pose_animation/corrective_experiment_A/reference_grounded_16seeds_metrics.csv`
- Visual Contact Sheet: `research_paper_assets/06_sprite_deformation_and_pose_animation/corrective_experiment_A/corrective_experiment_a_contact_sheet.png`

