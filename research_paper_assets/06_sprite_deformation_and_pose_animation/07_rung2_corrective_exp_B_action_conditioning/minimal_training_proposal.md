# TEST 5: MINIMAL CORRECTIVE TRAINING PROPOSAL
## The Single Most Important Intervention: Action-Conditioned Discrete Diffusion

---

## 1. Diagnostic Synthesis of Failure Modes

Across Rungs 0, 1, 2, Forensic Phases 1–4, and Corrective Experiment A:
1. **Kinematics are Learnable:** Rung 1 direct regression proved that a 1.93M U-Net effortlessly maps Neutral Reference to Walk Stride ($IoU = 0.86$, $L_1 = 0.0817$).
2. **Reverse Sampling Divergence:** Rung 2 DDPM failed when initialized from pure Gaussian noise at $t=999$ because noise prediction error at $t > 800$ is multiplied by $rac{1-ar{lpha}}{ar{lpha}} = 1,140$, squashing generated sprites into flat static.
3. **Warm-Start Failure:** Corrective Experiment A proved that warm-starting at $t=500$ with $x_{	ext{ref}}$ merely reconstructs the neutral stance ($	ext{FG MSE} = 0.1136$, leg $\Delta x = -0.27$ vs target $+0.40$).
4. **Action Degeneracy:** Corrective Experiment B proved that the current pipeline has zero action conditioning and only 1 target pose per character.

---

## 2. The Minimal Corrective Intervention: Experiment C

To isolate and solve the root cause without unconstrained architectural churn, we propose **Corrective Experiment C**:

### Core Hypothesis:
Conditioning the diffusion model on both **(1) Clean Reference Appearance** and **(2) Discrete Action Identity**, trained with **SNR-Weighted Loss / $v$-Prediction**, will enable the model to generate the requested walking pose from noise while preserving character identity.

### Specification of Minimal Intervention:

| Component | Current Broken State (Rung 2) | Proposed Corrective Intervention (Exp C) |
| :--- | :--- | :--- |
| **Dataset Pairs** | 20 pairs (Walk F3 only) | **40 pairs:** 20 characters $	imes$ 2 actions (**Idle F0** and **Walk F3**) |
| **Conditioning** | Reference image only | Reference image + **Discrete Action Embedding** ($e_a \in \mathbb{R}^{128}$) |
| **Loss Formulation** | Raw $\epsilon$-MSE with uniform timesteps | **Min-SNR $(\gamma=5)$ Weighted Loss** or **$v$-prediction** to suppress high-$t$ noise blowup |
| **Diffusion Steps** | 50-step DDIM from $t=999$ | 50-step DDIM with Min-SNR weighting |
| **Model Size** | 1,990,340 parameters | 1,991,364 parameters ($+1,024$ params, $+0.05\%$) |

---

## 3. Quantitative Success Criteria for Experiment C

A successful corrective training run must satisfy ALL four criteria:
1. **Pose Differentiation:**
   - When conditioned on `action_id = 0` (Idle), generated sprite has legs together ($\Delta x \le 0.10	ext{ px}$, $IoU_{	ext{idle}} \ge 0.85$).
   - When conditioned on `action_id = 1` (Walk), generated sprite has legs separated ($\Delta x \ge +0.35	ext{ px}$, $IoU_{	ext{walk}} \ge 0.82$).
2. **Identity Preservation:**
   - Generated sprite matches reference character clothing/palette ($	ext{PSNR} \ge 17.0	ext{ dB}$, $	ext{SSIM} \ge 0.85$).
3. **Absence of TV Noise:**
   - Zero divergent pixel clamping; background opacity $\le 0.05$.
4. **Kinematic Stride Fidelity:**
   - Left-leg changed centroid displacement matches ground truth direction ($\Delta x > 0$).
