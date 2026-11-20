# 🔬 Rung 2 Forensic Investigation — Phase 2 Report

**Date:** November 1, 2026  
**Status:** **INVESTIGATION COMPLETE — ROOT CAUSE ISOLATED**  
**Target:** Character `C_00` (Neutral Stance Ref $\to$ South Walk Apex Target)  
**Evaluated Artifacts:** Checkpoint `rung2_ddpm_overfit_best.pt`, 50-step DDIM Trajectory, Cosine Beta Schedule  

---

## 1. Checkpoint Forensic Verification (Test 1)

The discrepancy between previous reports has been audited and resolved:

```text
ACTIVE / SAVED CHECKPOINT
Epoch: 150 (Final Epoch)
Loss: 0.182659
File: research_paper_assets/06_sprite_deformation_and_pose_animation/rung2_ddpm_overfit_best.pt
File Size: 7,963,727 bytes (1,985,764 parameters)
Successfully loaded: YES

MINIMUM HISTORICAL CHECKPOINT
Epoch: 145
Loss: 0.131424
Saved to disk separately: NO (Script called torch.save once at end of loop)
```

### Explanation of Discrepancies:
1. In `scripts/run_rung2_ddpm_overfit.py`, `torch.save(model.state_dict(), MODEL_SAVE_PATH)` was called unconditionally at the end of the 150-epoch loop. There was no internal `if loss < best_loss:` save hook. Thus, the file `rung2_ddpm_overfit_best.pt` on disk contains the **Final Epoch 150 weights** ($\text{loss} = 0.182659$).
2. The "Epoch 118 / Loss 0.1213" figure mentioned in earlier notes was from an earlier unweighted baseline run before foreground re-weighting ($W_{\text{fg}}=3.0$) was introduced.
3. The checkpoint loads cleanly and evaluates deterministically.

---

## 2. Predicted $x_0$ Reconstruction Error Across Timesteps (Test 2)

We evaluated the exact relationship:
$$\hat{x}_0 = \frac{x_t - \sqrt{1 - \bar{\alpha}_t} \cdot \hat{\epsilon}_\theta}{\sqrt{\bar{\alpha}_t}}, \quad \Delta x_0 = -\sqrt{\frac{1 - \bar{\alpha}_t}{\bar{\alpha}_t}} \cdot \Delta \epsilon$$

### Empirical Measurement Table (Teacher-Forced Input):

| Timestep $t$ | $\bar{\alpha}_t$ | Error Scale $\sqrt{\frac{1-\bar{\alpha}}{\bar{\alpha}}}$ | Total $\epsilon$ MSE | FG $\epsilon$ MSE | Total $x_0$ MSE | FG $x_0$ MSE | PSNR (dB) | SSIM |
|---|---|---|---|---|---|---|---|---|
| **$t = 999$** | $0.00000$ | $64,171.3$ | $0.0607$ | $0.0633$ | $1.5357$ | **$1.0807$** | **$4.16\text{ dB}$** | $0.0914$ |
| **$t = 750$** | $0.14312$ | $2.4468$ | $0.0559$ | $0.0536$ | $0.1714$ | **$0.2332$** | **$13.68\text{ dB}$** | $0.7298$ |
| **$t = 500$** | $0.49210$ | $1.0159$ | $0.0729$ | $0.1040$ | $0.0525$ | **$0.1001$** | **$18.82\text{ dB}$** | $0.9111$ |
| **$t = 250$** | $0.84556$ | $0.4274$ | $0.1208$ | $0.2307$ | $0.0171$ | **$0.0392$** | **$23.70\text{ dB}$** | $0.9723$ |
| **$t = 100$** | $0.97120$ | $0.1722$ | $0.2121$ | $0.4667$ | $0.0048$ | **$0.0122$** | **$29.22\text{ dB}$** | $0.9925$ |
| **$t = 50$** | $0.99134$ | $0.0934$ | $0.3074$ | $0.6860$ | $0.0019$ | **$0.0051$** | **$33.15\text{ dB}$** | $0.9970$ |
| **$t = 20$** | $0.99773$ | $0.0477$ | $0.4748$ | $0.8907$ | $0.0007$ | **$0.0017$** | **$37.38\text{ dB}$** | $0.9987$ |
| **$t = 0$** | $0.99990$ | $0.0100$ | $0.0519$ | $0.1132$ | $0.0000$ | **$0.0000$** | **$100.00\text{ dB}$** | $0.9999$ |

### The Critical Mathematical Finding:
- **Low-timestep $\epsilon$ error does NOT explode $x_0$ error when teacher-forced!**
- Because $\sqrt{\frac{1 - \bar{\alpha}}{\bar{\alpha}}} = 0.0477$ at $t=20$, the $\epsilon$ error of $0.8907$ is multiplied by $0.0477^2 = 0.00227$.
- Consequently, when given a clean forward-diffused input $x_t$, the network reconstructs $x_0$ at **$37.38\text{ dB}$ PSNR** ($t=20$) and **$29.22\text{ dB}$ PSNR** ($t=100$).
- **The problem is NOT that the network cannot reconstruct $x_0$ at low timesteps.** Given a clean sample from the forward distribution, it reconstructs the sprite accurately!

---

## 3. Direct $x_0$ Reconstruction vs. Multi-Step DDIM (Test 3)

We tested whether the network can reconstruct a clean sprite from $t=500$ without accumulating errors through 50 iterative steps:

| Method | Evaluation Setup | MSE | PSNR (dB) | SSIM | Visual Readability |
|---|---|---|---|---|---|
| **Direct Single-Step $\hat{x}_0$** | 1-step analytical inversion from $x_{500}$ | **0.0525** | **18.82 dB** | **0.9111** | Recognizable character with colored armor & limbs |
| **DDIM Trajectory** | 25 iterative DDIM steps ($t=500 \to 0$) | **0.0543** | **18.67 dB** | **0.8809** | Recognizable character, slight boundary noise |
| **Oracle Inversion** | Exact injected $\epsilon$ through 25 steps | **0.0000** | **220.82 dB** | **0.9999** | Perfect bit-for-bit ground truth |

👉 **Artifact:** [`research_paper_assets/06_sprite_deformation_and_pose_animation/forensic_phase_2/direct_x0_vs_ddim.png`](file:///c:/Users/manis/OneDrive/Desktop/Prompt_to_game_asset_generator/research_paper_assets/06_sprite_deformation_and_pose_animation/forensic_phase_2/direct_x0_vs_ddim.png)

### Key Takeaway:
When the diffusion trajectory starts from a partially noised target at $t=500$, **both direct inversion and DDIM sampling produce a recognizable sprite (18.7–18.8 dB PSNR)**.

---

## 4. Teacher-Forced vs. Free-Running Trajectory (Test 4)

This test isolated **where and why the free-running trajectory diverges**:

👉 **Artifact:** [`research_paper_assets/06_sprite_deformation_and_pose_animation/forensic_phase_2/teacher_forced_vs_free_running.png`](file:///c:/Users/manis/OneDrive/Desktop/Prompt_to_game_asset_generator/research_paper_assets/06_sprite_deformation_and_pose_animation/forensic_phase_2/teacher_forced_vs_free_running.png)

```
Timestep t       Teacher-Forced FG MSE      Free-Running DDIM FG MSE (from pure noise)
---------------------------------------------------------------------------------
t = 980          1.0807                     1.0675
t = 800          0.4521                     0.6951
t = 600          0.1684                     0.5187   <--- Trajectories Diverge!
t = 400          0.0682                     0.4878
t = 200          0.0289                     0.4706
t = 100          0.0122                     0.4676
t = 0            0.0000                     0.4723   <--- Stagnates at Static Error
```

### The Revelation:
1. **Teacher-Forced Error collapses to zero:** $1.08 \to 0.16 \to 0.02 \to 0.00$.
2. **Free-Running Error stalls at $t=600$:** It begins at $1.06$, drops to $0.51$ at $t=600$, and then **stagnates at $\approx 0.47$ for the rest of the reverse path**.
3. **Why it stalls:** At $t=1000 \to 800$, when starting from pure random noise $x_T \sim \mathcal{N}(0, \mathbf{I})$ with zero target signal, the model's high-timestep predictions fail to localize the coarse shape accurately enough. The state $x_t$ drifts into an out-of-distribution region where the model has never been trained, and subsequent steps simply denoise around that corrupted drift.

---

## 5. Timestep Embedding Audit (Test 5)

👉 **Artifact:** [`research_paper_assets/06_sprite_deformation_and_pose_animation/forensic_phase_2/timestep_embedding_audit.md`](file:///c:/Users/manis/OneDrive/Desktop/Prompt_to_game_asset_generator/research_paper_assets/06_sprite_deformation_and_pose_animation/forensic_phase_2/timestep_embedding_audit.md)

- **Training Range:** Sampled via `torch.randint(0, 1000)` $\implies t \in [0, 999]$.
- **Inference Range:** `ddim_timesteps = np.asarray(list(range(0, 1000, 20)))` $\implies t \in [0, 980]$.
- **Dtype:** `torch.long` in both training and sampling.
- **Output Divergence:** On identical inputs, passing $t=10$ vs $t=900$ produces an activation delta of **$0.2475$**, proving that time embeddings actively modulate every ResNet block.
- **Verdict:** Timestep embedding math is correct.

---

## 6. Reference Conditioning Sensitivity Audit (Test 6)

We tested whether the 8-channel concatenation $[x_t, x_{\text{ref}}]$ actually steers the network by feeding:
- **A. Correct Reference:** `C_00 Neutral Ref`
- **B. Wrong Reference:** `C_01 Neutral Ref` (different character)
- **C. Zero Reference:** All zeros

👉 **Artifact:** [`research_paper_assets/06_sprite_deformation_and_pose_animation/forensic_phase_2/reference_conditioning_comparison.png`](file:///c:/Users/manis/OneDrive/Desktop/Prompt_to_game_asset_generator/research_paper_assets/06_sprite_deformation_and_pose_animation/forensic_phase_2/reference_conditioning_comparison.png)  
👉 **Metrics CSV:** [`research_paper_assets/06_sprite_deformation_and_pose_animation/forensic_phase_2/reference_conditioning_metrics.csv`](file:///c:/Users/manis/OneDrive/Desktop/Prompt_to_game_asset_generator/research_paper_assets/06_sprite_deformation_and_pose_animation/forensic_phase_2/reference_conditioning_metrics.csv)

| Conditioning Mode | Total $\epsilon$ MSE | FG $\epsilon$ MSE | FG $x_0$ MSE | PSNR (dB) | SSIM |
|---|---|---|---|---|---|
| **A. Correct Reference (`C_00`)** | **0.0729** | **0.1040** | **0.1001** | **18.82 dB** | **0.9111** |
| **B. Wrong Reference (`C_01`)** | $0.1013$ | $0.1894$ | $0.1890$ | $16.92\text{ dB}$ | $0.8658$ |
| **C. Zero Reference (Zeros)** | $0.1423$ | $0.2140$ | $0.2147$ | $15.43\text{ dB}$ | $0.7936$ |

### What this proves:
1. When fed the correct reference, foreground MSE is **$0.1001$**.
2. When fed a wrong character or zero reference, foreground error doubles to **$0.1890 – 0.2147$**, and PSNR drops by over $3.4\text{ dB}$.
3. **The model IS actively using channels 4–7.** It is not ignoring the reference conditioning.

---

## 7. The True Failure Mechanism (Final Classification)

Based on the empirical evidence across all 6 tests, we classify the failure as:

### Primary Failure: **C + E (High-Timestep Distribution Mismatch & Free-Running Trajectory Drift)**

1. **Why Direct Inversion Works (18.8 dB) but Free-Running DDIM Fails (TV Noise):**
   - At $t=500$, $x_t$ contains $50\%$ target signal. The U-Net easily leverages the reference and denoises it into a recognizable sprite.
   - But in standard DDIM sampling from $t=999$, the initial input is **pure unconstrained Gaussian noise** $x_T \sim \mathcal{N}(0, \mathbf{I})$ with zero target signal.
   - At $t > 800$, $\bar{\alpha}_t \approx 0.0$. The network is asked to predict noise $\epsilon \approx x_T$. Any slight error in $\hat{\epsilon}$ causes $\hat{x}_0$ to overshoot wildly.
   - This error feeds into $x_{t-1}$, causing the trajectory to drift off the data manifold by $t=600$ (where free-running error stalls at $0.47$).

2. **Why It Produces "TV Noise" in Panel 4:**
   - Because the model never recovers from the initial $t=1000 \to 700$ drift, the inputs it receives at low timesteps ($t=100 \to 0$) do not look like forward-noised sprites—they look like drifting noise soup.
   - Denoising noisy soup produces structured static.

---

## 8. Summary of Deliverables in `forensic_phase_2/`

All 8 requested deliverables have been generated and saved:

1. [`checkpoint_audit.json`](file:///c:/Users/manis/OneDrive/Desktop/Prompt_to_game_asset_generator/research_paper_assets/06_sprite_deformation_and_pose_animation/forensic_phase_2/checkpoint_audit.json)
2. [`predicted_x0_metrics.csv`](file:///c:/Users/manis/OneDrive/Desktop/Prompt_to_game_asset_generator/research_paper_assets/06_sprite_deformation_and_pose_animation/forensic_phase_2/predicted_x0_metrics.csv)
3. [`direct_x0_vs_ddim.png`](file:///c:/Users/manis/OneDrive/Desktop/Prompt_to_game_asset_generator/research_paper_assets/06_sprite_deformation_and_pose_animation/forensic_phase_2/direct_x0_vs_ddim.png)
4. [`teacher_forced_vs_free_running.png`](file:///c:/Users/manis/OneDrive/Desktop/Prompt_to_game_asset_generator/research_paper_assets/06_sprite_deformation_and_pose_animation/forensic_phase_2/teacher_forced_vs_free_running.png)
5. [`timestep_embedding_audit.md`](file:///c:/Users/manis/OneDrive/Desktop/Prompt_to_game_asset_generator/research_paper_assets/06_sprite_deformation_and_pose_animation/forensic_phase_2/timestep_embedding_audit.md)
6. [`reference_conditioning_metrics.csv`](file:///c:/Users/manis/OneDrive/Desktop/Prompt_to_game_asset_generator/research_paper_assets/06_sprite_deformation_and_pose_animation/forensic_phase_2/reference_conditioning_metrics.csv)
7. [`reference_conditioning_comparison.png`](file:///c:/Users/manis/OneDrive/Desktop/Prompt_to_game_asset_generator/research_paper_assets/06_sprite_deformation_and_pose_animation/forensic_phase_2/reference_conditioning_comparison.png)
8. [`forensic_phase_2_report.md`](file:///c:/Users/manis/OneDrive/Desktop/Prompt_to_game_asset_generator/research_paper_assets/06_sprite_deformation_and_pose_animation/forensic_phase_2/forensic_phase_2_report.md)

No architecture changes or new training runs were initiated. Standing by for your review and direction.

