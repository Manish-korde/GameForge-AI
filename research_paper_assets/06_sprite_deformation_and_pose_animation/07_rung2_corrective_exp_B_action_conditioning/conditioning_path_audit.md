# TEST 2: CONDITIONING PATH & ARCHITECTURAL AUDIT
## Forward Pass & Conditioning Mechanism of `ReferenceConditionedUNet`

**Audit Target:** `scripts/run_rung2_ddpm_overfit.py` (Lines 93–175)  
**Total Trainable Parameters:** 1,990,340 (~1.99M params)  

---

## 1. Exact Input Channel Configuration

```python
# scripts/run_rung2_ddpm_overfit.py, Line 130
self.in_conv = nn.Conv2d(8, 32, 3, padding=1)

# Forward call, Line 154
x = torch.cat([x_t, x_ref], dim=1)
```

- **Input Dimension:** $[B, 8, 64, 64]$
  - Channels $0..3$: $x_t \in \mathbb{R}^{B 	imes 4 	imes 64 	imes 64}$ (Noisy target RGBA at timestep $t$).
  - Channels $4..7$: $x_{	ext{ref}} \in \mathbb{R}^{B 	imes 4 	imes 64 	imes 64}$ (Clean reference sprite RGBA).
- **Mechanism:** Pure **input concatenation (early spatial fusion)**.
- **Missing Conditioning Mechanisms:**
  - Zero cross-attention layers.
  - Zero FiLM / Adaptive Group Normalization (AdaGN) blocks.
  - Zero ControlNet residual zero-convolutions.

---

## 2. Timestep vs Reference Conditioning Interaction

- **Timestep Embedding:**
  ```python
  self.time_mlp = nn.Sequential(
      SinusoidalPositionEmbeddings(128),
      nn.Linear(128, 128),
      nn.SiLU()
  )
  ```
  In every residual block:
  ```python
  time_emb = self.silu(self.time_mlp(t)) # [B, out_c]
  h = h + time_emb[(..., ) + (None, ) * 2] # Spatially broadcasted additive bias
  ```
- **Spatial Independence:**
  The timestep vector modulates feature channels uniformly across all $64 	imes 64$ spatial locations. It carries zero spatial steering information (cannot command "move left leg by +4 px").
- **Reference Preservation:**
  The reference image features enter at the $64 	imes 64$ stem and propagate down through encoder stages ($32 	imes 32, 16 	imes 16$) and back through skip connections ($h_1, h_2$). High-frequency color palettes and clothing boundaries are strongly preserved, which explains why character appearance and color scheme are faithfully memorized.

---

## 3. Action Conditioning Audit: The Total Absence of Pose Conditioning

| Conditioning Channel | Present in Architecture? | Implementation Details |
| :--- | :---: | :--- |
| **Action Class Embedding** | ❌ **NONE** | No discrete action index or embedding table. |
| **Action Text Token / CLIP** | ❌ **NONE** | No text encoder or token projection. |
| **Pose Heatmap / Skeleton** | ❌ **NONE** | No OpenPose / keypoint spatial channel. |
| **Target Frame Number** | ❌ **NONE** | Model has no awareness of frame index (F0 vs F3). |

### Mathematical Consequence:
The forward pass is strictly parameterized as:
$$\epsilon_	heta(x_t, t, x_{	ext{ref}})$$
It contains **no variable representing the desired action $a \in \mathcal{A}$**.
The model is structurally incapable of distinguishing between:
- "Synthesize Walk Frame 3"
- "Synthesize Slash Frame 3"
- "Synthesize Neutral Idle"

When the user requests an action, the model has no channel through which that request can be delivered.
