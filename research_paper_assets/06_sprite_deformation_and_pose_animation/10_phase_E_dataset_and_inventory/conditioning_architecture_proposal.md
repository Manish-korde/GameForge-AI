# RUNG 2 PHASE E: CONDITIONING ARCHITECTURE PROPOSAL
## Multi-Pose Discrete Conditioning for ActionConditionedUNet

### 1. MATHEMATICAL FORMULATION

In the current production model (Refined C2), the conditioning vector combines continuous timestep embedding with discrete action embedding:

$$c(t, a) = \text{MLP}_{\text{time}}(\text{PE}(t)) + \mathbf{E}_{\text{act}}[a], \quad c \in \mathbb{R}^{128}$$

To enable full animation pose generation, we extend this conditioning mechanism to incorporate discrete **frame / pose index** $p \in \{0, 1, \dots, K-1\}$:

$$c(t, a, p) = \text{MLP}_{\text{time}}(\text{PE}(t)) + \mathbf{E}_{\text{act}}[a] + \mathbf{E}_{\text{pose}}[p], \quad c \in \mathbb{R}^{128}$$

Where:
- $\text{PE}(t) \in \mathbb{R}^{128}$ is sinusoidal positional encoding for timestep $t \in [0, 999]$.
- $\mathbf{E}_{\text{act}} \in \mathbb{R}^{A \times 128}$ is an embedding table for $A$ discrete action categories (e.g. Idle, Walk, Attack, Cast).
- $\mathbf{E}_{\text{pose}} \in \mathbb{R}^{P \times 128}$ is an embedding table for $P$ discrete frame indices within the selected action.

---

### 2. PARAMETER COUNT & CAPACITY IMPACT

The proposed extension is **exceptionally lightweight and non-disruptive**:

| Component | Dimensions | Parameters | Notes |
|---|---|---|---|
| Existing `time_mlp` | $128 \to 128$ | 16,640 | Unchanged |
| Existing `action_embed` | $4 \times 128$ | 512 | Supports 4 actions |
| Proposed `pose_embed` | $10 \times 128$ | 1,280 | Supports up to 10 frames per action |
| Total Additional Parameters | — | **+1,280 params** | **+0.06% increase over C2** |

*Crucial Design Advantage:*
Because $\mathbf{E}_{\text{pose}}[p]$ has the identical dimension ($128$) and injects via additive linear conditioning, **no modifications are made to the convolutional channels, downsampling stages, bottleneck, skip connections, or DDIM sampler**. The architecture remains 100% mathematically compatible with Refined C2.

---

### 3. ACTION & POSE DISCRETE TAXONOMY

| Action ID ($a$) | Action Name | Valid Pose IDs ($p$) | Semantic Meaning |
|---|---|---|---|
| `0` | Idle | `0` | Neutral standing stance |
| `1` | Walk | `0` to `8` | 9-frame complete walk cycle |
| `2` | Slash Attack | `0` to `7` | 8-frame sword strike cycle |
| `3` | Spellcast | `0` to `5` | 6-frame magic casting cycle |

This discrete formulation completely eliminates ambiguity: given reference $x_{\text{ref}}$, specifying $(a=1, p=3)$ unambiguously queries the Walk Left Apex stride, while $(a=2, p=3)$ queries the Attack Strike Apex.
