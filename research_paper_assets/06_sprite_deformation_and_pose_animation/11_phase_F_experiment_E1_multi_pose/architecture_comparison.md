# RUNG 2 PHASE F: ARCHITECTURE COMPARISON & PARAMETER AUDIT
## Refined C2 vs. Multi-Pose Experiment E1

| Model | Conditioning Vectors | Trainable Parameters | Transferred From C2 | Newly Initialized |
|---|---|---|---|---|
| **Refined C2 Baseline** | $t$ (Timestep) + $a$ (Action: 2) | 1,986,020 | Baseline | 0 |
| **Experiment E1 (Multi-Pose)** | $t$ (Timestep) + $a$ (Action: 2) + $p$ (Pose: 8) | **1,987,044** | **1,986,020 (99.95%)** | **+1,024 (0.05%)** |

### Conditioning Equation:
$$c(t, a, p) = \text{MLP}_{\text{time}}(\text{PE}(t)) + \mathbf{E}_{\text{act}}[a] + \mathbf{E}_{\text{pose}}[p], \quad c \in \mathbb{R}^{128}$$

All convolutional feature dimensions, normalization statistics, residual bottlenecks, and reverse DDIM update steps remain 100% identical.
