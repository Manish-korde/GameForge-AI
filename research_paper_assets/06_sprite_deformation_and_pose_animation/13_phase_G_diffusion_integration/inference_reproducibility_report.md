# Multi-Pose Velocity Diffusion Reproducibility Report

## Test Configuration
- **Model Checkpoint**: `phase_F_experiment_E1/checkpoint_best.pt`
- **Checkpoint SHA-256**: `10d33c5b2e7ad4d6f173f92fee2ee47101597bfadb1a677f6ee59d8bb7a1bc25`
- **Architecture**: `ActionAndPoseConditionedUNet` (1,987,044 parameters)
- **Reference Image**: `GameForge_AI_Phase2_audit/gameforge_phase2_1_audit/data/candidate_crops/action/action_row000000_r02_c00.png` (Character 00, Idle Reference)
- **Action Condition**: `walk` (Action Index 1)
- **Pose Condition**: `F3` (Pose Index 2 - Left Foot Apex)
- **Deterministic Seed**: `42`
- **DDIM Sampler Steps**: `50`
- **Scheduler**: Cosine $(\beta_1=0.0001, \beta_T=0.02, T=1000)$
- **Sampler Parameterization**: Velocity Formulation ($v_\theta$)

## Determinism Results
| Metric | Value | Threshold | Status |
| :--- | :--- | :--- | :--- |
| **Max Absolute Pixel Difference** ($L_\infty$) | **0.000000** | 0.000000 | PASS (Bit-Exact) |
| **Mean Absolute Pixel Difference** ($L_1$) | **0.000000** | 0.000000 | PASS (Bit-Exact) |
| **Pixel Match Percentage** | **100.00%** | 100.00% | PASS |
| **Numerical Consistency** | Deterministic PyTorch CPU Generator | Exact match across repeated invocations | PASS |

## Conclusion
The deployed multi-pose velocity diffusion service produces strictly bit-identical outputs when supplied with an identical seed, reference sprite, and condition tuple. Reverse DDIM sampling from pure Gaussian noise satisfies all reproducibility requirements.
