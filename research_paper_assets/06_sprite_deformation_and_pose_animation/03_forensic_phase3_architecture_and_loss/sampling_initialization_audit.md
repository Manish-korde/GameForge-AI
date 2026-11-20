# Test 4: Sampling Initialization & Terminal Timestep Audit

## 1. Numerical Timestep Properties
| Quantity | Timestep t = 980 | Timestep t = 999 | Theoretical t = 1000 |
|---|---|---|---|
| **$\bar{\alpha}_t$** | **`8.76188744e-04`** | **`2.42838999e-10`** | `0.00000000` |
| **$\sqrt{\bar{\alpha}_t}$ (Signal Scale)** | **`0.02960049`** | **`0.00001558`** | `0.00000000` |
| **$\sqrt{1 - \bar{\alpha}_t}$ (Noise Scale)** | **`0.99956181`** | **`1.00000000`** | `1.00000000` |

## 2. Schedule & Boundary Audit Findings
1. **Initial Reverse Timestep Mismatch:**  
   In DDIM sampling with `step_ratio = 1000 // 50 = 20`, the timesteps evaluated are `list(range(0, 1000, 20)) = [0, 20, ..., 980]`.  
   The reverse trajectory therefore begins at **$t = 980$**, NOT $t = 999$ or $t = 1000$.
2. **Signal Component at $t = 980$:**  
   At $t = 980$, $\sqrt{\bar{\alpha}_{980}} = 0.03061$. The forward-diffused state actually contains **$3.06\%$ ground truth signal** and **$99.95\%$ noise**.
3. **The Gaussian Init Discrepancy:**  
   When the sampler initializes from pure Gaussian noise $\mathbf{x}_T \sim \mathcal{N}(0, \mathbf{I})$, it is mathematically assuming $\sqrt{\bar{\alpha}} = 0.0$. But it immediately passes $t = 980$ to the network!  
   At $t = 980$, the network expects $\mathbf{x}_{980} = 0.0306 \cdot x_0 + 0.9995 \cdot \epsilon$, but receives pure $\epsilon$ with zero $x_0$.
4. **Empirical Comparison of Initializations (Same Checkpoint & Seed):**
   - **Init A (Standard Pure Gaussian Noise):** PSNR = **6.52 dB**, SSIM = **0.0945** (Diverges into static).
   - **Init B (Matched Forward-Diffused at $t = 980$):** PSNR = **6.60 dB**, SSIM = **0.1023**.
