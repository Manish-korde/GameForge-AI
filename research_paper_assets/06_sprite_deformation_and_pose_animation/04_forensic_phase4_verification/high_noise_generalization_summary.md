# Test 3: High-Noise Generalization Across 100 Independent Seeds

## 1. Statistical Stability Audit (100 Seeds vs 16 Seeds)
Across 100 independent random noise vectors $\epsilon \sim \mathcal{N}(0, \mathbf{I})$:
- At $t=980$: Mean $\epsilon$-MSE is **0.06292** (Median: **0.06278**, P95: **0.06643**).
- At $t=750$: Mean $\epsilon$-MSE is **0.05334** (Median: **0.0533**).

## 2. Generalization Finding
The 100-seed sample confirms statistical stability:
1. Standard deviation across seeds is extremely small ($\sigma pprox 0.005$), proving the model does not suffer from high seed variance.
2. The network predicts noise with consistent average error ($pprox 0.05 - 0.06$) across the entire noise sphere.
3. However, because $\frac{1 - \bar{\alpha}}{\bar{\alpha}}$ equals $1,140$ at $t=980$, that consistent $0.062$ noise error produces an unclipped $x_0$ MSE of **71.7**.
4. Clamping to $[-1, 1]$ artificially squashes the error to **0.9156**, creating a false visual impression of bounded error while destroying the signal into static.
