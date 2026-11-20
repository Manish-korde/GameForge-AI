# Test 5: Training Objective & Loss Formulation Audit

## 1. Pixel Distribution & Effective Loss Contribution
- **Total Canvas Pixels:** 4096 ($64 \times 64$)
- **Foreground Pixels (Character Mass):** 1007 (24.58%)
- **Background Pixels (Transparent Canvas):** 3089 (75.42%)
- **Weighting Applied in Script:** $W_{\text{fg}} = 3.0$, $W_{\text{bg}} = 1.0$
- **Total Weighted Mass:** 6110.0
- **Foreground Effective Contribution to Loss:** **49.44%**
- **Background Effective Contribution to Loss:** **50.56%**

## 2. Key Findings on Objective Design
1. **Background Still Dominates the Gradients:**  
   Because background accounts for 75.4% of the spatial canvas, even with a $3.0\times$ multiplier on foreground pixels, the background still accounts for **50.6% of the total loss**. The model can reduce its loss substantially simply by predicting background noise accurately.
2. **Uniform Timestep Sampling:**  
   `t = torch.randint(0, TIMESTEPS, (batch_size,))` assigns equal probability ($0.1\%$) to every timestep.  
   Timesteps $t > 800$ receive only $20\%$ of all gradient steps, yet require orders of magnitude higher numerical precision to prevent trajectory divergence.
3. **Loss Definition in Training vs Evaluation:**  
   During training, the reported scalar was the weighted MSE: $\mathcal{L} = \frac{1}{4HW} \sum W \cdot (\epsilon - \hat{\epsilon})^2$.  
   In unweighted evaluation, $\text{MSE} = \frac{1}{4HW} \sum (\epsilon - \hat{\epsilon})^2$.
