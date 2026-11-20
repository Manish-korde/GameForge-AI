import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['font.family'] = 'sans-serif'

os.makedirs("reports/poster_figures", exist_ok=True)
brain_dir = r"C:\Users\manis\.gemini\antigravity\brain\0e75ca3f-2ae4-4b95-8ef8-c8e5292ad012"

models = [
    '1. Conv AE\n(QA Screener)',
    '2. Beta-VAE\n(Latent Retrieval)',
    '3. Discrete DCGAN\n(Asset Synthesizer)',
    '4. Flan-T5\n(Semantic Planner)',
    '5. Velocity Diffusion\n(Motion Animator)'
]

dimensions = [
    'Reconstruction\nFidelity (PSNR/SSIM)',
    'Latent Manifold\n& Retrieval (k-NN)',
    'Novel Asset\nSynthesis (Coherence)',
    'Semantic Concept\nAdherence (Schema)',
    'Kinematic Motion\n& Animation (IoU)'
]

# Matrix of scores (Models x Dimensions)
data_matrix = np.array([
    [98.5, 75.0, 15.0, 10.0, 10.0],  # AE
    [76.0, 94.0, 35.0, 10.0, 15.0],  # VAE
    [20.0, 30.0, 92.5, 20.0, 10.0],  # DCGAN
    [10.0, 20.0, 15.0, 98.0, 10.0],  # Flan-T5
    [65.0, 45.0, 70.0, 40.0, 95.6]   # Velocity Diffusion
])

colors = ['#2b5c8f', '#2ca02c', '#ff7f0e', '#9467bd', '#d62728']

# ==============================================================================
# VARIANT A: GROUPED BAR CHART (Clean, Direct, Highly Readable on Posters)
# ==============================================================================
print("Generating Variant A: Grouped Bar Chart...")
fig_a, ax_a = plt.subplots(figsize=(11.5, 6.0), dpi=300, facecolor='#ffffff')

x = np.arange(len(dimensions))
width = 0.15

for i in range(len(models)):
    offset = (i - 2) * width
    rects = ax_a.bar(x + offset, data_matrix[i], width, label=models[i].replace('\n', ' '), 
                     color=colors[i], edgecolor='#222222', linewidth=0.9, zorder=3)

ax_a.set_ylabel('Capability Benchmark Score (0 - 100%)', fontsize=11, fontweight='bold')
ax_a.set_title('GameForge AI: Multi-Model Functional Specialization & Capability Comparison\n(Grouped Metric View)', 
               fontsize=13, fontweight='bold', pad=15)
ax_a.set_xticks(x)
ax_a.set_xticklabels(dimensions, fontsize=10, fontweight='bold')
ax_a.set_ylim(0, 115)
ax_a.grid(axis='y', linestyle='--', alpha=0.6, zorder=0)
ax_a.legend(loc='upper right', frameon=True, facecolor='#fbfbfb', edgecolor='#cccccc', fontsize=8.5, ncol=2)

plt.tight_layout()
p_a = "reports/poster_figures/alt_1A_grouped_bar_chart.png"
fig_a.savefig(p_a, bbox_inches='tight')
fig_a.savefig(os.path.join(brain_dir, "alt_1A_grouped_bar_chart.png"), bbox_inches='tight')
plt.close(fig_a)

# ==============================================================================
# VARIANT B: ANNOTATED HEATMAP (High-End Academic Paper / Executive Dashboard)
# ==============================================================================
print("Generating Variant B: Annotated Heatmap...")
fig_b, ax_b = plt.subplots(figsize=(9.5, 5.5), dpi=300, facecolor='#ffffff')

sns.heatmap(data_matrix, annot=True, fmt=".1f", cmap="YlGnBu", cbar_kws={'label': 'Normalized Performance Score (%)'},
            xticklabels=[d.replace('\n', ' ') for d in dimensions],
            yticklabels=[m.replace('\n', ' ') for m in models],
            linewidths=1.2, linecolor='white', ax=ax_b, annot_kws={"fontsize": 11, "fontweight": "bold"})

ax_b.set_title('GameForge AI: Specialization Heatmap Across Operational Dimensions\n(Darker = Stronger Specialization)', 
               fontsize=12, fontweight='bold', pad=15)
ax_b.set_xticklabels(ax_b.get_xticklabels(), rotation=20, ha='right', fontsize=9.5, fontweight='bold')
ax_b.set_yticklabels(ax_b.get_yticklabels(), rotation=0, fontsize=9.5, fontweight='bold')

plt.tight_layout()
p_b = "reports/poster_figures/alt_1B_specialization_heatmap.png"
fig_b.savefig(p_b, bbox_inches='tight')
fig_b.savefig(os.path.join(brain_dir, "alt_1B_specialization_heatmap.png"), bbox_inches='tight')
plt.close(fig_b)

# ==============================================================================
# VARIANT C: 5-PANEL PARALLEL SUBPLOT (Small Multiples / Zero Clutter)
# ==============================================================================
print("Generating Variant C: 5-Panel Small Multiples...")
fig_c, axes = plt.subplots(1, 5, figsize=(15.5, 4.2), sharey=True, dpi=300, facecolor='#ffffff')

dim_short = ['Reconstruction', 'Latent/Search', 'Generation', 'Planning', 'Motion']

for i in range(5):
    ax = axes[i]
    bars = ax.bar(dim_short, data_matrix[i], color=colors[i], edgecolor='#222222', linewidth=1.1, width=0.6, zorder=3)
    ax.set_title(models[i], fontsize=10, fontweight='bold', pad=10, color=colors[i])
    ax.set_ylim(0, 115)
    ax.grid(axis='y', linestyle='--', alpha=0.6, zorder=0)
    ax.set_xticklabels(dim_short, rotation=45, ha='right', fontsize=8.5, fontweight='bold')
    
    # Highlight top capability with star/badge
    top_idx = np.argmax(data_matrix[i])
    top_val = data_matrix[i][top_idx]
    ax.text(top_idx, top_val + 3.0, f"{top_val:.1f}%", ha='center', va='bottom', fontsize=8, fontweight='bold')

axes[0].set_ylabel('Capability Score (%)', fontsize=10, fontweight='bold')
fig_c.suptitle('GameForge AI: Individual Model Functional Signatures (Small Multiples View)', 
               fontsize=13, fontweight='bold', y=1.05)

plt.tight_layout()
p_c = "reports/poster_figures/alt_1C_small_multiples_panels.png"
fig_c.savefig(p_c, bbox_inches='tight')
fig_c.savefig(os.path.join(brain_dir, "alt_1C_small_multiples_panels.png"), bbox_inches='tight')
plt.close(fig_c)

print("[SUCCESS] All 3 alternative graphs generated successfully!")

