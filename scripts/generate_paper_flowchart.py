import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Set up canvas for publication-grade crisp flowchart
fig, ax = plt.subplots(figsize=(14, 14), dpi=300)
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis('off')

# Style configuration matching IEEE / Springer paper standards
box_face = "#ffffff"
box_edge = "#000000"
text_color = "#000000"

def draw_box(x, y, w, h, text, fontsize=9.5, bold=False, sub_bg=box_face):
    rect = patches.FancyBboxPatch((x - w/2, y - h/2), w, h,
                                  boxstyle="square,pad=0.0",
                                  linewidth=1.4, edgecolor=box_edge,
                                  facecolor=sub_bg, zorder=3)
    ax.add_patch(rect)
    weight = 'bold' if bold else 'normal'
    ax.text(x, y, text, ha='center', va='center', fontsize=fontsize,
            fontfamily='DejaVu Sans', weight=weight, color=text_color, zorder=4)

def draw_arrow(x1, y1, x2, y2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color="black", lw=1.3, mutation_scale=14),
                zorder=2)

def draw_polyline(points):
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    ax.plot(xs, ys, color="black", lw=1.3, zorder=2)
    # arrow at the final segment
    ax.annotate("", xy=points[-1], xytext=points[-2],
                arrowprops=dict(arrowstyle="-|>", color="black", lw=1.3, mutation_scale=14),
                zorder=2)

# ==================== NODES LAYOUT ====================
# Level 0: Top Input
top_x, top_y = 50, 93
draw_box(top_x, top_y, 28, 6.2, "Input: Free-Form Natural Language\nGame Concept / Pitch", bold=True)

# Level 1: Four Main Modules
col1_x = 13    # Module 1: AE / VAE
col2_x = 38    # Module 2: Transformer
col3_x = 63    # Module 3: DCGAN
col4_x = 87    # Module 4: Diffusion

mod_y = 80
w_mod = 19
h_mod = 5.8

draw_box(col1_x, mod_y, w_mod, h_mod, "Module 1:\nAutoencoder / VAE", bold=True)
draw_box(col2_x, mod_y, w_mod, h_mod, "Module 2:\nTransformer Planner", bold=True)
draw_box(col3_x, mod_y, w_mod, h_mod, "Module 3:\nDCGAN (169K Characters)", bold=True)
draw_box(col4_x, mod_y, w_mod, h_mod, "Module 4:\nDiffusion Pose Engine", bold=True)

# Top Bus routing to 4 modules
bus_y = 86.5
# from top box to bus
ax.plot([top_x, top_x], [top_y - 3.1, bus_y], color="black", lw=1.3)
# horizontal bus
ax.plot([col1_x, col4_x], [bus_y, bus_y], color="black", lw=1.3)
# drop arrows to each module
draw_arrow(col1_x, bus_y, col1_x, mod_y + h_mod/2)
draw_arrow(col2_x, bus_y, col2_x, mod_y + h_mod/2)
draw_arrow(col3_x, bus_y, col3_x, mod_y + h_mod/2)
draw_arrow(col4_x, bus_y, col4_x, mod_y + h_mod/2)

# ==================== COLUMN 1: AE / VAE ====================
c1_y1 = 66
c1_y2 = 51
c1_y3 = 36
w_sub = 19
h_sub = 6.2

draw_box(col1_x, c1_y1, w_sub, h_sub, "Sprite Feature Encoding &\nLatent Vector Extraction")
draw_box(col1_x, c1_y2, w_sub, h_sub, "Automated QA Outlier Screening\n(MSE > P95 Threshold 0.001313)")
draw_box(col1_x, c1_y3, w_sub, h_sub, "Continuous Latent Search &\nSmooth SLERP Character Morphing")

draw_arrow(col1_x, mod_y - h_mod/2, col1_x, c1_y1 + h_sub/2)
draw_arrow(col1_x, c1_y1 - h_sub/2, col1_x, c1_y2 + h_sub/2)
draw_arrow(col1_x, c1_y2 - h_sub/2, col1_x, c1_y3 + h_sub/2)

# ==================== COLUMN 2: TRANSFORMER ====================
c2_y1 = 66
c2_y2 = 51
c2_y3 = 36

draw_box(col2_x, c2_y1, w_sub, h_sub, "Semantic Tokenization &\nIntent Disambiguation")
draw_box(col2_x, c2_y2, w_sub, h_sub, "Game Specification Structuring\n(Hero, Enemies, Biomes, Props)")
draw_box(col2_x, c2_y3, w_sub, h_sub, "Structured Game Manifest JSON &\nAttribute Conditioning Vectors")

draw_arrow(col2_x, mod_y - h_mod/2, col2_x, c2_y1 + h_sub/2)
draw_arrow(col2_x, c2_y1 - h_sub/2, col2_x, c2_y2 + h_sub/2)
draw_arrow(col2_x, c2_y2 - h_sub/2, col2_x, c2_y3 + h_sub/2)

# ==================== COLUMN 3: DCGAN ====================
c3_y1 = 66
c3_y2 = 51
c3_y3 = 36

draw_box(col3_x, c3_y1, w_sub, h_sub, "Adversarial Prior Modeling\n(169K Alucard Characters)")
draw_box(col3_x, c3_y2, w_sub, h_sub, "Unconditional Novel Sprite Synthesis\nfrom Latent Noise z ~ N(0, I)")
draw_box(col3_x, c3_y3, w_sub, h_sub, "Canonical Idle Base Sprites\n(64x64 RGBA High-Fidelity)")

draw_arrow(col3_x, mod_y - h_mod/2, col3_x, c3_y1 + h_sub/2)
draw_arrow(col3_x, c3_y1 - h_sub/2, col3_x, c3_y2 + h_sub/2)
draw_arrow(col3_x, c3_y2 - h_sub/2, col3_x, c3_y3 + h_sub/2)

# ==================== COLUMN 4: DIFFUSION (POSE GENERATION) ====================
c4_y1 = 66
c4_y2 = 51
c4_y3 = 36

draw_box(col4_x, c4_y1, w_sub, h_sub, "Action & Pose Conditioning Prompt\n(Walk, Attack, Cast Spell, Hurt)")
draw_box(col4_x, c4_y2, w_sub, h_sub, "Latent Pixel-Art Diffusion &\nDual-Stage Chroma Cutout (alpha=0)")
draw_box(col4_x, c4_y3, w_sub, h_sub, "Multi-Pose Action Sprite Sheet\n(Walk/Run Cycles & Animations)")

draw_arrow(col4_x, mod_y - h_mod/2, col4_x, c4_y1 + h_sub/2)
draw_arrow(col4_x, c4_y1 - h_sub/2, col4_x, c4_y2 + h_sub/2)
draw_arrow(col4_x, c4_y2 - h_sub/2, col4_x, c4_y3 + h_sub/2)

# Cross-Module Interaction Arrow:
# DCGAN Base Sprite -> Diffusion Pose Engine
# Col 3 -> Col 4 between level 3 boxes
ax.annotate("", xy=(col4_x - w_sub/2, c4_y2), xytext=(col3_x + w_sub/2, c3_y3),
            arrowprops=dict(arrowstyle="-|>", color="black", lw=1.3, mutation_scale=14, linestyle="--"),
            zorder=5)
ax.text((col3_x + col4_x)/2, (c3_y3 + c4_y2)/2 + 2.5, "Canonical Base\nSprite Guide",
        ha='center', va='center', fontsize=7.5, fontfamily='DejaVu Sans', style='italic',
        bbox=dict(boxstyle="square,pad=0.2", fc="white", ec="none"))

# ==================== BOTTOM DESTINATION ====================
bot_x, bot_y = 50, 11
w_bot, h_bot = 40, 7.5
draw_box(bot_x, bot_y, w_bot, h_bot,
         "Output: Production Game Asset Pack\n- Structured Game Spec Sheet (JSON)\n- Screened & QA-Verified Sprites (AE/VAE)\n- Novel Generated Characters (DCGAN)\n- Action Sprite Sheets & Pose Cycles (Diffusion)",
         bold=True, fontsize=9.2)

# Bottom Collection Bus
bus_bot_y = 22

# Drop lines from 4 columns to bottom bus
# Col 1: from c1_y3 - h_sub/2 to bus_bot_y
ax.plot([col1_x, col1_x], [c1_y3 - h_sub/2, bus_bot_y], color="black", lw=1.3)
# Col 2: from c2_y3 - h_sub/2 to bus_bot_y
ax.plot([col2_x, col2_x], [c2_y3 - h_sub/2, bus_bot_y], color="black", lw=1.3)
# Col 3: from c3_y3 - h_sub/2 to bus_bot_y
ax.plot([col3_x, col3_x], [c3_y3 - h_sub/2, bus_bot_y], color="black", lw=1.3)
# Col 4: from c4_y3 - h_sub/2 to bus_bot_y
ax.plot([col4_x, col4_x], [c4_y3 - h_sub/2, bus_bot_y], color="black", lw=1.3)

# Horizontal bottom bus from Col 1 to Col 4
ax.plot([col1_x, col4_x], [bus_bot_y, bus_bot_y], color="black", lw=1.3)

# Arrow from bottom bus center into Output Box
draw_arrow(bot_x, bus_bot_y, bot_x, bot_y + h_bot/2)

plt.tight_layout()
output_path = r"research_paper_assets\gameforge_system_architecture_flowchart.png"
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f"Flowchart successfully generated and saved to {output_path}")
