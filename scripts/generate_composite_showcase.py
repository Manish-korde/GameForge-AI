import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image
import os
import pandas as pd

paper_dir = r"research_paper_assets\06_sprite_deformation_and_pose_animation"
artifact_dir = r"C:\Users\manis\.gemini\antigravity-ide\brain\fdd8ce72-7172-4c22-b928-f5cab2ad951c"

real_img = Image.open(os.path.join(paper_dir, "diffusion_real_training_sprites.png"))
attack_sheet = Image.open(os.path.join(paper_dir, "diffusion_attack_contact_sheet.png"))
walk_sheet = Image.open(os.path.join(paper_dir, "diffusion_walk_contact_sheet.png"))
df_loss = pd.read_csv(os.path.join(paper_dir, "diffusion_quality_fix_150ep_loss_history.csv"))

fig = plt.figure(figsize=(16, 12), dpi=200)
gs = gridspec.GridSpec(3, 2, height_ratios=[1.1, 1.3, 1.3], width_ratios=[1.2, 1.0])

# Top Left: Real Ground Truth LPC Sprites
ax0 = fig.add_subplot(gs[0, 0])
ax0.imshow(real_img)
ax0.set_title("(a) Ground-Truth Training Sprites: Centered Liberated Pixel Cup (LPC) Archetypes", fontsize=11, fontweight="bold", pad=8)
ax0.axis("off")

# Top Right: Loss Convergence
ax1 = fig.add_subplot(gs[0, 1])
ax1.plot(df_loss["epoch"], df_loss["train_loss"], label="Training Loss", color="#1f77b4", lw=2)
ax1.plot(df_loss["epoch"], df_loss["val_loss"], label="Validation Loss", color="#d62728", lw=2)
best_row = df_loss.loc[df_loss["val_loss"].idxmin()]
ax1.scatter([best_row["epoch"]], [best_row["val_loss"]], color="#2ca02c", s=80, zorder=5, label=f"Best Val ({best_row['val_loss']:.4f})")
ax1.set_title("(b) 150-Epoch DDPM Loss Convergence (Best Epoch 145)", fontsize=11, fontweight="bold", pad=8)
ax1.set_xlabel("Epoch", fontsize=10)
ax1.set_ylabel("Diffusion Loss (MSE / L1)", fontsize=10)
ax1.grid(True, linestyle="--", alpha=0.5)
ax1.legend(loc="upper right", frameon=True, fontsize=9)

# Middle: Generated Attack Contact Sheet
ax2 = fig.add_subplot(gs[1, :])
ax2.imshow(attack_sheet)
ax2.set_title("(c) Synthesized Action 0 (Attack): 8 Denoised Combat Stance Sprites (Foreground-Centered)", fontsize=11, fontweight="bold", pad=8)
ax2.axis("off")

# Bottom: Generated Walk Contact Sheet
ax3 = fig.add_subplot(gs[2, :])
ax3.imshow(walk_sheet)
ax3.set_title("(d) Synthesized Action 1 (Walk): 8 Denoised Locomotion Stride Sprites (Foreground-Centered)", fontsize=11, fontweight="bold", pad=8)
ax3.axis("off")

plt.suptitle("GameForge AI Module 4: Empirical Architecture, Convergence Dynamics, and Generated Multi-Pose Sprites", fontsize=13, fontweight="bold", y=0.995)
plt.tight_layout()

out_path = os.path.join(paper_dir, "gameforge_diffusion_module4_empirical_showcase.png")
fig.savefig(out_path, bbox_inches="tight", dpi=200)
art_path = os.path.join(artifact_dir, "gameforge_diffusion_module4_empirical_showcase.png")
fig.savefig(art_path, bbox_inches="tight", dpi=200)
plt.close()
print("Composite showcase generated successfully at:", out_path)
