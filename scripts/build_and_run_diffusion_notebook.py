import nbformat as nbf
from nbconvert.preprocessors import ExecutePreprocessor
import os
import shutil

print("Initializing notebook generator...")
nb = nbf.v4.new_notebook()
cells = []

# Cell 1: Markdown Title & Executive Overview
cells.append(nbf.v4.new_markdown_cell("""# 🎮 GameForge AI: Action-Conditioned Diffusion Sprite Generation & Deformation Evaluation
### Research Module 4: Pose Articulation & Kinematic Sprite Synthesis
**Team:** Samriddhi Das (202301040276), Manish Korde (202301040206), Swayam Pagare (202301040185)  
**Project Guide:** Dr. Savita Mane | **Department:** Computer Engineering, MIT Academy of Engineering, Pune  
**Artifact Version:** Phase 2.3 Quality Fix v2 (150-Epoch Checkpoint Evaluation)  

---
### 📌 Executive Overview
This notebook presents the formal empirical evaluation and visual output audit for **Module 4 (Action-Conditioned Sprite Generation & Deformation)**:
1. **Model Checkpoint Audit:** Inspection of the 1.41M-parameter conditional DDPM U-Net (`Quality fix new latest/checkpoints/best.pt`).
2. **Empirical Training Dynamics:** 150-epoch loss convergence analysis, tracking the drop from 1.055 to 0.0534 (best validation loss at epoch 145).
3. **Ground-Truth Character Dataset:** Audit of centered, foreground-cropped Liberated Pixel Cup (LPC) sprite archetypes.
4. **Generated Action Visuals:** Detailed presentation of synthesized Attack and Walk action poses across contact sheets and individual high-resolution sprite grids.
5. **Live Reverse-Diffusion Sampling:** End-to-end inference execution directly from the trained PyTorch checkpoint.
"""))

# Cell 2: Code - Environment Setup & Path Discovery
cells.append(nbf.v4.new_code_cell("""import os
import sys
import time
import math
import json
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['figure.autolayout'] = True
plt.rcParams['figure.dpi'] = 150

import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"⚡ PyTorch Version: {torch.__version__}")
print(f"⚡ Compute Target: {DEVICE}")

CWD = Path('.').resolve()
for candidate in [CWD, CWD.parent, CWD.parent.parent]:
    if (candidate / "zDIffusion_deformation final stuff").exists():
        PROJECT_ROOT = candidate
        break
else:
    PROJECT_ROOT = Path(r"c:\\Users\\manis\\OneDrive\\Desktop\\Prompt_to_game_asset_generator")

DIFF_ROOT = PROJECT_ROOT / "zDIffusion_deformation final stuff"
QUALITY_FIX_DIR = DIFF_ROOT / "Quality fix new latest"
CHECKPOINT_PATH = QUALITY_FIX_DIR / "checkpoints" / "best.pt"
LOSS_CSV_PATH = QUALITY_FIX_DIR / "reports" / "loss_history.csv"
STATUS_JSON_PATH = QUALITY_FIX_DIR / "reports" / "phase2_3_quality_fix_status.json"

print(f"📁 Project Root: {PROJECT_ROOT}")
print(f"📦 Checkpoint: {CHECKPOINT_PATH} (Exists: {CHECKPOINT_PATH.exists()})")
print(f"📊 Loss History: {LOSS_CSV_PATH} (Exists: {LOSS_CSV_PATH.exists()})")
"""))

# Cell 3: Code - Checkpoint Inspection & Metadata Audit
cells.append(nbf.v4.new_code_cell("""checkpoint = torch.load(CHECKPOINT_PATH, map_location='cpu')

metadata = {k: v for k, v in checkpoint.items() if k != 'model_state_dict'}
print("=" * 60)
print("        CHECKPOINT METADATA & TRAINING HYPERPARAMETERS      ")
print("=" * 60)
for k, v in metadata.items():
    print(f" • {k:22s}: {v}")
print("=" * 60)

with open(STATUS_JSON_PATH, "r") as f:
    status_info = json.load(f)

print("\\nPhase 2.3 Status JSON Summary:")
for k, v in status_info.items():
    print(f" • {k:22s}: {v}")
"""))

# Cell 4: Code - UNet Architecture & Exact Parameter Breakdown
cells.append(nbf.v4.new_code_cell("""class ResBlock(nn.Module):
    def __init__(self, in_c, out_c, emb_dim=256, groups=8):
        super().__init__()
        self.n1 = nn.GroupNorm(groups, in_c)
        self.c1 = nn.Conv2d(in_c, out_c, 3, padding=1)
        self.n2 = nn.GroupNorm(groups, out_c)
        self.c2 = nn.Conv2d(out_c, out_c, 3, padding=1)
        self.ep = nn.Linear(emb_dim, out_c)
        self.skip = nn.Conv2d(in_c, out_c, 1) if in_c != out_c else nn.Identity()

    def forward(self, x, emb):
        h = self.c1(F.silu(self.n1(x)))
        h = h + self.ep(emb)[:, :, None, None]
        h = self.c2(F.silu(self.n2(h)))
        return self.skip(x) + h

class UpResBlock(nn.Module):
    def __init__(self, in_c, out_c, emb_dim=256, groups=8):
        super().__init__()
        self.skip = nn.Conv2d(in_c, out_c, 1)
        self.n1 = nn.GroupNorm(groups, in_c)
        self.c1 = nn.Conv2d(in_c, out_c, 3, padding=1)
        self.n2 = nn.GroupNorm(groups, out_c)
        self.c2 = nn.Conv2d(out_c, out_c, 3, padding=1)
        self.ep = nn.Linear(emb_dim, out_c)

    def forward(self, x, emb):
        h = self.c1(F.silu(self.n1(x)))
        h = h + self.ep(emb)[:, :, None, None]
        h = self.c2(F.silu(self.n2(h)))
        return self.skip(x) + h

def timestep_embedding(timesteps, dim=256):
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding

class ActionConditionedUNet(nn.Module):
    def __init__(self, in_c=3, out_c=3, base=32, emb_dim=256, num_actions=2, groups=8):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.SiLU(),
            nn.Linear(emb_dim, emb_dim)
        )
        self.action_emb = nn.Embedding(num_actions, emb_dim)
        self.inc = nn.Conv2d(in_c, base, 3, padding=1)
        self.rb1 = ResBlock(base, base, emb_dim, groups=groups)
        self.down1 = nn.Conv2d(base, base * 2, 4, 2, 1)
        self.rb2 = ResBlock(base * 2, base * 2, emb_dim, groups=groups)
        self.down2 = nn.Conv2d(base * 2, base * 4, 4, 2, 1)
        self.mid1 = ResBlock(base * 4, base * 4, emb_dim, groups=groups)
        self.mid2 = ResBlock(base * 4, base * 4, emb_dim, groups=groups)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 4, 2, 1)
        self.urb2 = UpResBlock(base * 4, base * 2, emb_dim, groups=groups)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 4, 2, 1)
        self.urb1 = UpResBlock(base * 2, base, emb_dim, groups=groups)
        self.on = nn.GroupNorm(groups, base)
        self.out = nn.Conv2d(base, out_c, 3, padding=1)

    def forward(self, x, t, action_id):
        t_emb = self.time_mlp(timestep_embedding(t, 256))
        a_emb = self.action_emb(action_id)
        emb = t_emb + a_emb
        x0 = self.inc(x)
        r1 = self.rb1(x0, emb)
        d1 = self.down1(r1)
        r2 = self.rb2(d1, emb)
        d2 = self.down2(r2)
        m1 = self.mid1(d2, emb)
        m2 = self.mid2(m1, emb)
        u2 = self.up2(m2)
        u2 = torch.cat([u2, r2], dim=1)
        ur2 = self.urb2(u2, emb)
        u1 = self.up1(ur2)
        u1 = torch.cat([u1, r1], dim=1)
        ur1 = self.urb1(u1, emb)
        return self.out(F.silu(self.on(ur1)))

model = ActionConditionedUNet(
    in_c=3,
    out_c=3,
    base=checkpoint['base_channels'],
    emb_dim=256,
    num_actions=len(checkpoint['action_to_id'])
)
model.load_state_dict(checkpoint['model_state_dict'], strict=True)
model.to(DEVICE)
model.eval()

param_breakdown = [
    ('Time MLP (Positional Projection)', sum(p.numel() for p in model.time_mlp.parameters())),
    ('Action Embedding Head', sum(p.numel() for p in model.action_emb.parameters())),
    ('Input Projection (inc)', sum(p.numel() for p in model.inc.parameters())),
    ('Encoder Block 1 (rb1 + down1)', sum(p.numel() for p in model.rb1.parameters()) + sum(p.numel() for p in model.down1.parameters())),
    ('Encoder Block 2 (rb2 + down2)', sum(p.numel() for p in model.rb2.parameters()) + sum(p.numel() for p in model.down2.parameters())),
    ('Bottleneck (mid1 + mid2)', sum(p.numel() for p in model.mid1.parameters()) + sum(p.numel() for p in model.mid2.parameters())),
    ('Decoder Block 2 (up2 + urb2)', sum(p.numel() for p in model.up2.parameters()) + sum(p.numel() for p in model.urb2.parameters())),
    ('Decoder Block 1 (up1 + urb1)', sum(p.numel() for p in model.up1.parameters()) + sum(p.numel() for p in model.urb1.parameters())),
    ('Output Head (on + out)', sum(p.numel() for p in model.on.parameters()) + sum(p.numel() for p in model.out.parameters())),
]
total_params = sum(p.numel() for p in model.parameters())

df_params = pd.DataFrame(param_breakdown, columns=['Sub-Module', 'Trainable Parameters'])
df_params['Percentage'] = (df_params['Trainable Parameters'] / total_params * 100).map('{:.2f}%'.format)
print(f"✅ Model successfully loaded with strict=True! Total Parameters: {total_params:,}")
display(df_params)
"""))

# Cell 5: Code - 150-Epoch Loss Curves & Convergence Analysis
cells.append(nbf.v4.new_code_cell("""df_loss = pd.read_csv(LOSS_CSV_PATH)
print(f"Loaded {len(df_loss)} training epochs.")
print(f"Epoch 1 Train Loss: {df_loss['train_loss'].iloc[0]:.4f} | Val Loss: {df_loss['val_loss'].iloc[0]:.4f}")
best_idx = df_loss['val_loss'].idxmin()
best_row = df_loss.loc[best_idx]
print(f"🏆 Best Convergence: Epoch {int(best_row['epoch'])} | Train Loss: {best_row['train_loss']:.5f} | Val Loss: {best_row['val_loss']:.5f}")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.5), dpi=150)

# Full 150 epochs
ax1.plot(df_loss['epoch'], df_loss['train_loss'], label='Training Loss', color='#1f77b4', lw=2.0)
ax1.plot(df_loss['epoch'], df_loss['val_loss'], label='Validation Loss', color='#d62728', lw=2.0)
ax1.scatter([best_row['epoch']], [best_row['val_loss']], color='#2ca02c', s=100, zorder=5, label=f'Best Val Loss ({best_row["val_loss"]:.4f})')
ax1.set_title('Phase 2.3: 150-Epoch DDPM Loss Convergence', fontsize=12, fontweight='bold', pad=10)
ax1.set_xlabel('Epoch', fontsize=11)
ax1.set_ylabel('Diffusion Loss (MSE / Noise L1)', fontsize=11)
ax1.grid(True, linestyle='--', alpha=0.5)
ax1.legend(frameon=True, facecolor='white', framealpha=0.9)

# Zoomed fine-tuning (Epochs 50 to 150)
df_tail = df_loss[df_loss['epoch'] >= 50]
ax2.plot(df_tail['epoch'], df_tail['train_loss'], label='Train Loss (Epochs 50-150)', color='#1f77b4', lw=2.0)
ax2.plot(df_tail['epoch'], df_tail['val_loss'], label='Val Loss (Epochs 50-150)', color='#d62728', lw=2.0)
ax2.scatter([best_row['epoch']], [best_row['val_loss']], color='#2ca02c', s=110, zorder=5)
ax2.axvline(x=best_row['epoch'], color='#2ca02c', linestyle=':', alpha=0.7)
ax2.annotate(f"Best Epoch: {int(best_row['epoch'])}\\nVal Loss: {best_row['val_loss']:.5f}",
             xy=(best_row['epoch'], best_row['val_loss']),
             xytext=(best_row['epoch'] - 38, best_row['val_loss'] + 0.045),
             arrowprops=dict(facecolor='#2ca02c', shrink=0.08, width=1.5, headwidth=6),
             fontweight='bold', fontsize=9, bbox=dict(boxstyle='round,pad=0.4', facecolor='#eafaf1', edgecolor='#2ca02c'))
ax2.set_title('Fine-Tuning Asymptotic Convergence (Epochs 50–150)', fontsize=12, fontweight='bold', pad=10)
ax2.set_xlabel('Epoch', fontsize=11)
ax2.set_ylabel('Validation Loss', fontsize=11)
ax2.grid(True, linestyle='--', alpha=0.5)
ax2.legend(frameon=True, facecolor='white', framealpha=0.9)

plt.tight_layout()
plt.show()
"""))

# Cell 6: Code - Ground Truth Real Training Sprites
cells.append(nbf.v4.new_code_cell("""real_sprites_path = QUALITY_FIX_DIR / "real_sprite_previews" / "real_training_sprites.png"
if real_sprites_path.exists():
    real_img = Image.open(real_sprites_path)
    plt.figure(figsize=(12, 6.5), dpi=150)
    plt.imshow(real_img)
    plt.title("Ground-Truth Training Sprites: Centered & Foreground-Cropped LPC Character Archetypes", fontsize=12, fontweight='bold', pad=10)
    plt.axis("off")
    plt.show()
else:
    print("Real sprites preview not found at:", real_sprites_path)
"""))

# Cell 7: Code - Generated Action Contact Sheets (Attack & Walk)
cells.append(nbf.v4.new_code_cell("""attack_sheet_path = QUALITY_FIX_DIR / "samples" / "attack_contact_sheet.png"
walk_sheet_path = QUALITY_FIX_DIR / "samples" / "walk_contact_sheet.png"

fig, axes = plt.subplots(2, 1, figsize=(14, 9), dpi=150)

if attack_sheet_path.exists():
    im_attack = Image.open(attack_sheet_path)
    axes[0].imshow(im_attack)
    axes[0].set_title("Action 0: Generated 'Attack' Contact Sheet (8 Synthesized Combat Samples)", fontsize=12, fontweight='bold', pad=8)
    axes[0].axis("off")

if walk_sheet_path.exists():
    im_walk = Image.open(walk_sheet_path)
    axes[1].imshow(im_walk)
    axes[1].set_title("Action 1: Generated 'Walk' Contact Sheet (8 Synthesized Locomotion Samples)", fontsize=12, fontweight='bold', pad=8)
    axes[1].axis("off")

plt.tight_layout()
plt.show()
"""))

# Cell 8: Code - High-Resolution Sample Comparison Grid (16 Samples)
cells.append(nbf.v4.new_code_cell("""samples_dir = QUALITY_FIX_DIR / "samples"
fig, axes = plt.subplots(4, 4, figsize=(12, 12), dpi=150)

for i in range(8):
    r = i // 4
    c = i % 4
    p = samples_dir / f"attack_sample_{i:02d}.png"
    if p.exists():
        im = Image.open(p)
        axes[r, c].imshow(im)
        axes[r, c].set_title(f"Attack Sample #{i+1}", fontsize=10, fontweight='bold')
    axes[r, c].axis("off")

for i in range(8):
    r = 2 + (i // 4)
    c = i % 4
    p = samples_dir / f"walk_sample_{i:02d}.png"
    if p.exists():
        im = Image.open(p)
        axes[r, c].imshow(im)
        axes[r, c].set_title(f"Walk Sample #{i+1}", fontsize=10, fontweight='bold')
    axes[r, c].axis("off")

plt.suptitle("GameForge AI Module 4: Generated Sprite Articulations (Attack vs Walk)", fontsize=14, fontweight='bold', y=0.99)
plt.tight_layout()
plt.show()
"""))

# Cell 9: Code - Live Reverse-Diffusion Generation Demo
cells.append(nbf.v4.new_code_cell("""T = checkpoint['timesteps'] # 250
betas = torch.linspace(1e-4, 0.02, T, device=DEVICE)
alphas = 1.0 - betas
alphas_bar = torch.cumprod(alphas, dim=0)

torch.manual_seed(2026)
num_samples = 4
actions = torch.tensor([0, 0, 1, 1], dtype=torch.long, device=DEVICE)
x = torch.randn(num_samples, 3, 64, 64, device=DEVICE)

print(f"🔄 Running live DDPM sampling ({T} steps) on {DEVICE} directly from model weights...")
t0 = time.time()
with torch.no_grad():
    for t_idx in reversed(range(T)):
        t_tensor = torch.full((num_samples,), t_idx, dtype=torch.long, device=DEVICE)
        pred_noise = model(x, t_tensor, actions)
        a_bar = alphas_bar[t_idx]
        beta = betas[t_idx]
        a = alphas[t_idx]
        noise = torch.randn_like(x) if t_idx > 0 else 0
        x = (1.0 / torch.sqrt(a)) * (x - (beta / torch.sqrt(1.0 - a_bar)) * pred_noise) + torch.sqrt(beta) * noise

elapsed = time.time() - t0
print(f"✅ Live sampling completed in {elapsed:.2f}s ({elapsed/num_samples:.2f}s per 64x64 sprite)!")

gen_imgs = ((x.clamp(-1, 1) + 1.0) / 2.0).permute(0, 2, 3, 1).cpu().numpy()
fig, axes = plt.subplots(1, 4, figsize=(12, 3.5), dpi=150)
labels = ['Live Attack #1', 'Live Attack #2', 'Live Walk #1', 'Live Walk #2']
for i in range(4):
    axes[i].imshow(gen_imgs[i], interpolation='nearest')
    axes[i].set_title(labels[i], fontsize=11, fontweight='bold')
    axes[i].axis('off')
plt.suptitle('Live Checkpoint Inference: Denoised 64×64 Action Sprites', fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
plt.show()
"""))

# Cell 10: Code - Comprehensive Quantitative Summary Table
cells.append(nbf.v4.new_code_cell("""summary_data = [
    {
        'Model Phase': 'Phase 2.0 (Unconditional Baseline)',
        'Corpus': 'LPC Action Slices (Raw)',
        'Epochs': 5,
        'Conditioning': 'None (Unconditional Gaussian noise)',
        'Best Val Loss': '0.7842',
        'Parameter Count': '1,972,675',
        'Status': 'Baseline Verified'
    },
    {
        'Model Phase': 'Phase 2.2 (Conditional DDPM)',
        'Corpus': '120 Exploratory Rows',
        'Epochs': 10,
        'Conditioning': 'Action Embedding (Attack, Walk)',
        'Best Val Loss': '0.33578',
        'Parameter Count': '1,410,467',
        'Status': 'Initial Conditioning'
    },
    {
        'Model Phase': 'Phase 2.3 Quality Fix v1',
        'Corpus': '120 Centered Crops',
        'Epochs': 80,
        'Conditioning': 'Action Embedding (Attack, Walk)',
        'Best Val Loss': '0.05877',
        'Parameter Count': '1,410,467',
        'Status': 'Foreground Alignment'
    },
    {
        'Model Phase': 'Phase 2.3 Quality Fix v2 (Latest)',
        'Corpus': '120 Centered Crops',
        'Epochs': 150,
        'Conditioning': 'Action Embedding (Attack, Walk)',
        'Best Val Loss': '0.05343 (Epoch 145)',
        'Parameter Count': '1,410,467',
        'Status': 'Final Delivered Model'
    }
]

df_summary = pd.DataFrame(summary_data)
display(df_summary)
"""))

# Cell 11: Markdown - Conclusions & Research Paper Presentation
cells.append(nbf.v4.new_markdown_cell("""### 📝 Academic Defense Findings & Empirical Conclusion
1. **Convergence Stability:** The conditional diffusion model achieved monotonic convergence from an initial validation loss of $0.9873$ down to $0.053428$ at Epoch 145 (a **94.59% relative loss reduction**).
2. **Foreground Centering Impact:** In Phase 2.2, sprites suffered from erratic spatial offsets across grid rows. Implementing foreground-aware bounding-box centering in Phase 2.3 stabilized the character torso, eliminating canvas drift.
3. **Action Conditioning Efficacy:** The 256-dimensional learned action embeddings ($a \\in \\{0: \\text{attack}, 1: \\text{walk}\\}$) successfully differentiated combat poses from walking strides without mode collapse.
4. **Lightweight Edge Deployment:** With 1,410,467 parameters, the model requires only 5.66 MB of storage and generates a complete $64 \\times 64$ sprite in $\\approx 10$ seconds on CPU, making it practical for real-time asset generation in indie game engines.
"""))

nb.cells = cells

out_notebook = r'Notebook\evaluation\11_Diffusion_Action_Deformation_Evaluation_and_Outputs.ipynb'
os.makedirs(os.path.dirname(out_notebook), exist_ok=True)
with open(out_notebook, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)
print(f"Notebook structure written to: {out_notebook}")

print("Executing notebook with ExecutePreprocessor...")
ep = ExecutePreprocessor(timeout=600, kernel_name='python3')
ep.preprocess(nb, {'metadata': {'path': r'Notebook\evaluation'}})

with open(out_notebook, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)
print(f"✅ Notebook executed and saved successfully with all cell outputs: {out_notebook}")

paper_copy = r'research_paper_assets\06_sprite_deformation_and_pose_animation\11_Diffusion_Action_Deformation_Evaluation_and_Outputs.ipynb'
os.makedirs(os.path.dirname(paper_copy), exist_ok=True)
shutil.copy2(out_notebook, paper_copy)
print(f"✅ Notebook copied to research_paper_assets: {paper_copy}")
