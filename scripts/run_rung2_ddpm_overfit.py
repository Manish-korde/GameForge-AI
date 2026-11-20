import os
import sys
import math
import time
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset

# Configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
OUTPUT_DIR = "research_paper_assets/06_sprite_deformation_and_pose_animation"
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODEL_SAVE_PATH = os.path.join(OUTPUT_DIR, "rung2_ddpm_overfit_best.pt")
LOSS_CSV_PATH = os.path.join(OUTPUT_DIR, "rung2_ddpm_loss_history.csv")
PREVIEW_GRID_PATH = os.path.join(OUTPUT_DIR, "rung2_ddpm_reverse_sampling_grid.png")
COMPARISON_4WAY_PATH = os.path.join(OUTPUT_DIR, "rung2_decisive_4way_milestone.png")

print("--- Rung 2: Reference-Conditioned DDPM Memorization & Reverse Sampling ---")
print(f"Device: {DEVICE}")

# 1. Dataset Streaming: 20 Verified Characters
print("Loading 20 verified character pairs...")
ds = load_dataset("carlosuperb/lpc-action-pixel-art-diffusion", split="train", streaming=True)

pairs_data = []
count = 0

for item in ds:
    sheet = item['image'].convert('RGBA')
    if sheet.size != (576, 768):
        sheet = sheet.resize((576, 768), Image.Resampling.NEAREST)
        
    ref_crop = sheet.crop((0 * 64, 2 * 64, 1 * 64, 3 * 64))
    walk_crop = sheet.crop((3 * 64, 2 * 64, 4 * 64, 3 * 64))
    
    if np.array(ref_crop)[:, :, 3].sum() < 500:
        continue
        
    # Scale RGBA strictly to [-1, 1]
    ref_arr = np.array(ref_crop, dtype=np.float32) / 127.5 - 1.0
    walk_arr = np.array(walk_crop, dtype=np.float32) / 127.5 - 1.0
    
    pairs_data.append({
        'id': f"C_{count:02d}",
        'ref_crop': ref_crop,
        'walk_crop': walk_crop,
        'ref_t': torch.tensor(ref_arr).permute(2, 0, 1).float(), # [4, 64, 64]
        'tgt_t': torch.tensor(walk_arr).permute(2, 0, 1).float()  # [4, 64, 64]
    })
    count += 1
    if count >= 20:
        break

print(f"Loaded {len(pairs_data)} pairs into memory.")

class DDPMDataset(Dataset):
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]['ref_t'], self.data[idx]['tgt_t']

train_loader = DataLoader(DDPMDataset(pairs_data), batch_size=20, shuffle=False)

# 2. Cosine Beta Schedule & DDPM Forward Math
TIMESTEPS = 1000

def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clamp(betas, 0.0001, 0.9999)

betas = cosine_beta_schedule(TIMESTEPS).to(DEVICE)
alphas = 1.0 - betas
alphas_cumprod = torch.cumprod(alphas, dim=0).to(DEVICE)
alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)

# 3. 8-Channel Reference-Conditioned U-Net
class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        return torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)

class Block(nn.Module):
    def __init__(self, in_c, out_c, time_emb_dim):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_c)
        self.conv1 = nn.Conv2d(in_c, out_c, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_c)
        self.silu = nn.SiLU()
    def forward(self, x, t):
        h = self.silu(self.norm1(self.conv1(x)))
        time_emb = self.silu(self.time_mlp(t))
        h = h + time_emb[(..., ) + (None, ) * 2]
        h = self.silu(self.norm2(self.conv2(h)))
        return h

class ReferenceConditionedUNet(nn.Module):
    def __init__(self, time_dim=128):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU()
        )
        # Input: 4 ch noisy target + 4 ch clean reference = 8 channels
        self.in_conv = nn.Conv2d(8, 32, 3, padding=1)
        
        # Down
        self.b1 = Block(32, 64, time_dim)
        self.down1 = nn.MaxPool2d(2)
        
        self.b2 = Block(64, 128, time_dim)
        self.down2 = nn.MaxPool2d(2)
        
        # Mid
        self.mid = Block(128, 256, time_dim)
        
        # Up
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.b_up2 = Block(256, 128, time_dim)
        
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.b_up1 = Block(128, 64, time_dim)
        
        # Out: Predicts 4 channels of noise epsilon
        self.out_conv = nn.Conv2d(64, 4, 3, padding=1)
        
    def forward(self, x_t, t, x_ref):
        # Concatenate noisy target and clean reference along channel dim
        x = torch.cat([x_t, x_ref], dim=1)
        t_emb = self.time_mlp(t)
        
        h0 = self.in_conv(x)
        h1 = self.b1(h0, t_emb)
        d1 = self.down1(h1)
        
        h2 = self.b2(d1, t_emb)
        d2 = self.down2(h2)
        
        m = self.mid(d2, t_emb)
        
        u2 = self.up2(m)
        cat2 = torch.cat([u2, h2], dim=1)
        dec2 = self.b_up2(cat2, t_emb)
        
        u1 = self.up1(dec2)
        cat1 = torch.cat([u1, h1], dim=1)
        dec1 = self.b_up1(cat1, t_emb)
        
        return self.out_conv(dec1)

model = ReferenceConditionedUNet().to(DEVICE)
param_count = sum(p.numel() for p in model.parameters())
print(f"Reference-Conditioned DDPM U-Net Initialized | Parameter Count: {param_count:,}")

optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)

# 4. Foreground-Weighted Training Loop
EPOCHS = 150
print(f"Beginning Rung 2 optimization ({EPOCHS} epochs, Foreground Weight = 3.0)...")

history = []
t0 = time.time()

for epoch in range(1, EPOCHS + 1):
    model.train()
    for ref_b, tgt_b in train_loader:
        ref_b = ref_b.to(DEVICE)
        tgt_b = tgt_b.to(DEVICE)
        batch_size = tgt_b.shape[0]
        
        # Sample random timesteps
        t = torch.randint(0, TIMESTEPS, (batch_size,), device=DEVICE).long()
        
        # Sample pure Gaussian noise
        noise = torch.randn_like(tgt_b)
        
        # Forward diffusion formula: x_t = sqrt(alpha_bar)*x_0 + sqrt(1 - alpha_bar)*noise
        sqrt_alpha_bar = sqrt_alphas_cumprod[t][:, None, None, None]
        sqrt_one_minus_alpha_bar = sqrt_one_minus_alphas_cumprod[t][:, None, None, None]
        x_t = sqrt_alpha_bar * tgt_b + sqrt_one_minus_alpha_bar * noise
        
        # Predict noise
        pred_noise = model(x_t, t, ref_b)
        
        # Foreground weighting mask: weight 3.0 if target alpha > 0.0 (in [-1, 1] scale)
        # Background pixels receive weight 1.0
        fg_mask = (tgt_b[:, 3:4] > -0.8).float()
        weights = 1.0 + 2.0 * fg_mask # 3.0 on FG, 1.0 on BG
        
        loss = ((pred_noise - noise) ** 2 * weights).mean()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    history.append({
        'epoch': epoch,
        'loss': loss.item()
    })
    
    if epoch % 25 == 0 or epoch == 1:
        print(f"Epoch [{epoch:03d}/{EPOCHS}] | Noise Prediction Loss: {loss.item():.4f}")

elapsed = time.time() - t0
print(f"Training completed in {elapsed:.2f} seconds.")

# Save checkpoint and loss history
torch.save(model.state_dict(), MODEL_SAVE_PATH)
pd.DataFrame(history).to_csv(LOSS_CSV_PATH, index=False)
print(f"Weights saved to: {MODEL_SAVE_PATH}")
print(f"Loss history saved to: {LOSS_CSV_PATH}")

# 5. Deterministic 50-Step DDIM Reverse Sampling
print("Executing 50-step DDIM reverse sampling on all 20 characters...")
model.eval()

# DDIM 50-step sub-sequence
DDIM_STEPS = 50
step_ratio = TIMESTEPS // DDIM_STEPS
ddim_timesteps = np.asarray(list(range(0, TIMESTEPS, step_ratio)))

with torch.no_grad():
    all_refs = torch.stack([p['ref_t'] for p in pairs_data]).to(DEVICE)
    N = all_refs.shape[0]
    
    # Initialize from pure Gaussian noise
    torch.manual_seed(42)
    cur_x = torch.randn_like(all_refs)
    
    for i, step in enumerate(reversed(range(len(ddim_timesteps)))):
        t_idx = ddim_timesteps[step]
        t = torch.full((N,), t_idx, device=DEVICE, dtype=torch.long)
        
        # Predict noise at timestep t
        pred_eps = model(cur_x, t, all_refs)
        
        # DDIM update math
        alpha_bar = alphas_cumprod[t_idx]
        alpha_bar_prev = alphas_cumprod[ddim_timesteps[step - 1]] if step > 0 else torch.tensor(1.0, device=DEVICE)
        
        # Estimate x_0
        pred_x0 = (cur_x - torch.sqrt(1.0 - alpha_bar) * pred_eps) / torch.sqrt(alpha_bar)
        pred_x0 = torch.clamp(pred_x0, -1.0, 1.0)
        
        # Direction pointing to x_t
        dir_xt = torch.sqrt(1.0 - alpha_bar_prev) * pred_eps
        
        # Sample x_{t-1}
        cur_x = torch.sqrt(alpha_bar_prev) * pred_x0 + dir_xt

# Denormalize generated outputs from [-1, 1] to [0, 255]
sampled_tensors = cur_x.cpu()
generated_sprites = []

for i in range(N):
    img_arr = sampled_tensors[i].permute(1, 2, 0).numpy()
    img_arr = ((img_arr + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
    generated_sprites.append(Image.fromarray(img_arr, mode='RGBA'))

# 6. Save Full 20-Character Evaluation Grid
fig, axes = plt.subplots(10, 6, figsize=(14, 22), dpi=150)
plt.subplots_adjust(wspace=0.1, hspace=0.3)

for i in range(20):
    rb = i // 2
    cb = (i % 2) * 3
    c = pairs_data[i]
    
    axes[rb, cb].imshow(c['ref_crop'])
    axes[rb, cb].set_title(f"{c['id']} Ref", fontsize=8)
    axes[rb, cb].axis('off')
    
    axes[rb, cb + 1].imshow(c['walk_crop'])
    axes[rb, cb + 1].set_title("Ground Truth Walk", fontsize=8)
    axes[rb, cb + 1].axis('off')
    
    axes[rb, cb + 2].imshow(generated_sprites[i])
    axes[rb, cb + 2].set_title("DDPM Sample (50-step)", fontsize=8)
    axes[rb, cb + 2].axis('off')

plt.suptitle("Rung 2: Reference-Conditioned DDPM 50-Step Reverse Sampling (20 Characters)", fontsize=13, y=0.99)
plt.savefig(PREVIEW_GRID_PATH, bbox_inches='tight')
plt.close()
print(f"Full 20-character DDPM grid saved to: {PREVIEW_GRID_PATH}")

# 7. Generate The Decisive 4-Way Comparison Card for Character C_00
# [1. Reference Idle | 2. Ground Truth Walk | 3. Direct U-Net | 4. DDPM Reverse Sample]
direct_weights = torch.load("research_paper_assets/06_sprite_deformation_and_pose_animation/rung1_direct_unet_best.pt", map_location=DEVICE)
from run_rung1_direct_regression import DirectRegressionUNet
direct_model = DirectRegressionUNet().to(DEVICE)
direct_model.load_state_dict(direct_weights)
direct_model.eval()

with torch.no_grad():
    c0_ref = pairs_data[0]['ref_t'].unsqueeze(0).to(DEVICE)
    # Rung 1 Direct model takes RGBA in [-1, 1] RGB and [0, 1] Alpha
    c0_ref_r1 = torch.cat([c0_ref[:, :3], (c0_ref[:, 3:4] + 1.0) / 2.0], dim=1)
    direct_pred = direct_model(c0_ref_r1).cpu().squeeze(0)
    
d_rgb = ((direct_pred[:3].permute(1, 2, 0).numpy() + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
d_alpha = (direct_pred[3].numpy() * 255.0).clip(0, 255).astype(np.uint8)
direct_pil = Image.fromarray(np.dstack([d_rgb, d_alpha]), mode='RGBA')

fig, axes = plt.subplots(1, 4, figsize=(14, 4), dpi=200)
axes[0].imshow(pairs_data[0]['ref_crop'])
axes[0].set_title("1. Neutral Stance Ref\n(LPC Character C_00)", fontsize=10)
axes[0].axis('off')

axes[1].imshow(pairs_data[0]['walk_crop'])
axes[1].set_title("2. Ground Truth Walk\n(Apex Stride Frame 3)", fontsize=10)
axes[1].axis('off')

axes[2].imshow(direct_pil)
axes[2].set_title("3. Direct U-Net (Rung 1)\n(Supervised Regression)", fontsize=10)
axes[2].axis('off')

axes[3].imshow(generated_sprites[0])
axes[3].set_title("4. DDPM Sample (Rung 2)\n(50-step Reverse DDIM)", fontsize=10)
axes[3].axis('off')

plt.suptitle("The Milestone Evaluation Card: Controlled Character Kinematics", fontsize=12, y=1.02)
plt.savefig(COMPARISON_4WAY_PATH, bbox_inches='tight')
plt.close()
print(f"Decisive 4-Way Comparison Card saved to: {COMPARISON_4WAY_PATH}")

