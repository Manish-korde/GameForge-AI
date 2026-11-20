import os
import sys
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

MODEL_SAVE_PATH = os.path.join(OUTPUT_DIR, "rung1_direct_unet_best.pt")
LOSS_CSV_PATH = os.path.join(OUTPUT_DIR, "rung1_direct_loss_history.csv")
PREVIEW_GRID_PATH = os.path.join(OUTPUT_DIR, "rung1_direct_regression_overfit_grid.png")

print(f"--- Rung 1: Deterministic Direct-Regression Baseline ---")
print(f"Device: {DEVICE}")

# 1. Dataset Streaming & Pre-caching 20 Characters
print("Loading 20 verified characters from dataset...")
ds = load_dataset("carlosuperb/lpc-action-pixel-art-diffusion", split="train", streaming=True)

pairs_data = []
count = 0

for item in ds:
    sheet = item['image'].convert('RGBA')
    if sheet.size != (576, 768):
        sheet = sheet.resize((576, 768), Image.Resampling.NEAREST)
        
    # Crop Row 2 Col 0 (Neutral Stance) and Row 2 Col 3 (Walk Apex)
    ref_crop = sheet.crop((0 * 64, 2 * 64, 1 * 64, 3 * 64))
    walk_crop = sheet.crop((3 * 64, 2 * 64, 4 * 64, 3 * 64))
    
    if np.array(ref_crop)[:, :, 3].sum() < 500:
        continue
        
    ref_arr = np.array(ref_crop, dtype=np.float32)
    walk_arr = np.array(walk_crop, dtype=np.float32)
    
    # Normalize: RGB to [-1, 1], Alpha to [0, 1]
    ref_rgb = ref_arr[:, :, :3] / 127.5 - 1.0
    ref_alpha = ref_arr[:, :, 3:4] / 255.0
    ref_tensor = np.concatenate([ref_rgb, ref_alpha], axis=2) # [64, 64, 4]
    
    walk_rgb = walk_arr[:, :, :3] / 127.5 - 1.0
    walk_alpha = walk_arr[:, :, 3:4] / 255.0
    walk_tensor = np.concatenate([walk_rgb, walk_alpha], axis=2)
    
    pairs_data.append({
        'id': f"C_{count:02d}",
        'ref_crop': ref_crop,
        'walk_crop': walk_crop,
        'ref_t': torch.tensor(ref_tensor).permute(2, 0, 1).float(), # [4, 64, 64]
        'tgt_t': torch.tensor(walk_tensor).permute(2, 0, 1).float()  # [4, 64, 64]
    })
    count += 1
    if count >= 20:
        break

print(f"Loaded {len(pairs_data)} character pairs into memory.")

class DirectPairDataset(Dataset):
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]['ref_t'], self.data[idx]['tgt_t']

train_loader = DataLoader(DirectPairDataset(pairs_data), batch_size=20, shuffle=False)

# 2. Direct-Regression U-Net Architecture (Decoupled Heads)
class DoubleConv(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.GroupNorm(8, out_c),
            nn.SiLU(),
            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.GroupNorm(8, out_c),
            nn.SiLU()
        )
    def forward(self, x):
        return self.conv(x)

class DirectRegressionUNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Encoder
        self.in_block = DoubleConv(4, 32)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(32, 64))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        
        # Bottleneck
        self.mid = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))
        
        # Decoder
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = DoubleConv(256, 128)
        
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = DoubleConv(128, 64)
        
        self.up0 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec0 = DoubleConv(64, 32)
        
        # Decoupled Output Heads
        self.head_rgb = nn.Conv2d(32, 3, 1)    # Continuous [-1, 1] output
        self.head_alpha = nn.Conv2d(32, 1, 1)  # Sigmoid [0, 1] output
        
    def forward(self, x):
        h0 = self.in_block(x)      # 32, 64, 64
        h1 = self.down1(h0)        # 64, 32, 32
        h2 = self.down2(h1)        # 128, 16, 16
        m = self.mid(h2)           # 256, 8, 8
        
        u2 = self.up2(m)           # 128, 16, 16
        d2 = self.dec2(torch.cat([u2, h2], dim=1))
        
        u1 = self.up1(d2)          # 64, 32, 32
        d1 = self.dec1(torch.cat([u1, h1], dim=1))
        
        u0 = self.up0(d1)          # 32, 64, 64
        d0 = self.dec0(torch.cat([u0, h0], dim=1))
        
        out_rgb = torch.tanh(self.head_rgb(d0))
        out_alpha = torch.sigmoid(self.head_alpha(d0))
        return torch.cat([out_rgb, out_alpha], dim=1)

model = DirectRegressionUNet().to(DEVICE)
param_count = sum(p.numel() for p in model.parameters())
print(f"Direct U-Net Initialized | Parameter Count: {param_count:,}")

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
bce_loss_fn = nn.BCELoss()

# 3. Supervised Overfit Training Loop
EPOCHS = 200
print(f"Beginning overfit optimization ({EPOCHS} epochs on 20 characters)...")

history = []
t0 = time.time()

for epoch in range(1, EPOCHS + 1):
    model.train()
    for ref_b, tgt_b in train_loader:
        ref_b = ref_b.to(DEVICE)
        tgt_b = tgt_b.to(DEVICE)
        
        pred = model(ref_b)
        
        pred_rgb = pred[:, :3]
        pred_alpha = pred[:, 3:4]
        tgt_rgb = tgt_b[:, :3]
        tgt_alpha = tgt_b[:, 3:4]
        
        # Loss 1: Alpha-Weighted RGB L1 Loss
        # Only penalize RGB in foreground regions
        fg_mask = (tgt_alpha > 0.1).float()
        l1_rgb = (torch.abs(pred_rgb - tgt_rgb) * fg_mask).sum() / (fg_mask.sum() * 3 + 1e-6)
        
        # Loss 2: Alpha Binary Cross Entropy
        loss_alpha = bce_loss_fn(pred_alpha, tgt_alpha)
        
        # Total Objective
        loss = l1_rgb + 0.5 * loss_alpha
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    history.append({
        'epoch': epoch,
        'total_loss': loss.item(),
        'l1_rgb': l1_rgb.item(),
        'bce_alpha': loss_alpha.item()
    })
    
    if epoch % 25 == 0 or epoch == 1:
        print(f"Epoch [{epoch:03d}/{EPOCHS}] | Total Loss: {loss.item():.4f} | L1 RGB: {l1_rgb.item():.4f} | Alpha BCE: {loss_alpha.item():.4f}")

elapsed = time.time() - t0
print(f"Training completed in {elapsed:.2f} seconds.")

# Save Model & Metrics
torch.save(model.state_dict(), MODEL_SAVE_PATH)
df_loss = pd.DataFrame(history)
df_loss.to_csv(LOSS_CSV_PATH, index=False)
print(f"Weights saved to: {MODEL_SAVE_PATH}")
print(f"Loss history saved to: {LOSS_CSV_PATH}")

# 4. Evaluation & Visual Overfit Grid
model.eval()
with torch.no_grad():
    all_refs = torch.stack([p['ref_t'] for p in pairs_data]).to(DEVICE)
    all_preds = model(all_refs).cpu()

# Plot 10 rows x 6 cols (2 characters per row block: [Ref | Target GT | Predicted Direct U-Net])
fig, axes = plt.subplots(10, 6, figsize=(14, 22), dpi=150)
plt.subplots_adjust(wspace=0.1, hspace=0.3)

for i in range(20):
    rb = i // 2
    cb = (i % 2) * 3
    
    c = pairs_data[i]
    ref_pil = c['ref_crop']
    tgt_pil = c['walk_crop']
    
    # Reconstruct predicted RGBA
    p_t = all_preds[i]
    p_rgb = ((p_t[:3].permute(1, 2, 0).numpy() + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
    p_alpha = (p_t[3].numpy() * 255.0).clip(0, 255).astype(np.uint8)
    pred_rgba = np.dstack([p_rgb, p_alpha])
    pred_pil = Image.fromarray(pred_rgba, mode='RGBA')
    
    # Calculate Reconstruction IoU on this overfit pair
    gt_bin = (np.array(tgt_pil)[:, :, 3] > 40)
    pred_bin = (p_alpha > 40)
    iou = float(np.sum(gt_bin & pred_bin)) / float(np.sum(gt_bin | pred_bin))
    
    axes[rb, cb].imshow(ref_pil)
    axes[rb, cb].set_title(f"{c['id']} Neutral Ref", fontsize=8)
    axes[rb, cb].axis('off')
    
    axes[rb, cb + 1].imshow(tgt_pil)
    axes[rb, cb + 1].set_title("Ground Truth Walk", fontsize=8)
    axes[rb, cb + 1].axis('off')
    
    axes[rb, cb + 2].imshow(pred_pil)
    axes[rb, cb + 2].set_title(f"Direct U-Net (IoU:{iou:.2f})", fontsize=8)
    axes[rb, cb + 2].axis('off')

plt.suptitle("Rung 1: Deterministic Direct-Regression Overfit Test (20 Characters)", fontsize=13, y=0.99)
plt.savefig(PREVIEW_GRID_PATH, bbox_inches='tight')
plt.close()
print(f"Visual overfit showcase saved to: {PREVIEW_GRID_PATH}")

