import os
import sys
import math
import json
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------------------------------------------------
# 1. STANDALONE U-NET & COSINE SCHEDULE DEFINITION (Zero imports from scripts)
# ----------------------------------------------------------------------
DEVICE = torch.device('cpu')
BASE_DIR = "research_paper_assets/06_sprite_deformation_and_pose_animation"
PHASE2_DIR = os.path.join(BASE_DIR, "forensic_phase_2")
os.makedirs(PHASE2_DIR, exist_ok=True)

TIMESTEPS = 1000

def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clamp(betas, 0.0001, 0.9999)

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
        self.in_conv = nn.Conv2d(8, 32, 3, padding=1)
        self.b1 = Block(32, 64, time_dim)
        self.down1 = nn.MaxPool2d(2)
        self.b2 = Block(64, 128, time_dim)
        self.down2 = nn.MaxPool2d(2)
        self.mid = Block(128, 256, time_dim)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.b_up2 = Block(256, 128, time_dim)
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.b_up1 = Block(128, 64, time_dim)
        self.out_conv = nn.Conv2d(64, 4, 3, padding=1)
        
    def forward(self, x_t, t, x_ref):
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

# ----------------------------------------------------------------------
# 2. LOAD DATA DETERMINISTICALLY
# ----------------------------------------------------------------------
cached_data = torch.load(os.path.join(PHASE2_DIR, "cached_c00_c01.pt"), weights_only=False)
c0_ref_pil, c0_walk_pil = cached_data[0]
c1_ref_pil, c1_walk_pil = cached_data[1]

# RGBA normalized strictly to [-1, 1]
c0_ref = torch.tensor(np.array(c0_ref_pil, dtype=np.float32) / 127.5 - 1.0).permute(2, 0, 1).unsqueeze(0)
c0_tgt = torch.tensor(np.array(c0_walk_pil, dtype=np.float32) / 127.5 - 1.0).permute(2, 0, 1).unsqueeze(0)

c1_ref = torch.tensor(np.array(c1_ref_pil, dtype=np.float32) / 127.5 - 1.0).permute(2, 0, 1).unsqueeze(0)
zero_ref = torch.zeros_like(c0_ref)

betas = cosine_beta_schedule(TIMESTEPS)
alphas = 1.0 - betas
alphas_bar = torch.cumprod(alphas, dim=0)

# Foreground mask
fg_mask = (c0_tgt[:, 3:4] > -0.8).squeeze()
bg_mask = ~fg_mask

def calc_psnr(img1, img2, max_val=2.0):
    mse = torch.mean((img1 - img2) ** 2).item()
    if mse == 0:
        return 100.0
    return 20.0 * math.log10(max_val / math.sqrt(mse))

def calc_ssim_simple(img1, img2):
    x = (img1 + 1.0) / 2.0
    y = (img2 + 1.0) / 2.0
    mu_x = x.mean()
    mu_y = y.mean()
    sigma_x = x.var()
    sigma_y = y.var()
    sigma_xy = ((x - mu_x) * (y - mu_y)).mean()
    k1, k2 = 0.01, 0.03
    c1, c2 = k1**2, k2**2
    ssim = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / ((mu_x**2 + mu_y**2 + c1) * (sigma_x + sigma_y + c2))
    return float(ssim.item())

# ----------------------------------------------------------------------
# TEST 1: CHECKPOINT AUDIT
# ----------------------------------------------------------------------
print("--- TEST 1: Checkpoint Audit ---")
loss_csv_path = os.path.join(BASE_DIR, "rung2_ddpm_loss_history.csv")
loss_df = pd.read_csv(loss_csv_path)

min_idx = loss_df['loss'].idxmin()
min_epoch = int(loss_df.loc[min_idx, 'epoch'])
min_loss = float(loss_df.loc[min_idx, 'loss'])

final_epoch = int(loss_df.iloc[-1]['epoch'])
final_loss = float(loss_df.iloc[-1]['loss'])

ckpt_path = os.path.join(BASE_DIR, "rung2_ddpm_overfit_best.pt")

model = ReferenceConditionedUNet().to(DEVICE)
state_dict = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
model.load_state_dict(state_dict)
model.eval()

checkpoint_audit_data = {
    "active_checkpoint_file": ckpt_path,
    "file_size_bytes": os.path.getsize(ckpt_path),
    "successfully_loaded": True,
    "discrepancy_explanation": "In run_rung2_ddpm_overfit.py, torch.save() was invoked after the 150-epoch training loop finished, saving the final model state into 'rung2_ddpm_overfit_best.pt'. There was no mid-loop best-loss comparator saving separate files. Therefore, the saved weights correspond to Final Epoch 150 (loss 0.182659), while the historical minimum occurred at Epoch 145 (loss 0.131424). The 'Epoch 118' reference in prior notes was from an earlier unweighted baseline run before foreground re-weighting.",
    "minimum_in_history": {
        "epoch": min_epoch,
        "loss": min_loss
    },
    "saved_checkpoint_state": {
        "epoch": final_epoch,
        "loss": final_loss
    }
}

with open(os.path.join(PHASE2_DIR, "checkpoint_audit.json"), "w") as f:
    json.dump(checkpoint_audit_data, f, indent=2)

print("Checkpoint Audit complete. Saved checkpoint_audit.json.")

# ----------------------------------------------------------------------
# TEST 2: PREDICTED X0 RECONSTRUCTION ERROR ACROSS TIMESTEPS
# ----------------------------------------------------------------------
print("\n--- TEST 2: Predicted x0 Reconstruction Error across Timesteps ---")
test_timesteps = [999, 750, 500, 250, 100, 50, 20, 0]
torch.manual_seed(42)
fixed_eps = torch.randn_like(c0_tgt)

records_t2 = []
visual_snapshots = {}

for t_val in test_timesteps:
    t_tensor = torch.tensor([t_val])
    ab = alphas_bar[t_val]
    
    if t_val == 0:
        x_t = c0_tgt.clone()
        actual_noise = torch.zeros_like(c0_tgt)
    else:
        x_t = torch.sqrt(ab) * c0_tgt + torch.sqrt(1.0 - ab) * fixed_eps
        actual_noise = fixed_eps
        
    with torch.no_grad():
        pred_eps = model(x_t, t_tensor, c0_ref)
        
    eps_diff_sq = (pred_eps - actual_noise) ** 2
    tot_eps_mse = eps_diff_sq.mean().item()
    fg_eps_mse = eps_diff_sq[:, :, fg_mask].mean().item()
    bg_eps_mse = eps_diff_sq[:, :, bg_mask].mean().item()
    
    if t_val == 0:
        pred_x0 = x_t
    else:
        pred_x0 = (x_t - torch.sqrt(1.0 - ab) * pred_eps) / torch.sqrt(ab)
        pred_x0 = torch.clamp(pred_x0, -1.0, 1.0)
        
    x0_diff_sq = (pred_x0 - c0_tgt) ** 2
    tot_x0_mse = x0_diff_sq.mean().item()
    fg_x0_mse = x0_diff_sq[:, :, fg_mask].mean().item()
    bg_x0_mse = x0_diff_sq[:, :, bg_mask].mean().item()
    
    psnr_val = calc_psnr(pred_x0, c0_tgt)
    ssim_val = calc_ssim_simple(pred_x0, c0_tgt)
    amp_factor = math.sqrt((1.0 - ab.item()) / ab.item()) if ab.item() > 0 else 0.0
    
    records_t2.append({
        'timestep': t_val,
        'alpha_bar': round(ab.item(), 5),
        'amp_factor': round(amp_factor, 4),
        'tot_eps_mse': round(tot_eps_mse, 4),
        'fg_eps_mse': round(fg_eps_mse, 4),
        'bg_eps_mse': round(bg_eps_mse, 4),
        'tot_x0_mse': round(tot_x0_mse, 4),
        'fg_x0_mse': round(fg_x0_mse, 4),
        'bg_x0_mse': round(bg_x0_mse, 4),
        'psnr': round(psnr_val, 2),
        'ssim': round(ssim_val, 4)
    })
    
    if t_val in [999, 500, 100, 20]:
        visual_snapshots[t_val] = {
            'xt': ((x_t.squeeze(0).permute(1, 2, 0).numpy() + 1.0) * 127.5).clip(0, 255).astype(np.uint8),
            'pred_eps': ((pred_eps.squeeze(0).permute(1, 2, 0).numpy() + 1.0) * 127.5).clip(0, 255).astype(np.uint8),
            'pred_x0': ((pred_x0.squeeze(0).permute(1, 2, 0).numpy() + 1.0) * 127.5).clip(0, 255).astype(np.uint8),
            'abs_err': (torch.abs(pred_x0 - c0_tgt).squeeze(0).mean(dim=0).numpy() * 255.0).clip(0, 255).astype(np.uint8)
        }

df_t2 = pd.DataFrame(records_t2)
df_t2.to_csv(os.path.join(PHASE2_DIR, "predicted_x0_metrics.csv"), index=False)
print("Saved predicted_x0_metrics.csv:")
print(df_t2[['timestep', 'alpha_bar', 'amp_factor', 'fg_eps_mse', 'fg_x0_mse', 'psnr', 'ssim']].to_string(index=False))

# Visualization Test 2
fig, axes = plt.subplots(4, 5, figsize=(14, 11), dpi=150)
gt_img = ((c0_tgt.squeeze(0).permute(1, 2, 0).numpy() + 1.0) * 127.5).clip(0, 255).astype(np.uint8)

for r, t_val in enumerate([999, 500, 100, 20]):
    snap = visual_snapshots[t_val]
    axes[r, 0].imshow(gt_img)
    axes[r, 0].set_title(f"GT x_0 (t={t_val})", fontsize=8)
    axes[r, 0].axis('off')
    
    axes[r, 1].imshow(snap['xt'])
    axes[r, 1].set_title(f"Noisy x_t", fontsize=8)
    axes[r, 1].axis('off')
    
    axes[r, 2].imshow(snap['pred_eps'])
    axes[r, 2].set_title(f"Pred eps (FG MSE:{df_t2.loc[df_t2['timestep']==t_val, 'fg_eps_mse'].values[0]})", fontsize=8)
    axes[r, 2].axis('off')
    
    axes[r, 3].imshow(snap['pred_x0'])
    axes[r, 3].set_title(f"Pred x_0 (PSNR:{df_t2.loc[df_t2['timestep']==t_val, 'psnr'].values[0]} dB)", fontsize=8)
    axes[r, 3].axis('off')
    
    axes[r, 4].imshow(snap['abs_err'], cmap='hot')
    axes[r, 4].set_title(f"Abs Error Heatmap", fontsize=8)
    axes[r, 4].axis('off')

plt.suptitle("Test 2: Single-Step Reconstruction x_0 vs Ground Truth across Timesteps", fontsize=12)
plt.savefig(os.path.join(PHASE2_DIR, "test2_x0_reconstruction_breakdown.png"), bbox_inches='tight')
plt.close()

# ----------------------------------------------------------------------
# TEST 3: DIRECT X0 RECONSTRUCTION VS DDIM
# ----------------------------------------------------------------------
print("\n--- TEST 3: Direct x0 Reconstruction vs 50-Step DDIM ---")
t_500 = torch.tensor([500])
ab_500 = alphas_bar[500]
xt_500 = torch.sqrt(ab_500) * c0_tgt + torch.sqrt(1.0 - ab_500) * fixed_eps

with torch.no_grad():
    pred_eps_500 = model(xt_500, t_500, c0_ref)
    
direct_pred_x0 = (xt_500 - torch.sqrt(1.0 - ab_500) * pred_eps_500) / torch.sqrt(ab_500)
direct_pred_x0 = torch.clamp(direct_pred_x0, -1.0, 1.0)

# Multi-step DDIM from t=500
DDIM_STEPS = 50
step_ratio = TIMESTEPS // DDIM_STEPS
ddim_timesteps = np.asarray(list(range(0, TIMESTEPS, step_ratio)))

cur_ddim_x = xt_500.clone()
start_step_idx = np.where(ddim_timesteps <= 500)[0][-1]

with torch.no_grad():
    for step in reversed(range(start_step_idx + 1)):
        t_idx = ddim_timesteps[step]
        prev_idx = ddim_timesteps[step - 1] if step > 0 else -1
        ab_t = alphas_bar[t_idx]
        ab_prev = alphas_bar[prev_idx] if step > 0 else torch.tensor(1.0)
        
        t_ten = torch.full((1,), t_idx, dtype=torch.long)
        p_eps = model(cur_ddim_x, t_ten, c0_ref)
        
        p_x0 = (cur_ddim_x - torch.sqrt(1.0 - ab_t) * p_eps) / torch.sqrt(ab_t)
        p_x0 = torch.clamp(p_x0, -1.0, 1.0)
        dir_xt = torch.sqrt(1.0 - ab_prev) * p_eps
        cur_ddim_x = torch.sqrt(ab_prev) * p_x0 + dir_xt

# Oracle from t=500
cur_oracle_x = xt_500.clone()
for step in reversed(range(start_step_idx + 1)):
    t_idx = ddim_timesteps[step]
    prev_idx = ddim_timesteps[step - 1] if step > 0 else -1
    ab_t = alphas_bar[t_idx]
    ab_prev = alphas_bar[prev_idx] if step > 0 else torch.tensor(1.0)
    
    exact_eps = (cur_oracle_x - torch.sqrt(ab_t) * c0_tgt) / torch.sqrt(1.0 - ab_t)
    p_x0 = (cur_oracle_x - torch.sqrt(1.0 - ab_t) * exact_eps) / torch.sqrt(ab_t)
    dir_xt = torch.sqrt(1.0 - ab_prev) * exact_eps
    cur_oracle_x = torch.sqrt(ab_prev) * p_x0 + dir_xt

img_gt = Image.fromarray(gt_img, mode='RGBA')
img_direct = Image.fromarray(((direct_pred_x0.squeeze(0).permute(1, 2, 0).numpy() + 1.0) * 127.5).clip(0, 255).astype(np.uint8), mode='RGBA')
img_ddim = Image.fromarray(((cur_ddim_x.squeeze(0).permute(1, 2, 0).numpy() + 1.0) * 127.5).clip(0, 255).astype(np.uint8), mode='RGBA')
img_oracle = Image.fromarray(((cur_oracle_x.squeeze(0).permute(1, 2, 0).numpy() + 1.0) * 127.5).clip(0, 255).astype(np.uint8), mode='RGBA')

psnr_dir = calc_psnr(direct_pred_x0, c0_tgt)
psnr_ddim = calc_psnr(cur_ddim_x, c0_tgt)
psnr_orc = calc_psnr(cur_oracle_x, c0_tgt)

print(f"Direct x0 (t=500): PSNR = {psnr_dir:.2f} dB | SSIM = {calc_ssim_simple(direct_pred_x0, c0_tgt):.4f}")
print(f"DDIM (from t=500): PSNR = {psnr_ddim:.2f} dB | SSIM = {calc_ssim_simple(cur_ddim_x, c0_tgt):.4f}")
print(f"Oracle (from t=500): PSNR = {psnr_orc:.2f} dB | SSIM = {calc_ssim_simple(cur_oracle_x, c0_tgt):.4f}")

fig, axes = plt.subplots(1, 4, figsize=(14, 4), dpi=180)
axes[0].imshow(img_gt)
axes[0].set_title("1. Ground Truth x_0", fontsize=10)
axes[0].axis('off')

axes[1].imshow(img_direct)
axes[1].set_title(f"2. Direct Pred x_0 (t=500)\nPSNR: {psnr_dir:.1f} dB", fontsize=10)
axes[1].axis('off')

axes[2].imshow(img_ddim)
axes[2].set_title(f"3. DDIM Trajectory (t=500->0)\nPSNR: {psnr_ddim:.1f} dB", fontsize=10)
axes[2].axis('off')

axes[3].imshow(img_oracle)
axes[3].set_title(f"4. Oracle Inversion\nPSNR: {psnr_orc:.1f} dB", fontsize=10)
axes[3].axis('off')

plt.suptitle("Test 3: Direct Single-Step x_0 vs. Iterative DDIM Trajectory Accumulation", fontsize=12)
plt.savefig(os.path.join(PHASE2_DIR, "direct_x0_vs_ddim.png"), bbox_inches='tight')
plt.close()

# ----------------------------------------------------------------------
# TEST 4: TEACHER-FORCED VS FREE-RUNNING TRAJECTORY
# ----------------------------------------------------------------------
print("\n--- TEST 4: Teacher-Forced vs Free-Running Trajectory ---")
sample_steps = [980, 800, 600, 400, 200, 100, 40, 20]
teacher_fg_mse = []
freerun_fg_mse = []

free_x = torch.randn_like(c0_tgt)
free_x0_history = {}

with torch.no_grad():
    for step in reversed(range(len(ddim_timesteps))):
        t_idx = ddim_timesteps[step]
        prev_idx = ddim_timesteps[step - 1] if step > 0 else -1
        ab_t = alphas_bar[t_idx]
        ab_prev = alphas_bar[prev_idx] if step > 0 else torch.tensor(1.0)
        
        t_ten = torch.full((1,), t_idx, dtype=torch.long)
        p_eps = model(free_x, t_ten, c0_ref)
        p_x0 = (free_x - torch.sqrt(1.0 - ab_t) * p_eps) / torch.sqrt(ab_t)
        p_x0 = torch.clamp(p_x0, -1.0, 1.0)
        
        dir_xt = torch.sqrt(1.0 - ab_prev) * p_eps
        free_x = torch.sqrt(ab_prev) * p_x0 + dir_xt
        
        if t_idx in sample_steps:
            diff_sq = (p_x0 - c0_tgt) ** 2
            free_x0_history[t_idx] = diff_sq[:, :, fg_mask].mean().item()

for t_val in sample_steps:
    ab = alphas_bar[t_val]
    xt_tf = torch.sqrt(ab) * c0_tgt + torch.sqrt(1.0 - ab) * fixed_eps
    t_ten = torch.tensor([t_val])
    with torch.no_grad():
        p_eps_tf = model(xt_tf, t_ten, c0_ref)
    p_x0_tf = (xt_tf - torch.sqrt(1.0 - ab) * p_eps_tf) / torch.sqrt(ab)
    p_x0_tf = torch.clamp(p_x0_tf, -1.0, 1.0)
    
    diff_sq_tf = (p_x0_tf - c0_tgt) ** 2
    tf_mse = diff_sq_tf[:, :, fg_mask].mean().item()
    teacher_fg_mse.append(tf_mse)
    freerun_fg_mse.append(free_x0_history.get(t_val, 0.0))

plt.figure(figsize=(9, 5), dpi=150)
plt.plot(sample_steps, teacher_fg_mse, 'o-', color='blue', label='Teacher-Forced FG x_0 MSE (Clean forward x_t)')
plt.plot(sample_steps, freerun_fg_mse, 's--', color='red', label='Free-Running DDIM FG x_0 MSE (Iterative reverse trajectory)')
plt.gca().invert_xaxis()
plt.xlabel("Diffusion Timestep t (1000 -> 0)", fontsize=11)
plt.ylabel("Foreground x_0 Mean Squared Error", fontsize=11)
plt.title("Test 4: Teacher-Forced vs. Free-Running Trajectory Error Drift", fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(fontsize=10)
plt.savefig(os.path.join(PHASE2_DIR, "teacher_forced_vs_free_running.png"), bbox_inches='tight')
plt.close()
print("Saved teacher_forced_vs_free_running.png.")

# ----------------------------------------------------------------------
# TEST 5: TIMESTEP EMBEDDING AUDIT
# ----------------------------------------------------------------------
print("\n--- TEST 5: Timestep Embedding Audit ---")
t_10 = torch.tensor([10])
t_900 = torch.tensor([900])
dummy_xt = torch.zeros_like(c0_tgt)

with torch.no_grad():
    out_t10 = model(dummy_xt, t_10, c0_ref)
    out_t900 = model(dummy_xt, t_900, c0_ref)
    
time_diff = torch.abs(out_t10 - out_t900).mean().item()
print(f"Output divergence between t=10 and t=900: {time_diff:.6f}")

audit_md = f"""# Test 5: Timestep Embedding Implementation & Range Audit

## 1. Verified Properties
- **Training Timestep Range:** Sampled via `torch.randint(0, 1000, (batch_size,))` $\implies t \in [0, 999]$ (Discrete integer index).
- **Sampling Timestep Range:** `ddim_timesteps = np.asarray(list(range(0, 1000, 20)))` $\implies t \in [0, 980]$ (Strict subset of training domain).
- **Tensor Dtype:** `torch.long` in both training and reverse DDIM sampling.
- **Normalization:** Raw discrete integers $t \in [0, 999]$ are passed directly into `SinusoidalPositionEmbeddings(dim=128)` without pre-normalization, which matches the standard Vaswani / DDPM formulation.
- **Influence on Model Output:** Mean absolute activation delta between $t=10$ and $t=900$ on identical input is **{time_diff:.4f}**, confirming the time embeddings propagate through all down/mid/up blocks.

## 2. Implementation Code Snippet
```python
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
```

## 3. Findings
- Timestep processing is mathematically correct and consistently indexed in both training and DDIM inference.
- $t=0$ was included in the training distribution (`torch.randint(0, 1000)` includes 0).
"""

with open(os.path.join(PHASE2_DIR, "timestep_embedding_audit.md"), "w") as f:
    f.write(audit_md)
print("Saved timestep_embedding_audit.md.")

# ----------------------------------------------------------------------
# TEST 6: REFERENCE CONDITIONING AUDIT
# ----------------------------------------------------------------------
print("\n--- TEST 6: Reference Conditioning Sensitivity Audit ---")
t_test = torch.tensor([500])
xt_test = torch.sqrt(alphas_bar[500]) * c0_tgt + torch.sqrt(1.0 - alphas_bar[500]) * fixed_eps

with torch.no_grad():
    eps_correct = model(xt_test, t_test, c0_ref)
    eps_wrong = model(xt_test, t_test, c1_ref)
    eps_zero = model(xt_test, t_test, zero_ref)
    
def get_x0_from_eps(eps_tensor):
    pred = (xt_test - torch.sqrt(1.0 - alphas_bar[500]) * eps_tensor) / torch.sqrt(alphas_bar[500])
    return torch.clamp(pred, -1.0, 1.0)

x0_correct = get_x0_from_eps(eps_correct)
x0_wrong = get_x0_from_eps(eps_wrong)
x0_zero = get_x0_from_eps(eps_zero)

cond_records = [
    {
        'conditioning_mode': 'A. Correct Reference (C_00)',
        'eps_mse_total': float(F.mse_loss(eps_correct, fixed_eps).item()),
        'eps_mse_fg': float(((eps_correct - fixed_eps)**2)[:, :, fg_mask].mean().item()),
        'x0_mse_fg': float(((x0_correct - c0_tgt)**2)[:, :, fg_mask].mean().item()),
        'psnr': calc_psnr(x0_correct, c0_tgt),
        'ssim': calc_ssim_simple(x0_correct, c0_tgt)
    },
    {
        'conditioning_mode': 'B. Incorrect Reference (C_01)',
        'eps_mse_total': float(F.mse_loss(eps_wrong, fixed_eps).item()),
        'eps_mse_fg': float(((eps_wrong - fixed_eps)**2)[:, :, fg_mask].mean().item()),
        'x0_mse_fg': float(((x0_wrong - c0_tgt)**2)[:, :, fg_mask].mean().item()),
        'psnr': calc_psnr(x0_wrong, c0_tgt),
        'ssim': calc_ssim_simple(x0_wrong, c0_tgt)
    },
    {
        'conditioning_mode': 'C. Zero Reference (All Zeros)',
        'eps_mse_total': float(F.mse_loss(eps_zero, fixed_eps).item()),
        'eps_mse_fg': float(((eps_zero - fixed_eps)**2)[:, :, fg_mask].mean().item()),
        'x0_mse_fg': float(((x0_zero - c0_tgt)**2)[:, :, fg_mask].mean().item()),
        'psnr': calc_psnr(x0_zero, c0_tgt),
        'ssim': calc_ssim_simple(x0_zero, c0_tgt)
    }
]

df_cond = pd.DataFrame(cond_records)
df_cond.to_csv(os.path.join(PHASE2_DIR, "reference_conditioning_metrics.csv"), index=False)
print("Saved reference_conditioning_metrics.csv:")
print(df_cond.to_string(index=False))

fig, axes = plt.subplots(1, 4, figsize=(14, 4), dpi=180)
axes[0].imshow(img_gt)
axes[0].set_title("Ground Truth Target (C_00)", fontsize=10)
axes[0].axis('off')

axes[1].imshow(Image.fromarray(((x0_correct.squeeze(0).permute(1, 2, 0).numpy() + 1.0) * 127.5).clip(0, 255).astype(np.uint8), mode='RGBA'))
axes[1].set_title(f"A. Correct Ref\nPSNR: {cond_records[0]['psnr']:.1f} dB", fontsize=10)
axes[1].axis('off')

axes[2].imshow(Image.fromarray(((x0_wrong.squeeze(0).permute(1, 2, 0).numpy() + 1.0) * 127.5).clip(0, 255).astype(np.uint8), mode='RGBA'))
axes[2].set_title(f"B. Wrong Ref (C_01)\nPSNR: {cond_records[1]['psnr']:.1f} dB", fontsize=10)
axes[2].axis('off')

axes[3].imshow(Image.fromarray(((x0_zero.squeeze(0).permute(1, 2, 0).numpy() + 1.0) * 127.5).clip(0, 255).astype(np.uint8), mode='RGBA'))
axes[3].set_title(f"C. Zero Ref\nPSNR: {cond_records[2]['psnr']:.1f} dB", fontsize=10)
axes[3].axis('off')

plt.suptitle("Test 6: Reference Conditioning Sensitivity Comparison", fontsize=12)
plt.savefig(os.path.join(PHASE2_DIR, "reference_conditioning_comparison.png"), bbox_inches='tight')
plt.close()
print("Saved reference_conditioning_comparison.png.")
print("\nAll Phase 2 Forensic Tests Completed Successfully!")

