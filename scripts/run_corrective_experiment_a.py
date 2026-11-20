import os
import sys
import math
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------------------------------------------------
# 1. SETUP & PATHS
# ----------------------------------------------------------------------
DEVICE = torch.device('cpu')
BASE_DIR = "research_paper_assets/06_sprite_deformation_and_pose_animation"
PHASE2_DIR = os.path.join(BASE_DIR, "forensic_phase_2")
EXP_A_DIR = os.path.join(BASE_DIR, "corrective_experiment_A")
os.makedirs(EXP_A_DIR, exist_ok=True)

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
# 2. LOAD DATA & MODEL
# ----------------------------------------------------------------------
cached_data = torch.load(os.path.join(PHASE2_DIR, "cached_c00_c01.pt"), weights_only=False)
c0_ref_pil, c0_walk_pil = cached_data[0]

c0_ref = torch.tensor(np.array(c0_ref_pil, dtype=np.float32) / 127.5 - 1.0).permute(2, 0, 1).unsqueeze(0)
c0_tgt = torch.tensor(np.array(c0_walk_pil, dtype=np.float32) / 127.5 - 1.0).permute(2, 0, 1).unsqueeze(0)

betas = cosine_beta_schedule(TIMESTEPS)
alphas = 1.0 - betas
alphas_bar = torch.cumprod(alphas, dim=0)

ckpt_path = os.path.join(BASE_DIR, "rung2_ddpm_overfit_best.pt")
model = ReferenceConditionedUNet().to(DEVICE)
state_dict = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
model.load_state_dict(state_dict)
model.eval()

# Masks
fg_mask_tgt = (c0_tgt[:, 3:4] > -0.8).squeeze()
fg_mask_ref = (c0_ref[:, 3:4] > -0.8).squeeze()

def calc_psnr(img1, img2, max_val=2.0):
    mse = torch.mean((img1 - img2) ** 2).item()
    if mse <= 0:
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

def calc_silhouette_iou(img1, img2, threshold=-0.5):
    # img in [-1, 1] range, channel 3 is alpha
    a1 = (img1[0, 3] > threshold).float()
    a2 = (img2[0, 3] > threshold).float()
    inter = (a1 * a2).sum().item()
    union = torch.clamp(a1 + a2, 0, 1).sum().item()
    return inter / union if union > 0 else 0.0

def calc_leg_displacement_stats(pred_img, ref_img, tgt_img):
    # Leg box: y in [42, 63], x in [16, 47]
    # Check changed pixels in leg region
    p_a = pred_img[0, 3, 42:64, 16:48] > -0.5
    r_a = ref_img[0, 3, 42:64, 16:48] > -0.5
    t_a = tgt_img[0, 3, 42:64, 16:48] > -0.5
    
    # Delta from ref: does pred leg move away from ref?
    diff_from_ref = (p_a != r_a).sum().item()
    # Match with tgt: does pred leg match tgt leg?
    diff_from_tgt = (p_a != t_a).sum().item()
    
    # Left leg centroid
    def centroid(mask):
        ys, xs = torch.where(mask)
        if len(ys) == 0:
            return 0.0, 0.0
        return xs.float().mean().item(), ys.float().mean().item()
        
    p_lx, p_ly = centroid(pred_img[0, 3, 42:64, 16:32] > -0.5)
    r_lx, r_ly = centroid(ref_img[0, 3, 42:64, 16:32] > -0.5)
    t_lx, t_ly = centroid(tgt_img[0, 3, 42:64, 16:32] > -0.5)
    
    disp_pred_vs_ref_x = p_lx - r_lx
    disp_tgt_vs_ref_x = t_lx - r_lx
    
    return {
        'leg_diff_from_ref': diff_from_ref,
        'leg_diff_from_tgt': diff_from_tgt,
        'pred_left_leg_dx': round(disp_pred_vs_ref_x, 2),
        'gt_left_leg_dx': round(disp_tgt_vs_ref_x, 2)
    }

# ----------------------------------------------------------------------
# SAMPLER CONFIGURATION
# ----------------------------------------------------------------------
DDIM_STEPS = 50
step_ratio = TIMESTEPS // DDIM_STEPS
ddim_timesteps = np.asarray(list(range(0, TIMESTEPS, step_ratio)))

# Find step index for t=500
step_500_idx = np.where(ddim_timesteps <= 500)[0][-1]
t_500_actual = ddim_timesteps[step_500_idx] # 500
ab_500 = alphas_bar[t_500_actual]
sqrt_ab_500 = math.sqrt(ab_500.item())
sqrt_one_minus_ab_500 = math.sqrt(1.0 - ab_500.item())

# ======================================================================
# EXPERIMENT 1: BASELINE (Gaussian Init from t=980)
# ======================================================================
print("--- Running Experiment 1: Baseline (Gaussian Init from t=980) ---")
torch.manual_seed(42)
x_base = torch.randn_like(c0_tgt)
cur_base = x_base.clone()

with torch.no_grad():
    for step in reversed(range(len(ddim_timesteps))):
        t_idx = ddim_timesteps[step]
        prev_idx = ddim_timesteps[step - 1] if step > 0 else -1
        ab_t = alphas_bar[t_idx]
        ab_prev = alphas_bar[prev_idx] if step > 0 else torch.tensor(1.0)
        t_ten = torch.full((1,), t_idx, dtype=torch.long)
        
        pe = model(cur_base, t_ten, c0_ref)
        p_x0 = (cur_base - torch.sqrt(1.0 - ab_t) * pe) / torch.sqrt(ab_t)
        p_x0 = torch.clamp(p_x0, -1.0, 1.0)
        dir_xt = torch.sqrt(1.0 - ab_prev) * pe
        cur_base = torch.sqrt(ab_prev) * p_x0 + dir_xt

# ======================================================================
# EXPERIMENT 2: TARGET-MATCHED RECONSTRUCTION CONTROL (from t=500)
# ======================================================================
print("--- Running Experiment 2: Target-Matched Control (Target + Noise at t=500) ---")
torch.manual_seed(42)
eps_fixed = torch.randn_like(c0_tgt)
x_target_500 = sqrt_ab_500 * c0_tgt + sqrt_one_minus_ab_500 * eps_fixed
cur_ctrl = x_target_500.clone()

with torch.no_grad():
    for step in reversed(range(step_500_idx + 1)):
        t_idx = ddim_timesteps[step]
        prev_idx = ddim_timesteps[step - 1] if step > 0 else -1
        ab_t = alphas_bar[t_idx]
        ab_prev = alphas_bar[prev_idx] if step > 0 else torch.tensor(1.0)
        t_ten = torch.full((1,), t_idx, dtype=torch.long)
        
        pe = model(cur_ctrl, t_ten, c0_ref)
        p_x0 = (cur_ctrl - torch.sqrt(1.0 - ab_t) * pe) / torch.sqrt(ab_t)
        p_x0 = torch.clamp(p_x0, -1.0, 1.0)
        dir_xt = torch.sqrt(1.0 - ab_prev) * pe
        cur_ctrl = torch.sqrt(ab_prev) * p_x0 + dir_xt

# ======================================================================
# EXPERIMENT 3: REFERENCE-GROUNDED WARM-START (Reference + Noise at t=500)
# ======================================================================
print("--- Running Experiment 3: Reference-Grounded Warm-Start (16 Seeds) ---")
NUM_SEEDS = 16
ref_warm_outputs = []
ref_warm_metrics = []

for seed in range(NUM_SEEDS):
    torch.manual_seed(seed + 100)
    eps_seed = torch.randn_like(c0_ref)
    
    # Warm start initialization strictly from REFERENCE
    x_ref_500 = sqrt_ab_500 * c0_ref + sqrt_one_minus_ab_500 * eps_seed
    cur_ref_warm = x_ref_500.clone()
    
    with torch.no_grad():
        for step in reversed(range(step_500_idx + 1)):
            t_idx = ddim_timesteps[step]
            prev_idx = ddim_timesteps[step - 1] if step > 0 else -1
            ab_t = alphas_bar[t_idx]
            ab_prev = alphas_bar[prev_idx] if step > 0 else torch.tensor(1.0)
            t_ten = torch.full((1,), t_idx, dtype=torch.long)
            
            pe = model(cur_ref_warm, t_ten, c0_ref)
            p_x0 = (cur_ref_warm - torch.sqrt(1.0 - ab_t) * pe) / torch.sqrt(ab_t)
            p_x0 = torch.clamp(p_x0, -1.0, 1.0)
            dir_xt = torch.sqrt(1.0 - ab_prev) * pe
            cur_ref_warm = torch.sqrt(ab_prev) * p_x0 + dir_xt
            
    # Calculate metrics
    psnr_tgt = calc_psnr(cur_ref_warm, c0_tgt)
    ssim_tgt = calc_ssim_simple(cur_ref_warm, c0_tgt)
    psnr_ref = calc_psnr(cur_ref_warm, c0_ref)
    ssim_ref = calc_ssim_simple(cur_ref_warm, c0_ref)
    fg_mse_tgt = ((cur_ref_warm - c0_tgt)**2)[:, :, fg_mask_tgt].mean().item()
    fg_mse_ref = ((cur_ref_warm - c0_ref)**2)[:, :, fg_mask_ref].mean().item()
    sil_iou_tgt = calc_silhouette_iou(cur_ref_warm, c0_tgt)
    sil_iou_ref = calc_silhouette_iou(cur_ref_warm, c0_ref)
    
    leg_stats = calc_leg_displacement_stats(cur_ref_warm, c0_ref, c0_tgt)
    
    ref_warm_metrics.append({
        'seed': seed,
        'psnr_vs_target': round(psnr_tgt, 2),
        'ssim_vs_target': round(ssim_tgt, 4),
        'psnr_vs_ref': round(psnr_ref, 2),
        'ssim_vs_ref': round(ssim_ref, 4),
        'fg_mse_vs_tgt': round(fg_mse_tgt, 4),
        'fg_mse_vs_ref': round(fg_mse_ref, 4),
        'sil_iou_vs_tgt': round(sil_iou_tgt, 4),
        'sil_iou_vs_ref': round(sil_iou_ref, 4),
        'leg_diff_from_ref': leg_stats['leg_diff_from_ref'],
        'leg_diff_from_tgt': leg_stats['leg_diff_from_tgt'],
        'pred_left_leg_dx': leg_stats['pred_left_leg_dx'],
        'gt_left_leg_dx': leg_stats['gt_left_leg_dx']
    })
    
    ref_warm_outputs.append(cur_ref_warm)

# ======================================================================
# COMPILE METRICS & COMPARISON TABLE
# ======================================================================
df_exp3 = pd.DataFrame(ref_warm_metrics)
df_exp3.to_csv(os.path.join(EXP_A_DIR, "reference_grounded_16seeds_metrics.csv"), index=False)

# Metrics for Baseline and Control
psnr_base_tgt = calc_psnr(cur_base, c0_tgt)
ssim_base_tgt = calc_ssim_simple(cur_base, c0_tgt)
psnr_base_ref = calc_psnr(cur_base, c0_ref)
fg_mse_base_tgt = ((cur_base - c0_tgt)**2)[:, :, fg_mask_tgt].mean().item()
sil_iou_base_tgt = calc_silhouette_iou(cur_base, c0_tgt)
leg_stats_base = calc_leg_displacement_stats(cur_base, c0_ref, c0_tgt)

psnr_ctrl_tgt = calc_psnr(cur_ctrl, c0_tgt)
ssim_ctrl_tgt = calc_ssim_simple(cur_ctrl, c0_tgt)
psnr_ctrl_ref = calc_psnr(cur_ctrl, c0_ref)
fg_mse_ctrl_tgt = ((cur_ctrl - c0_tgt)**2)[:, :, fg_mask_tgt].mean().item()
sil_iou_ctrl_tgt = calc_silhouette_iou(cur_ctrl, c0_tgt)
leg_stats_ctrl = calc_leg_displacement_stats(cur_ctrl, c0_ref, c0_tgt)

summary_rows = [
    {
        'Experiment': '1. Baseline (Gaussian from t=999)',
        'PSNR vs Target': f"{psnr_base_tgt:.2f} dB",
        'SSIM vs Target': f"{ssim_base_tgt:.4f}",
        'PSNR vs Ref': f"{psnr_base_ref:.2f} dB",
        'FG MSE vs Target': f"{fg_mse_base_tgt:.4f}",
        'Sil IoU vs Target': f"{sil_iou_base_tgt:.4f}",
        'Left Leg dx (GT=+0.40)': f"{leg_stats_base['pred_left_leg_dx']:+.2f}",
        'Visual Outcome': 'Failed TV static noise'
    },
    {
        'Experiment': '2. Target-Matched Control (Target @ t=500)',
        'PSNR vs Target': f"{psnr_ctrl_tgt:.2f} dB",
        'SSIM vs Target': f"{ssim_ctrl_tgt:.4f}",
        'PSNR vs Ref': f"{psnr_ctrl_ref:.2f} dB",
        'FG MSE vs Target': f"{fg_mse_ctrl_tgt:.4f}",
        'Sil IoU vs Target': f"{sil_iou_ctrl_tgt:.4f}",
        'Left Leg dx (GT=+0.40)': f"{leg_stats_ctrl['pred_left_leg_dx']:+.2f}",
        'Visual Outcome': 'Perfect target walk reconstruction'
    },
    {
        'Experiment': '3. Ref-Grounded Warm-Start (Ref @ t=500, Mean)',
        'PSNR vs Target': f"{df_exp3['psnr_vs_target'].mean():.2f} dB",
        'SSIM vs Target': f"{df_exp3['ssim_vs_target'].mean():.4f}",
        'PSNR vs Ref': f"{df_exp3['psnr_vs_ref'].mean():.2f} dB",
        'FG MSE vs Target': f"{df_exp3['fg_mse_vs_tgt'].mean():.4f}",
        'Sil IoU vs Target': f"{df_exp3['sil_iou_vs_tgt'].mean():.4f}",
        'Left Leg dx (GT=+0.40)': f"{df_exp3['pred_left_leg_dx'].mean():+.2f}",
        'Visual Outcome': 'Evaluated below'
    }
]

df_summary = pd.DataFrame(summary_rows)
df_summary.to_csv(os.path.join(EXP_A_DIR, "corrective_experiment_a_summary.csv"), index=False)
print("Saved corrective_experiment_a_summary.csv:")
print(df_summary.to_string(index=False))

# ======================================================================
# VISUAL CONTACT SHEET GENERATION
# ======================================================================
def tensor_to_rgba(t):
    arr = t.squeeze(0).permute(1, 2, 0).numpy()
    rgb = ((arr[:, :, :3] + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
    alpha = ((arr[:, :, 3] + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
    return Image.fromarray(np.dstack([rgb, alpha]), mode='RGBA')

img_ref = tensor_to_rgba(c0_ref)
img_tgt = tensor_to_rgba(c0_tgt)
img_base = tensor_to_rgba(cur_base)
img_ctrl = tensor_to_rgba(cur_ctrl)
img_warm0 = tensor_to_rgba(ref_warm_outputs[0])
img_warm1 = tensor_to_rgba(ref_warm_outputs[1])

fig, axes = plt.subplots(1, 5, figsize=(18, 4.5), dpi=200)

axes[0].imshow(img_ref)
axes[0].set_title("1. Ground Truth\nNeutral Reference", fontsize=10)
axes[0].axis('off')

axes[1].imshow(img_tgt)
axes[1].set_title("2. Ground Truth\nWalking Target (Apex F3)", fontsize=10)
axes[1].axis('off')

axes[2].imshow(img_base)
axes[2].set_title(f"3. Baseline DDIM (t=999)\nPSNR: {psnr_base_tgt:.1f} dB (TV Noise)", fontsize=10)
axes[2].axis('off')

axes[3].imshow(img_ctrl)
axes[3].set_title(f"4. Target Control (t=500)\nPSNR: {psnr_ctrl_tgt:.1f} dB (Reconstruction)", fontsize=10)
axes[3].axis('off')

axes[4].imshow(img_warm0)
axes[4].set_title(f"5. Ref Warm-Start (t=500)\nPSNR tgt: {df_exp3['psnr_vs_target'].iloc[0]:.1f} dB | ref: {df_exp3['psnr_vs_ref'].iloc[0]:.1f} dB", fontsize=10)
axes[4].axis('off')

plt.suptitle("Corrective Experiment A: Reference-Grounded Warm-Start Validation", fontsize=13, y=1.02)
contact_sheet_path = os.path.join(EXP_A_DIR, "corrective_experiment_a_contact_sheet.png")
plt.savefig(contact_sheet_path, bbox_inches='tight')
plt.close()
print(f"Saved contact sheet to: {contact_sheet_path}")

print("Experiment A Execution Completed Successfully.")

