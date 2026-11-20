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
from datasets import load_dataset

# Add scripts directory to path
sys.path.append("scripts")
from run_rung2_ddpm_overfit import ReferenceConditionedUNet, cosine_beta_schedule, TIMESTEPS

DEVICE = torch.device('cpu')
OUTPUT_DIR = "research_paper_assets/06_sprite_deformation_and_pose_animation"
CHECKPOINT_PATH = os.path.join(OUTPUT_DIR, "rung2_ddpm_overfit_best.pt")

print("======================================================================")
print("     FORENSIC DEBUGGING INVESTIGATION: RUNG 2 FAILURE AUDIT           ")
print("======================================================================")

# ----------------------------------------------------------------------
# TEST 1: Forward Diffusion & Schedule Verification
# ----------------------------------------------------------------------
betas = cosine_beta_schedule(TIMESTEPS)
alphas = 1.0 - betas
alphas_bar = torch.cumprod(alphas, dim=0)

print("\n--- TEST 1: Forward Diffusion Schedule Verification ---")
test_ts = [20, 100, 250, 500, 750, 999]
for t in test_ts:
    print(f"t={t:3d} | beta={betas[t].item():.6f} | alpha_bar={alphas_bar[t].item():.6f} | sqrt(alpha_bar)={math.sqrt(alphas_bar[t].item()):.4f} | sqrt(1-alpha_bar)={math.sqrt(1.0 - alphas_bar[t].item()):.4f}")

# ----------------------------------------------------------------------
# Load Dataset & C_00
# ----------------------------------------------------------------------
ds = load_dataset("carlosuperb/lpc-action-pixel-art-diffusion", split="train", streaming=True)
sheet = next(iter(ds))['image'].convert('RGBA')
if sheet.size != (576, 768):
    sheet = sheet.resize((576, 768), Image.Resampling.NEAREST)

ref_crop = sheet.crop((0 * 64, 2 * 64, 1 * 64, 3 * 64))
walk_crop = sheet.crop((3 * 64, 2 * 64, 4 * 64, 3 * 64))

ref_t = torch.tensor(np.array(ref_crop, dtype=np.float32) / 127.5 - 1.0).permute(2, 0, 1).unsqueeze(0)
tgt_t = torch.tensor(np.array(walk_crop, dtype=np.float32) / 127.5 - 1.0).permute(2, 0, 1).unsqueeze(0)

# ----------------------------------------------------------------------
# TEST 3: Oracle DDIM Sampler Verification
# ----------------------------------------------------------------------
print("\n--- TEST 3: Oracle DDIM Sampler Verification ---")
torch.manual_seed(42)
true_noise = torch.randn_like(tgt_t)

# Noised target at t=980
x_980 = torch.sqrt(alphas_bar[980]) * tgt_t + torch.sqrt(1.0 - alphas_bar[980]) * true_noise

# Test single step inversion
pred_x0_direct = (x_980 - torch.sqrt(1.0 - alphas_bar[980]) * true_noise) / torch.sqrt(alphas_bar[980])
direct_err = torch.abs(pred_x0_direct - tgt_t).max().item()
print(f"Direct Single-Step Oracle Reconstruction Max Error: {direct_err:.8f}")

# Test 50-step trajectory with Oracle epsilon
DDIM_STEPS = 50
step_ratio = TIMESTEPS // DDIM_STEPS
ddim_timesteps = np.asarray(list(range(0, TIMESTEPS, step_ratio)))

cur_oracle_x = x_980.clone()
for i, step in enumerate(reversed(range(len(ddim_timesteps)))):
    t_idx = ddim_timesteps[step]
    prev_idx = ddim_timesteps[step - 1] if step > 0 else -1
    alpha_bar = alphas_bar[t_idx]
    alpha_bar_prev = alphas_bar[prev_idx] if step > 0 else torch.tensor(1.0)
    
    # Injected oracle epsilon for current x
    # Forward formula: x_t = sqrt(ab)*x0 + sqrt(1-ab)*eps => eps = (x_t - sqrt(ab)*x0)/sqrt(1-ab)
    exact_eps = (cur_oracle_x - torch.sqrt(alpha_bar) * tgt_t) / torch.sqrt(1.0 - alpha_bar)
    
    pred_x0 = (cur_oracle_x - torch.sqrt(1.0 - alpha_bar) * exact_eps) / torch.sqrt(alpha_bar)
    pred_x0 = torch.clamp(pred_x0, -1.0, 1.0)
    dir_xt = torch.sqrt(1.0 - alpha_bar_prev) * exact_eps
    cur_oracle_x = torch.sqrt(alpha_bar_prev) * pred_x0 + dir_xt

oracle_max_err = torch.abs(cur_oracle_x - tgt_t).max().item()
print(f"50-Step Oracle DDIM Trajectory Reconstruction Max Error: {oracle_max_err:.8f}")

# Save Oracle Reconstructed Image
oracle_img = ((cur_oracle_x.squeeze(0).permute(1, 2, 0).numpy() + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
Image.fromarray(oracle_img, mode='RGBA').save(os.path.join(OUTPUT_DIR, "test3_oracle_reconstruction.png"))
print("Saved test3_oracle_reconstruction.png")

# ----------------------------------------------------------------------
# TEST 2: Model Noise Prediction Audit
# ----------------------------------------------------------------------
print("\n--- TEST 2: Model Noise Prediction Audit ---")
model = ReferenceConditionedUNet().to(DEVICE)
state = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
model.load_state_dict(state)
model.eval()

# Foreground mask: alpha in target > -0.8
fg_mask = (tgt_t[:, 3:4] > -0.8).squeeze() # [64, 64]
bg_mask = ~fg_mask

print(f"{'Timestep':<10} | {'alpha_bar':<10} | {'Total MSE':<12} | {'FG MSE':<12} | {'BG MSE':<12}")
print("-" * 62)

for t_val in [999, 750, 500, 250, 100, 20]:
    t_tensor = torch.tensor([t_val])
    x_t = torch.sqrt(alphas_bar[t_val]) * tgt_t + torch.sqrt(1.0 - alphas_bar[t_val]) * true_noise
    
    with torch.no_grad():
        pred_eps = model(x_t, t_tensor, ref_t)
        
    mse_total = torch.mean((pred_eps - true_noise) ** 2).item()
    diff_sq = (pred_eps - true_noise) ** 2 # [1, 4, 64, 64]
    mse_fg = diff_sq[:, :, fg_mask].mean().item()
    mse_bg = diff_sq[:, :, bg_mask].mean().item()
    
    print(f"t={t_val:<8d} | {alphas_bar[t_val].item():<10.4f} | {mse_total:<12.4f} | {mse_fg:<12.4f} | {mse_bg:<12.4f}")

# ----------------------------------------------------------------------
# TEST 4: Single-Character Fixed-Noise Deterministic Overfit Test
# ----------------------------------------------------------------------
print("\n--- TEST 4: Single-Character Fixed-Noise Deterministic Overfit ---")
# Train a fresh tiny U-Net on EXACTLY ONE (t=500, fixed noise, fixed target, fixed ref)
toy_model = ReferenceConditionedUNet().to(DEVICE)
toy_opt = torch.optim.Adam(toy_model.parameters(), lr=1e-3)

fixed_t = torch.tensor([500])
fixed_noise = true_noise.clone()
fixed_xt = torch.sqrt(alphas_bar[500]) * tgt_t + torch.sqrt(1.0 - alphas_bar[500]) * fixed_noise

print("Training toy model on fixed tuple (x_t, t=500, ref) -> known noise...")
for step in range(1, 101):
    toy_model.train()
    p_eps = toy_model(fixed_xt, fixed_t, ref_t)
    loss = F.mse_loss(p_eps, fixed_noise)
    toy_opt.zero_grad()
    loss.backward()
    toy_opt.step()
    if step % 25 == 0:
        print(f"  Step [{step:03d}/100] | MSE Loss: {loss.item():.6f}")

print(f"Final deterministic fixed-noise prediction MSE: {loss.item():.6f}")

# ----------------------------------------------------------------------
# TEST 5: Reverse Trajectory Forensic Tracing
# ----------------------------------------------------------------------
print("\n--- TEST 5: Reverse Trajectory Forensic Tracing ---")
# Trace actual trained model reverse DDIM sampling from random noise
torch.manual_seed(42)
test_x = torch.randn_like(tgt_t)

trajectory_snapshots = {}
record_steps = [980, 760, 500, 260, 100, 0]

for i, step in enumerate(reversed(range(len(ddim_timesteps)))):
    t_idx = ddim_timesteps[step]
    prev_idx = ddim_timesteps[step - 1] if step > 0 else -1
    alpha_bar = alphas_bar[t_idx]
    alpha_bar_prev = alphas_bar[prev_idx] if step > 0 else torch.tensor(1.0)
    
    t_tensor = torch.full((1,), t_idx, device=DEVICE, dtype=torch.long)
    with torch.no_grad():
        pred_eps = model(test_x, t_tensor, ref_t)
        
    pred_x0 = (test_x - torch.sqrt(1.0 - alpha_bar) * pred_eps) / torch.sqrt(alpha_bar)
    pred_x0 = torch.clamp(pred_x0, -1.0, 1.0)
    dir_xt = torch.sqrt(1.0 - alpha_bar_prev) * pred_eps
    test_x = torch.sqrt(alpha_bar_prev) * pred_x0 + dir_xt
    
    if t_idx in record_steps or step == 0:
        # Record reconstructed x0 at this timestep
        x0_arr = ((pred_x0.squeeze(0).permute(1, 2, 0).numpy() + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
        xt_arr = ((test_x.squeeze(0).permute(1, 2, 0).numpy() + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
        trajectory_snapshots[t_idx] = {'x0': x0_arr, 'xt': xt_arr}

# Build trajectory contact sheet
fig, axes = plt.subplots(2, 6, figsize=(16, 6), dpi=150)
for idx, t_val in enumerate(record_steps):
    snap = trajectory_snapshots.get(t_val, list(trajectory_snapshots.values())[0])
    axes[0, idx].imshow(snap['xt'])
    axes[0, idx].set_title(f"x_t (t={t_val})", fontsize=9)
    axes[0, idx].axis('off')
    
    axes[1, idx].imshow(snap['x0'])
    axes[1, idx].set_title(f"Pred x_0 at t={t_val}", fontsize=9)
    axes[1, idx].axis('off')

plt.suptitle("Rung 2 Trajectory Breakdown: x_t (Noisy State) vs. Predicted x_0 across Timesteps", fontsize=12)
trajectory_path = os.path.join(OUTPUT_DIR, "test5_reverse_trajectory_breakdown.png")
plt.savefig(trajectory_path, bbox_inches='tight')
plt.close()
print(f"Saved reverse trajectory contact sheet to: {trajectory_path}")

# ----------------------------------------------------------------------
# TEST 6: Checkpoint & Loss History Audit
# ----------------------------------------------------------------------
print("\n--- TEST 6: Checkpoint & Loss History Audit ---")
loss_csv = pd.read_csv(os.path.join(OUTPUT_DIR, "rung2_ddpm_loss_history.csv"))
min_row = loss_csv.loc[loss_csv['loss'].idxmin()]
print(f"Total Epochs in CSV: {len(loss_csv)}")
print(f"Minimum Loss: {min_row['loss']:.6f} at Epoch {int(min_row['epoch'])}")
print(f"Final Epoch 150 Loss: {loss_csv['loss'].iloc[-1]:.6f}")

print("\n======================================================================")
print("                    FORENSIC INVESTIGATION COMPLETE                   ")
print("======================================================================")

