import os, sys, math, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
import matplotlib.pyplot as plt

# Multi-threading & CPU settings
torch.manual_seed(42)
np.random.seed(42)
torch.set_num_threads(8)

DEVICE = torch.device('cpu')
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PHASE_F_DIR = os.path.join(ROOT_DIR, 'legacy_root_experiment_runs', 'phase_F_experiment_E1')
PHASE_F2_DIR = os.path.join(ROOT_DIR, 'legacy_root_experiment_runs', 'phase_F2_validation')
ARTIFACT_DIR = r"C:\Users\manis\.gemini\antigravity\brain\674ad8cd-af16-4b87-91e3-66973cf97bc5"

sys.path.insert(0, os.path.join(ROOT_DIR, 'backend'))
from multi_pose_diffusion_service import ActionAndPoseConditionedUNet, cosine_beta_schedule

TIMESTEPS = 1000
betas = cosine_beta_schedule(TIMESTEPS).to(DEVICE)
alphas = 1.0 - betas
alphas_bar = torch.cumprod(alphas, dim=0).to(DEVICE)

# ----------------------------------------------------------------------
# Metrics & Helper Functions
# ----------------------------------------------------------------------
def get_leg_centroid(tensor_chw, r0=38, r1=62, c0=16, c1=32):
    alpha = (tensor_chw[3] > 0.0).float().numpy()
    rgb = ((tensor_chw[:3] + 1.0) / 2.0).numpy()
    gray = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]
    mass = alpha[r0:r1, c0:c1] * (1.0 - 0.2 * gray[r0:r1, c0:c1])
    total_mass = mass.sum()
    if total_mass < 1e-4:
        return (c1 - c0) / 2.0, (r1 - r0) / 2.0
    ys, xs = np.indices(mass.shape)
    cx = float(np.sum(xs * mass) / total_mass)
    cy = float(np.sum(ys * mass) / total_mass)
    return cx, cy

def compute_psnr(img1, img2):
    mse = torch.mean((img1 - img2) ** 2).item()
    if mse < 1e-10:
        return 50.0
    return 10.0 * math.log10(4.0 / mse)

def compute_ssim(img1, img2):
    x = ((img1[:3] + 1.0) / 2.0).clamp(0, 1).unsqueeze(0)
    y = ((img2[:3] + 1.0) / 2.0).clamp(0, 1).unsqueeze(0)
    mu_x = x.mean()
    mu_y = y.mean()
    sigma_x = ((x - mu_x) ** 2).mean()
    sigma_y = ((y - mu_y) ** 2).mean()
    sigma_xy = ((x - mu_x) * (y - mu_y)).mean()
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    ssim = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / ((mu_x**2 + mu_y**2 + c1) * (sigma_x + sigma_y + c2))
    return float(ssim.item())

def compute_iou(img1, img2):
    m1 = img1[3] > 0.0
    m2 = img2[3] > 0.0
    intersection = (m1 & m2).sum().float().item()
    union = (m1 | m2).sum().float().item()
    return intersection / max(1.0, union)

def compute_seam_discontinuity(img_tensor, seam_row=38):
    # Measures the vertical L2 gradient jump across the seam boundary (row seam_row vs seam_row - 1)
    # relative to the normal row-to-row gradient in the torso
    t = (img_tensor + 1.0) / 2.0 # [4, 64, 64]
    diff_seam = torch.mean(torch.abs(t[:, seam_row, :] - t[:, seam_row - 1, :])).item()
    diff_baseline = torch.mean(torch.abs(t[:, seam_row - 2, :] - t[:, seam_row - 3, :])).item()
    seam_ratio = diff_seam / max(1e-4, diff_baseline)
    return diff_seam, seam_ratio

def tensor_to_pil(t_chw):
    t_np = ((t_chw.clamp(-1.0, 1.0) + 1.0) / 2.0 * 255.0).byte().permute(1, 2, 0).numpy()
    return Image.fromarray(t_np, mode='RGBA')

def ddim_sample(model, x_ref, action_id, pose_id, steps=25, seed=42):
    torch.manual_seed(seed)
    B = 1
    x_cur = torch.randn(B, 4, 64, 64, device=DEVICE)
    x_ref_b = x_ref.unsqueeze(0).to(DEVICE)
    act_t = torch.tensor([action_id], dtype=torch.long, device=DEVICE)
    pose_t = torch.tensor([pose_id], dtype=torch.long, device=DEVICE)
    
    step_ratio = TIMESTEPS // steps
    timesteps_seq = np.asarray(list(range(0, TIMESTEPS, step_ratio)))
    
    with torch.inference_mode():
        for i, step_idx in enumerate(reversed(range(len(timesteps_seq)))):
            t_idx = timesteps_seq[step_idx]
            t = torch.tensor([t_idx], device=DEVICE, dtype=torch.long)
            a_bar = alphas_bar[t_idx]
            a_bar_prev = alphas_bar[timesteps_seq[step_idx - 1]] if step_idx > 0 else torch.tensor(1.0, device=DEVICE)
            
            pred_v = model(x_cur, t, x_ref_b, act_t, pose_t)
            pred_x0 = torch.sqrt(a_bar) * x_cur - torch.sqrt(1.0 - a_bar) * pred_v
            pred_eps = torch.sqrt(1.0 - a_bar) * x_cur + torch.sqrt(a_bar) * pred_v
            pred_x0 = torch.clamp(pred_x0, -1.0, 1.0)
            
            dir_xt = torch.sqrt(torch.clamp(1.0 - a_bar_prev, min=0.0)) * pred_eps
            x_cur = torch.sqrt(a_bar_prev) * pred_x0 + dir_xt
            
    return torch.clamp(x_cur[0], -1.0, 1.0)

# ----------------------------------------------------------------------
# Patch Implementation Variants
# ----------------------------------------------------------------------
def apply_mirror_patch_hard(f7_tensor, f3_tensor, cut_row=38):
    # Mirror F3 horizontally
    f3_flipped = torch.flip(f3_tensor, dims=[-1])
    
    # Composite: Top rows from F7, bottom rows from flipped F3
    corrected = f7_tensor.clone()
    corrected[:, cut_row:, :] = f3_flipped[:, cut_row:, :]
    return corrected

def apply_mirror_patch_feathered(f7_tensor, f3_tensor, row_start=36, row_end=40):
    f3_flipped = torch.flip(f3_tensor, dims=[-1])
    corrected = f7_tensor.clone()
    
    # Rows below row_end are 100% flipped F3
    corrected[:, row_end:, :] = f3_flipped[:, row_end:, :]
    
    # Linear alpha blend across the transition band
    num_trans = row_end - row_start
    for i, r in enumerate(range(row_start, row_end)):
        weight_f3 = (i + 1) / float(num_trans + 1)
        corrected[:, r, :] = (1.0 - weight_f3) * f7_tensor[:, r, :] + weight_f3 * f3_flipped[:, r, :]
        
    return corrected

# ----------------------------------------------------------------------
# Main Execution
# ----------------------------------------------------------------------
def main():
    print("=" * 80)
    print("INVESTIGATION: DIFFUSION F7 RIGHT-LEG STRIDE & BILATERAL MIRROR PATCH")
    print("=" * 80)
    
    # 1. Load Model Checkpoint
    ckpt_path = os.path.join(ROOT_DIR, 'models', '5_Diffusion_LPC_v1', 'checkpoint_best.pt')
    if not os.path.exists(ckpt_path):
        ckpt_path = os.path.join(PHASE_F_DIR, 'checkpoint_best.pt')
    print(f"Loading checkpoint: {ckpt_path}")
    
    model = ActionAndPoseConditionedUNet()
    model.load_state_dict(torch.load(ckpt_path, map_location='cpu'))
    model.eval()
    print("Model loaded successfully.")
    
    # 2. Load Manifest & Validation Cohort
    manifest_path = os.path.join(PHASE_F_DIR, 'dataset_manifest.csv')
    df = pd.read_csv(manifest_path)
    val_df = df[df['split'] == 'val']
    val_chars = sorted(val_df['character_id'].unique())
    print(f"Total Validation Characters Available: {len(val_chars)} ({val_chars[0]} to {val_chars[-1]})")
    
    # We will evaluate across 20 validation characters (statistically representative sample)
    test_chars = val_chars[:20]
    print(f"Evaluating detailed patch metrics across {len(test_chars)} validation identities...")
    
    records = []
    comparison_samples = [] # for visualization
    
    for idx, cid in enumerate(test_chars):
        c_rows = val_df[val_df['character_id'] == cid]
        ref_path = c_rows[(c_rows['action_id']==0) & (c_rows['pose_id']==0)].iloc[0]['target_path']
        gt_f3_path = c_rows[(c_rows['action_id']==1) & (c_rows['pose_id']==2)].iloc[0]['target_path']
        gt_f7_path = c_rows[(c_rows['action_id']==1) & (c_rows['pose_id']==6)].iloc[0]['target_path']
        
        # Load Reference and GT tensors [-1.0, 1.0]
        ref_im = Image.open(ref_path).convert('RGBA')
        ref_t = torch.from_numpy(np.array(ref_im, dtype=np.float32) / 127.5 - 1.0).permute(2, 0, 1)
        
        gt_f7_im = Image.open(gt_f7_path).convert('RGBA')
        gt_f7_t = torch.from_numpy(np.array(gt_f7_im, dtype=np.float32) / 127.5 - 1.0).permute(2, 0, 1)
        
        gt_f3_im = Image.open(gt_f3_path).convert('RGBA')
        gt_f3_t = torch.from_numpy(np.array(gt_f3_im, dtype=np.float32) / 127.5 - 1.0).permute(2, 0, 1)
        
        # Generate F3 (Pose 2) and F7 (Pose 6) from identical reference
        gen_f3_t = ddim_sample(model, ref_t, action_id=1, pose_id=2, steps=25, seed=42)
        gen_f7_orig_t = ddim_sample(model, ref_t, action_id=1, pose_id=6, steps=25, seed=42)
        
        # Apply Mirror Patches
        gen_f7_hard38_t = apply_mirror_patch_hard(gen_f7_orig_t, gen_f3_t, cut_row=38)
        gen_f7_hard40_t = apply_mirror_patch_hard(gen_f7_orig_t, gen_f3_t, cut_row=40)
        gen_f7_feather_t = apply_mirror_patch_feathered(gen_f7_orig_t, gen_f3_t, row_start=36, row_end=40)
        
        # Centroid Displacements
        ref_l_cx, _ = get_leg_centroid(ref_t, 38, 62, 16, 32)
        ref_r_cx, _ = get_leg_centroid(ref_t, 38, 62, 32, 48)
        
        gt_l_cx, _ = get_leg_centroid(gt_f7_t, 38, 62, 16, 32)
        gt_r_cx, _ = get_leg_centroid(gt_f7_t, 38, 62, 32, 48)
        gt_r_dx = gt_r_cx - ref_r_cx
        gt_l_dx = gt_l_cx - ref_l_cx
        gt_spread = gt_l_dx - gt_r_dx
        
        def evaluate_candidate(cand_t, name):
            cand_l_cx, _ = get_leg_centroid(cand_t, 38, 62, 16, 32)
            cand_r_cx, _ = get_leg_centroid(cand_t, 38, 62, 32, 48)
            cand_r_dx = cand_r_cx - ref_r_cx
            cand_l_dx = cand_l_cx - ref_l_cx
            err_r = abs(cand_r_dx - gt_r_dx)
            err_l = abs(cand_l_dx - gt_l_dx)
            cand_spread = cand_l_dx - cand_r_dx
            comp_pct = (1.0 - cand_spread / gt_spread) * 100.0 if gt_spread != 0 else 0.0
            
            psnr_v = compute_psnr(cand_t, gt_f7_t)
            ssim_v = compute_ssim(cand_t, gt_f7_t)
            iou_v = compute_iou(cand_t, gt_f7_t)
            seam_diff, seam_ratio = compute_seam_discontinuity(cand_t, seam_row=38)
            
            return {
                'character_id': cid,
                'candidate': name,
                'gt_r_dx': gt_r_dx,
                'cand_r_dx': cand_r_dx,
                'error_r_dx': err_r,
                'error_l_dx': err_l,
                'gt_spread': gt_spread,
                'cand_spread': cand_spread,
                'compression_pct': comp_pct,
                'psnr': psnr_v,
                'ssim': ssim_v,
                'iou': iou_v,
                'seam_diff': seam_diff,
                'seam_ratio': seam_ratio
            }
            
        r_orig = evaluate_candidate(gen_f7_orig_t, "Original Gen F7")
        r_hard38 = evaluate_candidate(gen_f7_hard38_t, "Mirror Hard (Row 38)")
        r_hard40 = evaluate_candidate(gen_f7_hard40_t, "Mirror Hard (Row 40)")
        r_feather = evaluate_candidate(gen_f7_feather_t, "Mirror Feathered (36-40)")
        
        records.extend([r_orig, r_hard38, r_hard40, r_feather])
        
        if idx < 6:
            comparison_samples.append({
                'cid': cid,
                'ref': tensor_to_pil(ref_t),
                'gt_f7': tensor_to_pil(gt_f7_t),
                'orig_f7': tensor_to_pil(gen_f7_orig_t),
                'corr_f7': tensor_to_pil(gen_f7_feather_t),
                'f3_gen': tensor_to_pil(gen_f3_t)
            })
            
        print(f"Processed [{idx+1}/{len(test_chars)}] {cid}: Orig Err_R={r_orig['error_r_dx']:.3f} px | Corr Err_R={r_feather['error_r_dx']:.3f} px")
        
    df_results = pd.DataFrame(records)
    
    print("\n" + "=" * 80)
    print("STATISTICAL COMPARISON TABLE (20 VALIDATION IDENTITIES)")
    print("=" * 80)
    summary = df_results.groupby('candidate').agg({
        'error_r_dx': 'mean',
        'error_l_dx': 'mean',
        'compression_pct': 'mean',
        'psnr': 'mean',
        'ssim': 'mean',
        'iou': 'mean',
        'seam_ratio': 'mean'
    }).reset_index()
    
    print(summary.to_string(index=False))
    
    # ------------------------------------------------------------------
    # Generate Visual Side-by-Side Comparison Grid
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(len(comparison_samples), 6, figsize=(15, 2.5 * len(comparison_samples)))
    col_titles = [
        "Reference Stance\n(Idle Stance)",
        "Ground Truth F7\n(LPC Target)",
        "Original Gen F7\n(38.9% Compression)",
        "Gen F3 (Left Apex)\n(Source of Mirror)",
        "Corrected Gen F7\n(Bilateral Feathered)",
        "Pixel Difference Map\n(Corr vs GT)"
    ]
    
    for row_idx, sample in enumerate(comparison_samples):
        # 1. Ref
        axes[row_idx, 0].imshow(sample['ref'])
        axes[row_idx, 0].set_ylabel(sample['cid'], fontsize=11, fontweight='bold')
        
        # 2. GT F7
        axes[row_idx, 1].imshow(sample['gt_f7'])
        
        # 3. Orig F7
        axes[row_idx, 2].imshow(sample['orig_f7'])
        
        # 4. Gen F3
        axes[row_idx, 3].imshow(sample['f3_gen'])
        
        # 5. Corr F7
        axes[row_idx, 4].imshow(sample['corr_f7'])
        
        # 6. Diff Map between Corrected F7 and GT F7
        gt_arr = np.array(sample['gt_f7'], dtype=np.float32)
        corr_arr = np.array(sample['corr_f7'], dtype=np.float32)
        diff = np.abs(corr_arr - gt_arr)[:, :, :3].mean(axis=-1)
        im_diff = axes[row_idx, 6 - 1].imshow(diff, cmap='inferno', vmin=0, vmax=80)
        
        for col_idx in range(6):
            if row_idx == 0:
                axes[row_idx, col_idx].set_title(col_titles[col_idx], fontsize=10, fontweight='bold', pad=8)
            axes[row_idx, col_idx].set_xticks([])
            axes[row_idx, col_idx].set_yticks([])
            
    plt.tight_layout()
    out_img_path = os.path.join(ARTIFACT_DIR, "diffusion_f7_bilateral_mirror_comparison.png")
    plt.savefig(out_img_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"\nGenerated & Saved Comparison Grid to: {out_img_path}")
    
    # Also save summary JSON
    summary_dict = summary.to_dict(orient='records')
    import json
    with open(os.path.join(ARTIFACT_DIR, "diffusion_f7_patch_evaluation.json"), "w") as f:
        json.dump(summary_dict, f, indent=2)
    print("Saved evaluation metrics JSON.")

if __name__ == '__main__':
    main()
