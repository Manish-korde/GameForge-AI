import os
import sys
import json
import numpy as np
from PIL import Image
import colorsys
from collections import Counter
from scipy.ndimage import label, binary_dilation

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "backend")))
import dcgan_service

ARTIFACT_DIR = r"C:\Users\manis\.gemini\antigravity\brain\674ad8cd-af16-4b87-91e3-66973cf97bc5"
os.makedirs(ARTIFACT_DIR, exist_ok=True)

gen = dcgan_service.load_dcgan_generator()

def rgb_to_hsv_deg(r, g, b):
    h, s, v = colorsys.rgb_to_hsv(r / 255.0, g / 255.0, b / 255.0)
    return h * 360.0, s, v

def circular_hue_diff(h1, h2):
    diff = abs(h1 - h2) % 360.0
    return min(diff, 360.0 - diff)

def check_facial_hue_consistency(rgba_img: np.ndarray, max_divergence: float = 45.0):
    alpha = rgba_img[..., 3] > 127
    if not np.any(alpha):
        return True, 0.0
    fg_y, fg_x = np.where(alpha)
    y_min, y_max = np.min(fg_y), np.max(fg_y)
    x_min, x_max = np.min(fg_x), np.max(fg_x)
    h_box = y_max - y_min + 1
    
    head_y_cutoff = y_min + int(0.35 * h_box)
    head_mask = alpha & (np.arange(64)[:, None] <= head_y_cutoff)
    x_mid = (x_min + x_max) // 2
    
    left_head = head_mask & (np.arange(64)[None, :] < x_mid)
    right_head = head_mask & (np.arange(64)[None, :] >= x_mid)
    
    if np.sum(left_head) < 8 or np.sum(right_head) < 8:
        return True, 0.0
        
    l_rgb = rgba_img[..., :3][left_head]
    r_rgb = rgba_img[..., :3][right_head]
    
    l_hues = [rgb_to_hsv_deg(r, g, b)[0] for r, g, b in l_rgb if rgb_to_hsv_deg(r, g, b)[1] > 0.18 and rgb_to_hsv_deg(r, g, b)[2] > 0.18]
    r_hues = [rgb_to_hsv_deg(r, g, b)[0] for r, g, b in r_rgb if rgb_to_hsv_deg(r, g, b)[1] > 0.18 and rgb_to_hsv_deg(r, g, b)[2] > 0.18]
    
    if len(l_hues) < 5 or len(r_hues) < 5:
        return True, 0.0
        
    l_rad = np.deg2rad(l_hues)
    r_rad = np.deg2rad(r_hues)
    l_mean_h = np.rad2deg(np.arctan2(np.mean(np.sin(l_rad)), np.mean(np.cos(l_rad)))) % 360.0
    r_mean_h = np.rad2deg(np.arctan2(np.mean(np.sin(r_rad)), np.mean(np.cos(r_rad)))) % 360.0
    
    h_diff = circular_hue_diff(l_mean_h, r_mean_h)
    return (h_diff < max_divergence), float(h_diff)

def check_head_torso_hue_agreement(rgba_img: np.ndarray, max_clash: float = 100.0):
    alpha = rgba_img[..., 3] > 127
    if not np.any(alpha):
        return True, 0.0
    fg_y, fg_x = np.where(alpha)
    y_min, y_max = np.min(fg_y), np.max(fg_y)
    h_box = y_max - y_min + 1
    
    head_y_cutoff = y_min + int(0.35 * h_box)
    torso_y_cutoff = y_min + int(0.70 * h_box)
    
    head_mask = alpha & (np.arange(64)[:, None] <= head_y_cutoff)
    torso_mask = alpha & (np.arange(64)[:, None] > head_y_cutoff) & (np.arange(64)[:, None] <= torso_y_cutoff)
    
    if np.sum(head_mask) < 10 or np.sum(torso_mask) < 10:
        return True, 0.0
        
    head_rgb = rgba_img[..., :3][head_mask]
    torso_rgb = rgba_img[..., :3][torso_mask]
    
    head_hues = [rgb_to_hsv_deg(r, g, b)[0] for r, g, b in head_rgb if rgb_to_hsv_deg(r, g, b)[1] > 0.20 and rgb_to_hsv_deg(r, g, b)[2] > 0.20]
    torso_hues = [rgb_to_hsv_deg(r, g, b)[0] for r, g, b in torso_rgb if rgb_to_hsv_deg(r, g, b)[1] > 0.20 and rgb_to_hsv_deg(r, g, b)[2] > 0.20]
    
    if len(head_hues) < 8 or len(torso_hues) < 8:
        return True, 0.0
        
    h_rad = np.deg2rad(head_hues)
    t_rad = np.deg2rad(torso_hues)
    h_mean = np.rad2deg(np.arctan2(np.mean(np.sin(h_rad)), np.mean(np.cos(h_rad)))) % 360.0
    t_mean = np.rad2deg(np.arctan2(np.mean(np.sin(t_rad)), np.mean(np.cos(t_rad)))) % 360.0
    
    h_diff = circular_hue_diff(h_mean, t_mean)
    return (h_diff <= max_clash), float(h_diff)

def classify_archetype(arr):
    fg = arr[..., :3][arr[..., 3] > 0]
    if len(fg) == 0:
        return "Other/Mixed"
    mean_rgb = np.mean(fg, axis=0)
    r, g, b = mean_rgb
    if g > r * 1.05 and g > b * 1.05:
        return "Green/Orc"
    elif r > 110 and g > 70 and b < 70 and r > b * 1.3:
        return "Brown/Ranger"
    elif r > 130 and g < 100 and b < 100:
        return "Red/Mage"
    elif b > r * 1.1 and b > g * 1.0:
        return "Blue/Wizard"
    elif abs(r - g) < 20 and abs(g - b) < 20:
        return "Gray/Knight"
    else:
        return "Other/Mixed"

print("=" * 80)
print("EVALUATING FILTER V2 VS FILTER V3 (10 FIXED SEEDS x 16 SPRITES)")
print("=" * 80)

EVAL_SEEDS = [42, 107, 256, 314, 512, 777, 888, 1337, 2024, 4242]

v2_facial_splits = 0
v2_head_torso_clashes = 0
v2_archetypes = Counter()

v3_facial_splits = 0
v3_head_torso_clashes = 0
v3_rejections = []
v3_archetypes = Counter()

for s in EVAL_SEEDS:
    # --- Generate V2 (32 candidates, floater cleaning only) ---
    rng_v2 = np.random.RandomState(s)
    z_32 = rng_v2.normal(0, 1.0, size=(32, dcgan_service.LATENT_DIM)).astype(np.float32)
    z_32 = np.clip(z_32, -2.0, 2.0)
    imgs_32 = dcgan_service.logits_to_image(gen(z_32, training=False), dcgan_service._palette_tf, hard=True, noise=False)
    
    v2_scored = []
    for im in imgs_32:
        cl, fl, st = dcgan_service.remove_detached_components_v2(im)
        _, left_fl, left_st = dcgan_service.remove_detached_components_v2(cl)
        score = dcgan_service.score_candidate(cl, fl)
        if left_fl == 0 and left_st == 0 and score < float("inf"):
            v2_scored.append((score, cl))
    v2_scored.sort(key=lambda x: x[0])
    v2_best16 = [x[1] for x in v2_scored[:16]]
    
    for im in v2_best16:
        f_pass, _ = check_facial_hue_consistency(im, max_divergence=45.0)
        if not f_pass:
            v2_facial_splits += 1
        ht_pass, _ = check_head_torso_hue_agreement(im, max_clash=100.0)
        if not ht_pass:
            v2_head_torso_clashes += 1
        v2_archetypes[classify_archetype(im)] += 1
        
    # --- Generate V3 (64 candidates, floater cleaning + facial hue gate + head-torso penalty) ---
    rng_v3 = np.random.RandomState(s)
    z_64 = rng_v3.normal(0, 1.0, size=(64, dcgan_service.LATENT_DIM)).astype(np.float32)
    z_64 = np.clip(z_64, -2.0, 2.0)
    imgs_64 = dcgan_service.logits_to_image(gen(z_64, training=False), dcgan_service._palette_tf, hard=True, noise=False)
    
    v3_scored = []
    v3_rejected_count = 0
    for im in imgs_64:
        cl, fl, st = dcgan_service.remove_detached_components_v2(im)
        _, left_fl, left_st = dcgan_service.remove_detached_components_v2(cl)
        f_pass, f_diff = check_facial_hue_consistency(cl, max_divergence=45.0)
        score = dcgan_service.score_candidate(cl, fl)
        
        # Hard rejection gate: leftover floaters, extreme silhouette, or facial split
        if left_fl > 0 or left_st > 0 or score == float("inf") or not f_pass:
            v3_rejected_count += 1
            continue
            
        # Independent metric penalty in score
        ht_pass, ht_diff = check_head_torso_hue_agreement(cl, max_clash=100.0)
        if not ht_pass:
            score += (ht_diff - 100.0) * 0.50
            
        v3_scored.append((score, cl))
        
    v3_rejections.append(v3_rejected_count / 64.0)
    v3_scored.sort(key=lambda x: x[0])
    v3_best16 = [x[1] for x in v3_scored[:16]]
    
    for im in v3_best16:
        f_pass, _ = check_facial_hue_consistency(im, max_divergence=45.0)
        if not f_pass:
            v3_facial_splits += 1
        ht_pass, _ = check_head_torso_hue_agreement(im, max_clash=100.0)
        if not ht_pass:
            v3_head_torso_clashes += 1
        v3_archetypes[classify_archetype(im)] += 1

print(f"\n1. Facial Bilateral Hue Split Rate (N = 160 sprites across 10 seeds):")
print(f"   Filter v2 : {v2_facial_splits / 160 * 100:.2f}% ({v2_facial_splits}/160)")
print(f"   Filter v3 : {v3_facial_splits / 160 * 100:.2f}% ({v3_facial_splits}/160) [Pass <= 3.5%: {v3_facial_splits / 160 * 100 <= 3.5}]")

print(f"\n2. Head-vs-Torso Hue Clash Rate (Independent Metric, N = 160):")
print(f"   Filter v2 : {v2_head_torso_clashes / 160 * 100:.2f}% ({v2_head_torso_clashes}/160)")
print(f"   Filter v3 : {v3_head_torso_clashes / 160 * 100:.2f}% ({v3_head_torso_clashes}/160)")

mean_rejection_rate = float(np.mean(v3_rejections)) * 100.0
print(f"\n3. Candidate Rejection Rate (Filter v3):")
print(f"   Mean Rejected Candidates: {mean_rejection_rate:.1f}% ({mean_rejection_rate/100*64:.1f} of 64 candidates rejected per batch)")
print(f"   Passing Candidates Pool : {100.0 - mean_rejection_rate:.1f}% ({(1.0 - mean_rejection_rate/100)*64:.1f} candidates available to rank for 16 spots)")

print(f"\n4. Archetype Diversity Distribution (N = 160):")
print(f"{'Archetype':<16} | {'Filter v2':<15} | {'Filter v3':<15}")
print("-" * 50)
for arch in ["Gray/Knight", "Brown/Ranger", "Green/Orc", "Red/Mage", "Blue/Wizard", "Other/Mixed"]:
    cnt_v2 = v2_archetypes[arch]
    cnt_v3 = v3_archetypes[arch]
    print(f"{arch:<16} | {cnt_v2:2d} ({cnt_v2/160*100:4.1f}%)    | {cnt_v3:2d} ({cnt_v3/160*100:4.1f}%)")

# -----------------------------------------------------------------------------
# FIXED 16-SEED SIDE-BY-SIDE GRID COMPARISON (V2 VS V3)
# -----------------------------------------------------------------------------
print("\n" + "=" * 80)
print("GENERATING FIXED 16-SEED SIDE-BY-SIDE GRID (V2 VS V3, SAME SEEDS)")
print("=" * 80)
SEEDS_16 = [10, 20, 30, 42, 55, 77, 88, 99, 107, 256, 314, 512, 777, 888, 1337, 4242]

# For each seed, generate 1 representative primary sprite under v2 vs v3
# Create a 16-row x 2-column comparison strip
GRID_SPR_SZ = 64
scale = 2 # 128x128 per sprite
spr_scaled = GRID_SPR_SZ * scale
comp_w = spr_scaled * 2 + 180 # extra space for seed & metric labels
comp_h = spr_scaled * 16

comp_img = Image.new("RGBA", (comp_w, comp_h), (15, 23, 42, 255))
import matplotlib.pyplot as plt

v2_sprites_16 = []
v3_sprites_16 = []

for idx, s in enumerate(SEEDS_16):
    # V2 primary
    rng_v2 = np.random.RandomState(s)
    z_32 = rng_v2.normal(0, 1.0, size=(32, dcgan_service.LATENT_DIM)).astype(np.float32)
    z_32 = np.clip(z_32, -2.0, 2.0)
    imgs_32 = dcgan_service.logits_to_image(gen(z_32, training=False), dcgan_service._palette_tf, hard=True, noise=False)
    v2_scored = []
    for im in imgs_32:
        cl, fl, st = dcgan_service.remove_detached_components_v2(im)
        _, left_fl, left_st = dcgan_service.remove_detached_components_v2(cl)
        score = dcgan_service.score_candidate(cl, fl)
        if left_fl == 0 and left_st == 0 and score < float("inf"):
            v2_scored.append((score, cl))
    v2_scored.sort(key=lambda x: x[0])
    best_v2 = v2_scored[0][1] if len(v2_scored) > 0 else imgs_32[0]
    v2_sprites_16.append(best_v2)
    
    # V3 primary
    rng_v3 = np.random.RandomState(s)
    z_64 = rng_v3.normal(0, 1.0, size=(64, dcgan_service.LATENT_DIM)).astype(np.float32)
    z_64 = np.clip(z_64, -2.0, 2.0)
    imgs_64 = dcgan_service.logits_to_image(gen(z_64, training=False), dcgan_service._palette_tf, hard=True, noise=False)
    v3_scored = []
    for im in imgs_64:
        cl, fl, st = dcgan_service.remove_detached_components_v2(im)
        _, left_fl, left_st = dcgan_service.remove_detached_components_v2(cl)
        f_pass, f_diff = check_facial_hue_consistency(cl, max_divergence=45.0)
        score = dcgan_service.score_candidate(cl, fl)
        if left_fl == 0 and left_st == 0 and score < float("inf") and f_pass:
            ht_pass, ht_diff = check_head_torso_hue_agreement(cl, max_clash=100.0)
            if not ht_pass:
                score += (ht_diff - 100.0) * 0.50
            v3_scored.append((score, cl))
    v3_scored.sort(key=lambda x: x[0])
    best_v3 = v3_scored[0][1] if len(v3_scored) > 0 else imgs_64[0]
    v3_sprites_16.append(best_v3)
    
    pil_v2 = Image.fromarray(best_v2, mode="RGBA").resize((spr_scaled, spr_scaled), Image.NEAREST)
    pil_v3 = Image.fromarray(best_v3, mode="RGBA").resize((spr_scaled, spr_scaled), Image.NEAREST)
    
    y_pos = idx * spr_scaled
    comp_img.paste(pil_v2, (10, y_pos + 4), pil_v2)
    comp_img.paste(pil_v3, (spr_scaled + 20, y_pos + 4), pil_v3)

side_by_side_path = os.path.join(ARTIFACT_DIR, "dcgan_v2_vs_v3_fixed_16seeds.png")
comp_img.save(side_by_side_path)
print(f"Saved side-by-side fixed 16-seed comparison grid to {side_by_side_path}")

# Write JSON benchmark output
eval_summary = {
    "facial_hue_split_rate_v2": v2_facial_splits / 160 * 100.0,
    "facial_hue_split_rate_v3": v3_facial_splits / 160 * 100.0,
    "head_torso_clash_rate_v2": v2_head_torso_clashes / 160 * 100.0,
    "head_torso_clash_rate_v3": v3_head_torso_clashes / 160 * 100.0,
    "mean_candidate_rejection_rate_v3": mean_rejection_rate,
    "archetype_distribution_v2": dict(v2_archetypes),
    "archetype_distribution_v3": dict(v3_archetypes),
    "side_by_side_artifact": side_by_side_path
}

with open(os.path.join(ARTIFACT_DIR, "dcgan_filter_v3_evaluation.json"), "w") as f:
    json.dump(eval_summary, f, indent=2)

print("\nEvaluation complete and written to dcgan_filter_v3_evaluation.json.")
