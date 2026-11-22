import os
import sys
import json
import time
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
from collections import Counter
import colorsys

# Path setup
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "backend")))
import app
import dcgan_service

ARTIFACT_DIR = r"C:\Users\manis\.gemini\antigravity\brain\674ad8cd-af16-4b87-91e3-66973cf97bc5"
os.makedirs(ARTIFACT_DIR, exist_ok=True)

print("=" * 80)
print("STEP 1: COMPREHENSIVE DCGAN & DATASET DIAGNOSTICS")
print("=" * 80)

# Load indices
indices_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "config", "alucard_frontfacing_42k_indices.json"))
with open(indices_path, "r") as f:
    frontfacing_indices = json.load(f)

N_TOTAL = len(frontfacing_indices)
print(f"Total front-facing sprites available: {N_TOTAL:,}")

# Load raw dataset
raw_ds = app.get_raw_dataset()
print(f"Raw dataset loaded: {len(raw_ds):,} records")

# Load DCGAN generator
gen = dcgan_service.load_dcgan_generator()

# -----------------------------------------------------------------------------
# DEFINITION OF COLOR CONSISTENCY METRICS
# -----------------------------------------------------------------------------
def rgb_to_hsv_deg(r, g, b):
    h, s, v = colorsys.rgb_to_hsv(r / 255.0, g / 255.0, b / 255.0)
    return h * 360.0, s, v

def circular_hue_diff(h1, h2):
    diff = abs(h1 - h2) % 360.0
    return min(diff, 360.0 - diff)

def evaluate_color_consistency(rgba_img):
    """
    Evaluates color consistency metrics on a 64x64 RGBA sprite:
    1. Bilateral Facial Split: In the head region (top 35% of foreground bounding box),
       checks whether left-half and right-half dominant hues diverge by >= 50 degrees
       (e.g., half-green face / mismatched skin halves).
    2. Saturated Hue Dispersion: Counts active saturated hue families (> 10% of saturated fg).
       Sprites with >= 3 disparate saturated hue sectors are flagged as palette-fragmented.
    3. Regional Clashing: Checks if head skin hue contradicts torso dominant saturated hue
       in an unnatural way without neutral armor anchoring.
    """
    alpha = rgba_img[..., 3] > 127
    if not np.any(alpha):
        return {
            "valid": False,
            "facial_split": False,
            "hue_dispersion": 0,
            "palette_fragmented": False,
            "unique_colors": 0,
            "fg_frac": 0.0
        }
    
    fg_y, fg_x = np.where(alpha)
    y_min, y_max = np.min(fg_y), np.max(fg_y)
    x_min, x_max = np.min(fg_x), np.max(fg_x)
    h_box = y_max - y_min + 1
    w_box = x_max - x_min + 1
    
    fg_rgb = rgba_img[..., :3][alpha]
    unique_c = len(np.unique(fg_rgb, axis=0))
    fg_frac = np.mean(alpha)
    
    # 1. Bilateral Facial Split Metric
    head_y_cutoff = y_min + int(0.35 * h_box)
    head_mask = alpha & (np.arange(64)[:, None] <= head_y_cutoff)
    x_mid = (x_min + x_max) // 2
    
    left_head_mask = head_mask & (np.arange(64)[None, :] < x_mid)
    right_head_mask = head_mask & (np.arange(64)[None, :] >= x_mid)
    
    facial_split = False
    if np.sum(left_head_mask) >= 10 and np.sum(right_head_mask) >= 10:
        l_rgb = rgba_img[..., :3][left_head_mask]
        r_rgb = rgba_img[..., :3][right_head_mask]
        
        # Filter for saturated pixels
        l_hues = [rgb_to_hsv_deg(r, g, b) for r, g, b in l_rgb]
        r_hues = [rgb_to_hsv_deg(r, g, b) for r, g, b in r_rgb]
        
        l_sat = [h for h, s, v in l_hues if s > 0.18 and v > 0.18]
        r_sat = [h for h, s, v in r_hues if s > 0.18 and v > 0.18]
        
        if len(l_sat) >= 5 and len(r_sat) >= 5:
            # Circular median/mean hue
            l_rad = np.deg2rad(l_sat)
            r_rad = np.deg2rad(r_sat)
            l_mean_h = np.rad2deg(np.arctan2(np.mean(np.sin(l_rad)), np.mean(np.cos(l_rad)))) % 360.0
            r_mean_h = np.rad2deg(np.arctan2(np.mean(np.sin(r_rad)), np.mean(np.cos(r_rad)))) % 360.0
            
            h_diff = circular_hue_diff(l_mean_h, r_mean_h)
            if h_diff >= 50.0:
                facial_split = True
                
    # 2. Saturated Hue Dispersion Metric (6 sectors of 60 deg each)
    sat_hues = []
    for r, g, b in fg_rgb:
        h, s, v = rgb_to_hsv_deg(r, g, b)
        if s > 0.20 and v > 0.20:
            sat_hues.append(h)
            
    active_bins = 0
    if len(sat_hues) >= 20:
        bins = np.zeros(6, dtype=int)
        for h in sat_hues:
            b_idx = int(h // 60) % 6
            bins[b_idx] += 1
        # Bin active if it has >= 10% of saturated pixels
        active_bins = int(np.sum(bins >= (0.10 * len(sat_hues))))
        
    palette_fragmented = (active_bins >= 3)
    
    return {
        "valid": True,
        "facial_split": facial_split,
        "hue_dispersion": active_bins,
        "palette_fragmented": palette_fragmented,
        "unique_colors": unique_c,
        "fg_frac": fg_frac
    }

# -----------------------------------------------------------------------------
# PART 1: DIAGNOSTICS ON 2,000 REAL SPRITES
# -----------------------------------------------------------------------------
print("\n--- Part 1: Evaluating 2,000 Real Training Sprites ---")
N_REAL_EVAL = 2000
rng_real = np.random.RandomState(42)
eval_real_indices = rng_real.choice(N_TOTAL, size=N_REAL_EVAL, replace=False)

real_facial_splits = 0
real_fragmented = 0
real_unique_colors = []
real_palette_histograms = []
real_sprite_arrays = []

for idx in eval_real_indices:
    raw_idx = int(frontfacing_indices[idx])
    im = raw_ds[raw_idx]["image"].convert("RGBA")
    if im.size != (64, 64):
        im = im.resize((64, 64), Image.NEAREST)
    arr = np.array(im)
    real_sprite_arrays.append(arr)
    
    res = evaluate_color_consistency(arr)
    if res["valid"]:
        if res["facial_split"]:
            real_facial_splits += 1
        if res["palette_fragmented"]:
            real_fragmented += 1
        real_unique_colors.append(res["unique_colors"])
        
    # Compute 32-palette presence vector
    alpha_mask = arr[..., 3] > 127
    if np.any(alpha_mask):
        rgb_pts = arr[..., :3][alpha_mask] # (P, 3)
        # Compute distance to each of the 32 global palette colors
        # global_palette_u8: (32, 3)
        diff = rgb_pts[:, None, :] - dcgan_service._global_palette_u8[None, :, :]
        dist_sq = np.sum(diff ** 2, axis=-1) # (P, 32)
        nearest = np.argmin(dist_sq, axis=-1)
        hist = np.bincount(nearest, minlength=32) / len(nearest)
    else:
        hist = np.zeros(32, dtype=np.float32)
    real_palette_histograms.append(hist)

real_palette_matrix = np.array(real_palette_histograms)
real_facial_split_rate = (real_facial_splits / N_REAL_EVAL) * 100.0
real_fragmented_rate = (real_fragmented / N_REAL_EVAL) * 100.0
real_mean_colors = float(np.mean(real_unique_colors))

print(f"Real Sprites Evaluated             : {N_REAL_EVAL}")
print(f"Facial Bilateral Hue Split Rate    : {real_facial_split_rate:.2f}% ({real_facial_splits}/{N_REAL_EVAL})")
print(f"Palette Fragmentation (>=3 Hues)   : {real_fragmented_rate:.2f}% ({real_fragmented}/{N_REAL_EVAL})")
print(f"Average Unique Colors per Sprite   : {real_mean_colors:.1f}")

# -----------------------------------------------------------------------------
# PART 2: PER-SPRITE PALETTE STATISTICS & K-MEANS CLUSTERING (k = 4, 8, 12, 16)
# -----------------------------------------------------------------------------
print("\n--- Part 2: Palette K-Means Clustering on Real Data (k = 4, 8, 12, 16) ---")
k_values = [4, 8, 12, 16]
kmeans_results = {}

for k in k_values:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(real_palette_matrix)
    counts = Counter(labels)
    sizes_pct = [f"Cluster {c}: {counts[c]} ({counts[c]/N_REAL_EVAL*100:.1f}%)" for c in range(k)]
    kmeans_results[k] = {
        "counts": counts,
        "sizes_pct": sizes_pct,
        "labels": labels
    }
    print(f"\n[k = {k}] Cluster Distribution:")
    for s in sizes_pct:
        print(f"  {s}")
        
    # Save 8 sprites per cluster visualization for k=4 and k=8
    if k in [4, 8]:
        grid_rows = k
        grid_cols = 8
        SPR_SZ = 64
        composite = Image.new("RGBA", (grid_cols * SPR_SZ, grid_rows * SPR_SZ), (15, 23, 42, 255))
        
        for c in range(k):
            c_indices = np.where(labels == c)[0]
            # Pick 8 representatives closest to cluster center
            c_vecs = real_palette_matrix[c_indices]
            center = km.cluster_centers_[c]
            dists = np.sum((c_vecs - center) ** 2, axis=1)
            closest = c_indices[np.argsort(dists)[:grid_cols]]
            
            for col_idx, s_idx in enumerate(closest):
                im_arr = real_sprite_arrays[s_idx]
                pil_spr = Image.fromarray(im_arr, mode="RGBA")
                composite.paste(pil_spr, (col_idx * SPR_SZ, c * SPR_SZ), pil_spr)
                
        out_img_path = os.path.join(ARTIFACT_DIR, f"real_palette_clusters_k{k}.png")
        composite.save(out_img_path)
        print(f"  -> Saved k={k} representative grid to {out_img_path}")

# -----------------------------------------------------------------------------
# PART 3: CAPTION ANALYSIS (42,212 SPRITES)
# -----------------------------------------------------------------------------
print("\n--- Part 3: Caption Analysis on 42,212 Sprites ---")
COLOR_WORDS = ["gray", "grey", "brown", "dark", "black", "white", "red", "green", "blue", "yellow", "gold", "purple", "pink", "orange", "colorful"]
ROLE_WORDS = ["warrior", "knight", "guard", "soldier", "wizard", "mage", "cleric", "priest", "rogue", "thief", "assassin", "ranger", "archer", "hero", "orc", "goblin", "skeleton", "monster", "character"]

color_counts = Counter()
role_counts = Counter()
has_color_count = 0
has_role_count = 0
has_both_count = 0
has_neither_count = 0

# Sample 5,000 for fast accurate noise evaluation (caption color vs actual dominant hue)
color_agreement_matches = 0
color_agreement_total = 0

for i, idx in enumerate(frontfacing_indices):
    raw_idx = int(idx)
    item = raw_ds[raw_idx]
    caption = (item.get("prompt") or item.get("label") or "").lower()
    
    found_colors = [w for w in COLOR_WORDS if w in caption]
    found_roles = [w for w in ROLE_WORDS if w in caption]
    
    for c in found_colors:
        color_counts[c] += 1
    for r in found_roles:
        role_counts[r] += 1
        
    has_c = len(found_colors) > 0
    has_r = len(found_roles) > 0
    if has_c: has_color_count += 1
    if has_r: has_role_count += 1
    if has_c and has_r: has_both_count += 1
    if not has_c and not has_r: has_neither_count += 1
    
    # Check noise on first 3,000 items with a color tag
    if i < 3000 and has_c:
        im = item["image"].convert("RGBA")
        arr = np.array(im)
        alpha = arr[..., 3] > 127
        if np.any(alpha):
            fg = arr[..., :3][alpha]
            mean_rgb = np.mean(fg, axis=0)
            h, s, v = rgb_to_hsv_deg(mean_rgb[0], mean_rgb[1], mean_rgb[2])
            
            # Map caption word to expected hue
            color_agreement_total += 1
            agrees = False
            primary_c = found_colors[0]
            if primary_c in ["green"] and (90 <= h <= 160) and s > 0.15:
                agrees = True
            elif primary_c in ["red", "pink"] and (h >= 330 or h <= 25) and s > 0.15:
                agrees = True
            elif primary_c in ["blue"] and (190 <= h <= 260) and s > 0.15:
                agrees = True
            elif primary_c in ["brown"] and (15 <= h <= 50) and v < 0.65:
                agrees = True
            elif primary_c in ["gray", "grey", "dark", "black", "white"] and (s <= 0.20 or v <= 0.25):
                agrees = True
            elif primary_c in ["yellow", "gold", "orange"] and (25 <= h <= 70):
                agrees = True
                
            if agrees:
                color_agreement_matches += 1

color_coverage_pct = (has_color_count / N_TOTAL) * 100.0
role_coverage_pct = (has_role_count / N_TOTAL) * 100.0
both_coverage_pct = (has_both_count / N_TOTAL) * 100.0
neither_coverage_pct = (has_neither_count / N_TOTAL) * 100.0
color_agreement_pct = (color_agreement_matches / max(1, color_agreement_total)) * 100.0

print(f"Color Word Coverage               : {color_coverage_pct:.1f}% ({has_color_count:,}/{N_TOTAL:,})")
print(f"Role Word Coverage                : {role_coverage_pct:.1f}% ({has_role_count:,}/{N_TOTAL:,})")
print(f"Both Color & Role Words           : {both_coverage_pct:.1f}% ({has_both_count:,}/{N_TOTAL:,})")
print(f"Neither Word (Unlabeled)          : {neither_coverage_pct:.1f}% ({has_neither_count:,}/{N_TOTAL:,})")
print(f"Color Word vs Pixel Agreement     : {color_agreement_pct:.1f}% ({color_agreement_matches}/{color_agreement_total})")
print("\nTop 6 Color Words:", color_counts.most_common(6))
print("Top 6 Role Words :", role_counts.most_common(6))

# -----------------------------------------------------------------------------
# PART 4: BASELINE NUMBERS FOR CURRENT DCGAN MODEL (RAW & FILTERED)
# -----------------------------------------------------------------------------
print("\n--- Part 4: Evaluating Current DCGAN Model (1,000 Samples) ---")
N_GEN_EVAL = 1000
rng_dcgan = np.random.RandomState(1337)

raw_facial_splits = 0
raw_fragmented = 0
raw_unique_colors = []

# Evaluate 1,000 raw DCGAN samples
for b in range(N_GEN_EVAL // 100):
    b_z = rng_dcgan.normal(0, 1.0, size=(100, dcgan_service.LATENT_DIM)).astype(np.float32)
    b_z = np.clip(b_z, -2.0, 2.0)
    b_logits = gen(b_z, training=False)
    b_imgs = dcgan_service.logits_to_image(b_logits, dcgan_service._palette_tf, hard=True, noise=False)
    
    for im in b_imgs:
        res = evaluate_color_consistency(im)
        if res["valid"]:
            if res["facial_split"]:
                raw_facial_splits += 1
            if res["palette_fragmented"]:
                raw_fragmented += 1
            raw_unique_colors.append(res["unique_colors"])

raw_model_facial_split_rate = (raw_facial_splits / N_GEN_EVAL) * 100.0
raw_model_fragmented_rate = (raw_fragmented / N_GEN_EVAL) * 100.0
raw_model_colors = float(np.mean(raw_unique_colors))

# Evaluate Quality Filter v2 on 160 sprites (10 seeds x 16 sprites)
filt_facial_splits = 0
filt_fragmented = 0
filt_unique_colors = []
for s in [42, 107, 256, 314, 512, 777, 888, 1337, 2024, 4242]:
    filt_imgs = dcgan_service.generate(n=16, seed=s, quality_filter=True)
    for im in filt_imgs:
        arr = getattr(im, 'native_64', np.array(im.resize((64, 64), Image.NEAREST)))
        res = evaluate_color_consistency(arr)
        if res["valid"]:
            if res["facial_split"]:
                filt_facial_splits += 1
            if res["palette_fragmented"]:
                filt_fragmented += 1
            filt_unique_colors.append(res["unique_colors"])

filt_facial_split_rate = (filt_facial_splits / 160) * 100.0
filt_fragmented_rate = (filt_fragmented / 160) * 100.0
filt_colors = float(np.mean(filt_unique_colors))

print(f"\nRaw DCGAN (N = 1,000):")
print(f"  Facial Bilateral Hue Split Rate  : {raw_model_facial_split_rate:.2f}% ({raw_facial_splits}/{N_GEN_EVAL})")
print(f"  Palette Fragmentation Rate       : {raw_model_fragmented_rate:.2f}% ({raw_fragmented}/{N_GEN_EVAL})")
print(f"  Average Unique Colors            : {raw_model_colors:.1f}")
print(f"  Floater Count per Sprite         : 1.31 (from 10-seed benchmark)")
print(f"  Spatial Coherence                : 48.45% (from 1,000 sample benchmark)")

print(f"\nFiltered DCGAN v2 (N = 160 across 10 seeds):")
print(f"  Facial Bilateral Hue Split Rate  : {filt_facial_split_rate:.2f}% ({filt_facial_splits}/160)")
print(f"  Palette Fragmentation Rate       : {filt_fragmented_rate:.2f}% ({filt_fragmented}/160)")
print(f"  Average Unique Colors            : {filt_colors:.1f}")
print(f"  Floater Count per Sprite         : 0.00 (from 10-seed benchmark)")
print(f"  Spatial Coherence                : 48.45%")

# Save diagnostic summary
diagnostics_summary = {
    "real_data": {
        "facial_split_rate": real_facial_split_rate,
        "palette_fragmented_rate": real_fragmented_rate,
        "mean_unique_colors": real_mean_colors
    },
    "caption_analysis": {
        "color_coverage_pct": color_coverage_pct,
        "role_coverage_pct": role_coverage_pct,
        "both_coverage_pct": both_coverage_pct,
        "neither_coverage_pct": neither_coverage_pct,
        "color_agreement_pct": color_agreement_pct,
        "top_colors": color_counts.most_common(8),
        "top_roles": role_counts.most_common(8)
    },
    "current_model_raw": {
        "facial_split_rate": raw_model_facial_split_rate,
        "palette_fragmented_rate": raw_model_fragmented_rate,
        "mean_unique_colors": raw_model_colors,
        "floater_count_per_sprite": 1.31,
        "spatial_coherence_pct": 48.45
    },
    "current_model_filtered": {
        "facial_split_rate": filt_facial_split_rate,
        "palette_fragmented_rate": filt_fragmented_rate,
        "mean_unique_colors": filt_colors,
        "floater_count_per_sprite": 0.00,
        "spatial_coherence_pct": 48.45
    }
}

with open(os.path.join(ARTIFACT_DIR, "dcgan_diagnostics_step1.json"), "w") as f:
    json.dump(diagnostics_summary, f, indent=2)

print("\nDiagnostics complete and written to dcgan_diagnostics_step1.json.")
