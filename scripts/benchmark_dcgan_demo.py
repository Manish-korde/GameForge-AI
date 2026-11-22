import os
import sys
import json
import numpy as np
from PIL import Image
from scipy.ndimage import label, binary_dilation

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "backend")))
import dcgan_service
import app

# Setup directories
REAL_OUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "backend", "alucard_samples", "real_frontfacing"))
os.makedirs(REAL_OUT_DIR, exist_ok=True)

print("=" * 80)
print("PART 1: EXTRACT 8 REAL FRONT-FACING TRAINING SPRITES")
print("=" * 80)

indices_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "config", "alucard_frontfacing_42k_indices.json"))
with open(indices_path, "r") as f:
    frontfacing_indices = json.load(f)

print(f"Total front-facing indices available: {len(frontfacing_indices)}")

# Select 8 diverse real front-facing indices
# Use deterministic selection spread across dataset
selected_indices = [
    frontfacing_indices[10],   # Knight
    frontfacing_indices[55],   # Rogue/leather
    frontfacing_indices[120],  # Wizard/mage
    frontfacing_indices[250],  # Adventurer
    frontfacing_indices[500],  # Guard
    frontfacing_indices[1000], # Cleric
    frontfacing_indices[2500], # Fighter
    frontfacing_indices[5000], # Ranger
]

raw_ds = app.get_raw_dataset()
real_sprites = []
for i, idx in enumerate(selected_indices):
    item = raw_ds[int(idx)]
    img = item["image"].convert("RGBA")
    # Resize to 64x64 if not already
    if img.size != (64, 64):
        img = img.resize((64, 64), Image.NEAREST)
    out_path = os.path.join(REAL_OUT_DIR, f"real_{i}.png")
    img.save(out_path)
    real_sprites.append(img)
    print(f"  Extracted Real Sprite #{i+1} (Dataset Row {idx}) -> {out_path}")

print("\n" + "=" * 80)
print("PART 2: QUALITY FILTER V2 BENCHMARK ON 10 SEEDS (FILTERED VS RAW)")
print("=" * 80)

SEEDS = [42, 107, 256, 314, 512, 777, 888, 1337, 2024, 4242]

gen = dcgan_service.load_dcgan_generator()

def clean_and_measure_v2(rgba_img, touch_dilation=2):
    """
    Quality Filter v2:
    - Main body = largest connected component
    - Kept pieces = main body + any component touching main body within 2 px (dilation=2)
    - All other components are detached floaters
    Returns: cleaned_img, floater_count, stray_pixels
    """
    alpha = rgba_img[..., 3] > 127
    if not np.any(alpha):
        return rgba_img, 0, 0
    labeled_array, num_features = label(alpha)
    if num_features <= 1:
        return rgba_img, 0, 0
        
    sizes = np.bincount(labeled_array.ravel())
    largest_label = np.argmax(sizes[1:]) + 1
    body_mask = (labeled_array == largest_label)
    
    dilated_body = binary_dilation(body_mask, iterations=touch_dilation)
    
    keep_mask = body_mask.copy()
    floater_count = 0
    stray_pixels = 0
    
    for i in range(1, num_features + 1):
        if i == largest_label:
            continue
        comp_mask = (labeled_array == i)
        if np.any(comp_mask & dilated_body):
            keep_mask |= comp_mask
        else:
            floater_count += 1
            stray_pixels += int(np.sum(comp_mask))
            
    cleaned = rgba_img.copy()
    cleaned[~keep_mask] = 0
    return cleaned, floater_count, stray_pixels

raw_floaters_per_seed = []
raw_strays_per_seed = []
filtered_floaters_per_seed = []
filtered_strays_per_seed = []

print(f"{'Seed':>6} | {'Raw Floaters':>12} | {'Raw Strays':>12} | {'Filt Floaters':>14} | {'Filt Strays':>12}")
print("-" * 65)

for s in SEEDS:
    # 1. Generate 16 RAW sprites directly from anchor latents
    rng = np.random.RandomState(s)
    raw_latents = rng.normal(0, 1.0, size=(16, dcgan_service.LATENT_DIM)).astype(np.float32)
    raw_latents = np.clip(raw_latents, -2.0, 2.0)
    raw_logits = gen(raw_latents, training=False)
    raw_imgs = dcgan_service.logits_to_image(raw_logits, dcgan_service._palette_tf, hard=True, noise=False)
    
    s_raw_floaters = []
    s_raw_strays = []
    for im in raw_imgs:
        _, fl, st = clean_and_measure_v2(im)
        s_raw_floaters.append(fl)
        s_raw_strays.append(st)
        
    mean_raw_fl = float(np.mean(s_raw_floaters))
    mean_raw_st = float(np.mean(s_raw_strays))
    raw_floaters_per_seed.append(mean_raw_fl)
    raw_strays_per_seed.append(mean_raw_st)
    
    # 2. Quality Filter v2: Generate 32 candidates, clean, reject, rank, pick best 16
    rng_filt = np.random.RandomState(s)
    cand_latents = rng_filt.normal(0, 1.0, size=(32, dcgan_service.LATENT_DIM)).astype(np.float32)
    cand_latents = np.clip(cand_latents, -2.0, 2.0)
    cand_logits = gen(cand_latents, training=False)
    cand_imgs = dcgan_service.logits_to_image(cand_logits, dcgan_service._palette_tf, hard=True, noise=False)
    
    scored_candidates = []
    for c_img in cand_imgs:
        cleaned, fl, st = clean_and_measure_v2(c_img)
        # Verify no leftover floaters
        _, left_fl, left_st = clean_and_measure_v2(cleaned)
        if left_fl > 0 or left_st > 0:
            continue # Hard reject any candidate with leftover floaters
        score = dcgan_service.score_candidate(cleaned, fl)
        scored_candidates.append((score, cleaned))
        
    scored_candidates.sort(key=lambda x: x[0])
    best_16 = [x[1] for x in scored_candidates[:16]]
    
    s_filt_floaters = []
    s_filt_strays = []
    for im in best_16:
        _, fl, st = clean_and_measure_v2(im)
        s_filt_floaters.append(fl)
        s_filt_strays.append(st)
        
    mean_filt_fl = float(np.mean(s_filt_floaters))
    mean_filt_st = float(np.mean(s_filt_strays))
    filtered_floaters_per_seed.append(mean_filt_fl)
    filtered_strays_per_seed.append(mean_filt_st)
    
    print(f"{s:6d} | {mean_raw_fl:12.2f} | {mean_raw_st:12.1f} | {mean_filt_fl:14.2f} | {mean_filt_st:12.1f}")

print("-" * 65)
print(f"{'OVERALL MEAN':>6} | {np.mean(raw_floaters_per_seed):12.2f} | {np.mean(raw_strays_per_seed):12.1f} | {np.mean(filtered_floaters_per_seed):14.2f} | {np.mean(filtered_strays_per_seed):12.1f}")

print("\n" + "=" * 80)
print("PART 3: SPATIAL COHERENCE OF SERVED SAMPLES (1000 RAW SAMPLES)")
print("=" * 80)

def compute_spatial_coherence(imgs_u8):
    """
    Computes exact paper spatial color coherence:
    fraction of adjacent horizontal foreground pixels sharing the exact same color.
    """
    coherences = []
    for im in imgs_u8:
        fg = im[..., 3] > 127
        if not np.any(fg):
            coherences.append(0.0)
            continue
        same_h = (im[:, 1:, :3] == im[:, :-1, :3]).all(-1)
        valid_adj = fg[:, 1:] & fg[:, :-1]
        if np.any(valid_adj):
            coherences.append(float(same_h[valid_adj].mean()))
        else:
            coherences.append(0.0)
    return float(np.mean(coherences)), float(np.std(coherences))

# Generate 1000 raw samples in batches of 100
N_SAMPLES = 1000
BATCH_SZ = 100
all_raw_coherences = []

print(f"Generating {N_SAMPLES} raw DCGAN samples...")
rng_1000 = np.random.RandomState(42)

for b in range(N_SAMPLES // BATCH_SZ):
    b_latents = rng_1000.normal(0, 1.0, size=(BATCH_SZ, dcgan_service.LATENT_DIM)).astype(np.float32)
    b_latents = np.clip(b_latents, -2.0, 2.0)
    b_logits = gen(b_latents, training=False)
    b_imgs = dcgan_service.logits_to_image(b_logits, dcgan_service._palette_tf, hard=True, noise=False)
    m_coh, _ = compute_spatial_coherence(b_imgs)
    all_raw_coherences.append(m_coh)

raw_mean_coh = float(np.mean(all_raw_coherences))
print(f"1000 Raw Generated DCGAN Samples Spatial Coherence: {raw_mean_coh * 100:.2f}%")

# Compute on 1000 Real Training Sprites from alucard_frontfacing_42k_indices.json
print(f"\nComputing spatial coherence on 1000 Real Front-Facing Training Sprites...")
real_sample_indices = np.random.RandomState(42).choice(len(frontfacing_indices), size=1000, replace=False)
real_sample_imgs = []
for idx in real_sample_indices:
    raw_idx = int(frontfacing_indices[idx])
    im = raw_ds[raw_idx]["image"].convert("RGBA")
    if im.size != (64, 64):
        im = im.resize((64, 64), Image.NEAREST)
    real_sample_imgs.append(np.array(im))

real_mean_coh, real_std_coh = compute_spatial_coherence(np.array(real_sample_imgs))
print(f"1000 Real Training Sprites Spatial Coherence       : {real_mean_coh * 100:.2f}% (std: {real_std_coh * 100:.2f}%)")

# Output JSON summary for artifact
benchmark_results = {
    "raw_floaters_per_sprite": float(np.mean(raw_floaters_per_seed)),
    "raw_strays_per_sprite": float(np.mean(raw_strays_per_seed)),
    "filtered_floaters_per_sprite": float(np.mean(filtered_floaters_per_seed)),
    "filtered_strays_per_sprite": float(np.mean(filtered_strays_per_seed)),
    "raw_1000_spatial_coherence_percent": raw_mean_coh * 100,
    "real_1000_spatial_coherence_percent": real_mean_coh * 100
}

with open(r"C:\Users\manis\.gemini\antigravity\brain\674ad8cd-af16-4b87-91e3-66973cf97bc5\dcgan_benchmark_results.json", "w") as f:
    json.dump(benchmark_results, f, indent=2)

print("\nBenchmark saved to dcgan_benchmark_results.json successfully.")
