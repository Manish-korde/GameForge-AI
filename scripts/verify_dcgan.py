import os
import sys
import json
import numpy as np
from PIL import Image

# Ensure backend can be imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "backend")))

import dcgan_service

print("=" * 80)
print("STEP 1: IDENTIFY WHAT IS ACTUALLY LOADED")
print("=" * 80)

gen = dcgan_service.load_dcgan_generator()
weights_path = dcgan_service.WEIGHTS_PATH
output_shape = gen.output_shape
param_count = gen.count_params()
latent_dim = dcgan_service.LATENT_DIM
palette_size = dcgan_service.PALETTE_SIZE

print(f"Loaded Weights Path : {weights_path}")
print(f"Weights Exist       : {os.path.exists(weights_path)}")
print(f"Weights File Size   : {os.path.getsize(weights_path):,} bytes")
print(f"Generator Name      : {gen.name}")
print(f"Input Shape         : {gen.input_shape}")
print(f"Output Layer Shape  : {output_shape}")
print(f"Total Parameters    : {param_count:,}")
print(f"Latent Dimension    : {latent_dim}")
print(f"Palette Size        : {palette_size} colors")

is_discrete = (output_shape[-1] == 33)
print(f"\nModel Classification: {'DISCRETE PALETTE MODEL (32 colors + 1 alpha logit)' if is_discrete else 'CONTINUOUS BASELINE (4-channel tanh)'}")

print("\n" + "=" * 80)
print("STEP 2: QUANTISATION TOGGLE TEST")
print("=" * 80)

# Generate with same seed, pixel_art_mode=True vs False
test_seed = 42
res_ticked = dcgan_service.generate_character_sprites(seed=test_seed, count=1, pixel_art_mode=True)
res_unticked = dcgan_service.generate_character_sprites(seed=test_seed, count=1, pixel_art_mode=False)

# Decode images to numpy
import base64
import io

img_ticked_b64 = res_ticked["sprites"][0].split(",")[1]
img_unticked_b64 = res_unticked["sprites"][0].split(",")[1]

arr_ticked = np.array(Image.open(io.BytesIO(base64.b64decode(img_ticked_b64))))
arr_unticked = np.array(Image.open(io.BytesIO(base64.b64decode(img_unticked_b64))))

diff = np.abs(arr_ticked.astype(np.int32) - arr_unticked.astype(np.int32))
max_diff = int(np.max(diff))
mean_diff = float(np.mean(diff))

print(f"Seed Tested         : {test_seed}")
print(f"Ticked Shape        : {arr_ticked.shape}, Dtype: {arr_ticked.dtype}")
print(f"Unticked Shape      : {arr_unticked.shape}, Dtype: {arr_unticked.dtype}")
print(f"Max Pixel Diff      : {max_diff}")
print(f"Mean Pixel Diff     : {mean_diff:.6f}")
print(f"Are outputs identical: {max_diff == 0}")

print("\n" + "=" * 80)
print("STEP 3: OUTPUT QUALITY CHECKS (64 FIXED SEEDS)")
print("=" * 80)

N_SEEDS = 64
reproducible_count = 0
unique_colors_list = []
alpha_valid_count = 0
fg_fractions = []
orphan_counts = []
sprite_images = []

for s in range(1, N_SEEDS + 1):
    # Test reproducibility
    run1 = dcgan_service.generate(n=1, seed=s)[0]
    run2 = dcgan_service.generate(n=1, seed=s)[0]
    
    arr1 = getattr(run1, 'native_64', np.array(run1.resize((64, 64), Image.NEAREST)))
    arr2 = getattr(run2, 'native_64', np.array(run2.resize((64, 64), Image.NEAREST)))
    
    if np.array_equal(arr1, arr2):
        reproducible_count += 1
        
    sprite_images.append(arr1)
    
    # Check alpha values (should only be 0 and 255)
    alphas = np.unique(arr1[..., 3])
    if np.all(np.isin(alphas, [0, 255])):
        alpha_valid_count += 1
        
    # Check unique colors (non-transparent)
    fg_mask = arr1[..., 3] > 0
    fg_frac = np.mean(fg_mask)
    fg_fractions.append(fg_frac)
    
    if np.any(fg_mask):
        fg_rgb = arr1[..., :3][fg_mask]
        n_unique_c = len(np.unique(fg_rgb, axis=0))
    else:
        n_unique_c = 0
    unique_colors_list.append(n_unique_c)
    
    # Check orphans / floaters
    cleaned, orphans = dcgan_service.remove_stray_floaters_tight(arr1)
    orphan_counts.append(orphans)

print(f"1. Reproducibility Rate          : {reproducible_count}/{N_SEEDS} ({reproducible_count/N_SEEDS*100:.1f}%)")
print(f"2. Alpha Binary Validity (0, 255): {alpha_valid_count}/{N_SEEDS} ({alpha_valid_count/N_SEEDS*100:.1f}%)")
print(f"3. Unique Colors per Sprite (<=32): Max={max(unique_colors_list)}, Min={min(unique_colors_list)}, Mean={np.mean(unique_colors_list):.1f}")
all_under_32 = all(c <= 32 for c in unique_colors_list)
print(f"   All <= 32 colors              : {all_under_32}")
print(f"4. Foreground Area Fraction      : Mean={np.mean(fg_fractions)*100:.1f}%, Min={min(fg_fractions)*100:.1f}%, Max={max(fg_fractions)*100:.1f}%")
print(f"5. Stray Floater Pixels Removed  : Mean={np.mean(orphan_counts):.1f}, Max={max(orphan_counts)}")

# Diversity Analysis across the 64 sprites:
# Compute pairwise L1 difference between 64 sprites
pairwise_diffs = []
for i in range(N_SEEDS):
    for j in range(i + 1, N_SEEDS):
        d = np.mean(np.abs(sprite_images[i].astype(float) - sprite_images[j].astype(float))) / 255.0
        pairwise_diffs.append(d)

mean_pairwise_div = np.mean(pairwise_diffs)
print(f"6. Pairwise L1 Diversity         : {mean_pairwise_div:.4f}")

# Character archetype breakdown (inspect palettes and features):
# Cluster sprites by dominant color and foreground silhouette
archetypes = {
    "Green/Goblin/Orc": 0,
    "Brown/Leather/Ranger": 0,
    "Gray/Plate/Knight": 0,
    "Red/Crimson/Mage": 0,
    "Blue/Navy/Wizard": 0,
    "Other/Mixed": 0
}

for arr in sprite_images:
    fg = arr[..., :3][arr[..., 3] > 0]
    if len(fg) == 0:
        archetypes["Other/Mixed"] += 1
        continue
    mean_rgb = np.mean(fg, axis=0) # [R, G, B]
    r, g, b = mean_rgb
    if g > r * 1.05 and g > b * 1.05:
        archetypes["Green/Goblin/Orc"] += 1
    elif r > 110 and g > 70 and b < 70 and r > b * 1.3:
        archetypes["Brown/Leather/Ranger"] += 1
    elif r > 130 and g < 100 and b < 100:
        archetypes["Red/Crimson/Mage"] += 1
    elif b > r * 1.1 and b > g * 1.0:
        archetypes["Blue/Navy/Wizard"] += 1
    elif abs(r - g) < 20 and abs(g - b) < 20:
        archetypes["Gray/Plate/Knight"] += 1
    else:
        archetypes["Other/Mixed"] += 1

print("\nArchetype Diversity Distribution (N = 64):")
for arch, cnt in archetypes.items():
    print(f"  {arch:22s}: {cnt:2d} sprites ({cnt/N_SEEDS*100:5.1f}%)")
