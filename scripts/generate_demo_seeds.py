import os
import sys
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "backend")))
import dcgan_service

CANDIDATE_SEEDS = [42, 107, 256, 314, 512, 777, 888, 1337, 2024, 4242]
ARTIFACT_DIR = r"C:\Users\manis\.gemini\antigravity\brain\674ad8cd-af16-4b87-91e3-66973cf97bc5"

print("=" * 80)
print("TESTING 10 CANDIDATE SEEDS (16 SPRITES EACH, 4x4 GRID)")
print("=" * 80)

gen = dcgan_service.load_dcgan_generator()

results = []

for s in CANDIDATE_SEEDS:
    # Generate 16 sprites with quality_filter=False (the default raw generator)
    imgs = dcgan_service.generate(n=16, seed=s, quality_filter=False)
    
    fg_fracs = []
    unique_colors = []
    orphans_list = []
    native_sprites = []
    
    for img in imgs:
        arr = getattr(img, 'native_64', np.array(img.resize((64, 64), Image.NEAREST)))
        native_sprites.append(arr)
        
        fg_mask = arr[..., 3] > 127
        fg_frac = np.mean(fg_mask)
        fg_fracs.append(fg_frac)
        
        if np.any(fg_mask):
            rgb = arr[..., :3][fg_mask]
            uc = len(np.unique(rgb, axis=0))
        else:
            uc = 0
        unique_colors.append(uc)
        
        _, orp = dcgan_service.remove_stray_floaters_tight(arr)
        orphans_list.append(orp)
        
    mean_fg = float(np.mean(fg_fracs))
    min_fg = float(np.min(fg_fracs))
    mean_uc = float(np.mean(unique_colors))
    mean_orp = float(np.mean(orphans_list))
    
    # Compose 4x4 grid (upscaled nearest-neighbor to 128x128 per sprite -> 512x512 total grid)
    GRID_ROWS, GRID_COLS = 4, 4
    SPRITE_SZ = 128
    grid_img = Image.new("RGBA", (GRID_COLS * SPRITE_SZ, GRID_ROWS * SPRITE_SZ), (15, 23, 42, 255))
    
    for idx, arr in enumerate(native_sprites):
        r = idx // GRID_COLS
        c = idx % GRID_COLS
        pil_spr = Image.fromarray(arr, mode="RGBA").resize((SPRITE_SZ, SPRITE_SZ), Image.NEAREST)
        grid_img.paste(pil_spr, (c * SPRITE_SZ, r * SPRITE_SZ), pil_spr)
        
    save_path = os.path.join(ARTIFACT_DIR, f"dcgan_grid_seed_{s}.png")
    grid_img.save(save_path)
    
    # Composite score: reward ~32% foreground, penalize floaters, reward good color count (18-24)
    # Lower score is better
    score = abs(mean_fg - 0.32) * 100.0 + (mean_orp * 2.0) + abs(mean_uc - 21.0)
    
    res = {
        "seed": s,
        "mean_fg": mean_fg,
        "min_fg": min_fg,
        "mean_uc": mean_uc,
        "mean_orp": mean_orp,
        "score": score,
        "file": save_path
    }
    results.append(res)
    print(f"Seed {s:5d}: Mean FG = {mean_fg*100:4.1f}% | Min FG = {min_fg*100:4.1f}% | Colors = {mean_uc:4.1f} | Orphans = {mean_orp:3.1f} | Score = {score:5.2f} -> Saved to {save_path}")

# Sort by quality score
results.sort(key=lambda x: x["score"])

print("\n" + "=" * 80)
print("RANKED SEEDS (TOP 3 RECOMMENDED FOR DEMO)")
print("=" * 80)
for rank, r in enumerate(results, 1):
    tag = "★ TOP 3 DEMO-SAFE" if rank <= 3 else ""
    print(f"Rank {rank:2d}: Seed {r['seed']:5d} (Score: {r['score']:5.2f}, Mean FG: {r['mean_fg']*100:.1f}%, Colors: {r['mean_uc']:.1f}) {tag}")

# Verify Roll Seed retry logic
print("\n" + "=" * 80)
print("VERIFYING ROLL SEED RETRY LOGIC (Foreground >= 10%)")
print("=" * 80)
safe_seed = dcgan_service.find_safe_seed(seed=None, max_retries=5)
probe_imgs = dcgan_service.generate(n=1, seed=safe_seed, quality_filter=False)
probe_arr = getattr(probe_imgs[0], 'native_64', np.array(probe_imgs[0].resize((64, 64), Image.NEAREST)))
probe_fg = np.mean(probe_arr[..., 3] > 127)
print(f"Sample Rolled Safe Seed: {safe_seed} -> Foreground Area Fraction: {probe_fg*100:.1f}% (Pass >= 10%: {probe_fg >= 0.10})")
