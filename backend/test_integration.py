import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dcgan_service import (
    load_dcgan_generator,
    load_cdcgan_generator,
    generate_character_sprites,
    get_status,
    get_cdcgan_status,
    CONDITIONAL_CLASSES,
    CONDITIONAL_CLASS_INFO
)

print("Testing DCGAN and cDCGAN v3 model loading...")
load_dcgan_generator()
load_cdcgan_generator()

print(f"DCGAN Status: {get_status()}")
print(f"cDCGAN Status: {get_cdcgan_status()}")

# 1. Test unconditional generation
print("\n--- Testing Unconditional Generation ---")
res_uncond = generate_character_sprites(
    count=4,
    seed=1337,
    generator_mode="unconditional",
    quality_filter=True
)
print(f"Unconditional: model={res_uncond['model']}, count={res_uncond['count']}, filter_stats={res_uncond['filter_stats']}")
assert len(res_uncond["sprites"]) == 4, f"Expected 4 sprites, got {len(res_uncond['sprites'])}"
assert res_uncond["generator_mode"] == "unconditional"

# 2. Test conditional generation for each of the 4 classes
print("\n--- Testing Conditional Palette (cDCGAN v3) Generation ---")
results = {}
for cls_info in CONDITIONAL_CLASS_INFO:
    cid = cls_info["id"]
    cname = cls_info["name"]
    res_cond = generate_character_sprites(
        count=4,
        seed=1337,
        generator_mode="conditional_v3",
        class_id=cid,
        quality_filter=True
    )
    results[cid] = res_cond
    print(f"Class {cid} ({cname}): model={res_cond['model']}, filter_stats={res_cond['filter_stats']}")
    assert len(res_cond["sprites"]) == 4, f"Expected 4 sprites, got {len(res_cond['sprites'])}"
    assert res_cond["generator_mode"] == "conditional_v3"
    assert res_cond["class_id"] == cid

print("\n--- Testing Palette Discrimination Across Classes ---")
import base64
import io
from PIL import Image

def get_mean_rgb(b64_str):
    header, data = b64_str.split(",", 1)
    img = Image.open(io.BytesIO(base64.b64decode(data))).convert("RGBA")
    arr = np.array(img)
    fg = arr[..., 3] > 127
    if np.sum(fg) == 0:
        return np.array([0, 0, 0], dtype=float)
    return np.mean(arr[fg][:, :3], axis=0)

mean_rgbs = {}
for cid in range(4):
    rgbs = [get_mean_rgb(s) for s in results[cid]["sprites"]]
    mean_rgbs[cid] = np.mean(rgbs, axis=0)
    print(f"Class {cid} ({CONDITIONAL_CLASS_INFO[cid]['name']}) Mean FG RGB: {np.round(mean_rgbs[cid], 1)}")

# Check that RGBs are distinct across all class pairs
for i in range(4):
    for j in range(i + 1, 4):
        dist = np.linalg.norm(mean_rgbs[i] - mean_rgbs[j])
        print(f"RGB Distance between Class {i} and Class {j}: {dist:.2f}")
        assert dist > 8.0, f"Classes {i} and {j} did not show distinct palettes: dist={dist}"

print("\nAll integration checks PASSED successfully!")

# Assemble visual smoke test grid (5 rows x 4 cols: Uncond, Knight, Orc, Ranger, Mage)
artifact_dir = r"C:\Users\manis\.gemini\antigravity\brain\674ad8cd-af16-4b87-91e3-66973cf97bc5"
grid_path = os.path.join(artifact_dir, "dcgan_studio_integration_smoke_test_grid.png")

grid_img = Image.new("RGBA", (4 * 128, 5 * 128), (20, 24, 36, 255))
rows = [("Unconditional", res_uncond["sprites"])] + [
    (CONDITIONAL_CLASS_INFO[cid]["name"], results[cid]["sprites"]) for cid in range(4)
]

for row_idx, (label, sprite_b64s) in enumerate(rows):
    for col_idx, b64_str in enumerate(sprite_b64s):
        header, data = b64_str.split(",", 1)
        sp = Image.open(io.BytesIO(base64.b64decode(data))).convert("RGBA").resize((128, 128), Image.NEAREST)
        grid_img.paste(sp, (col_idx * 128, row_idx * 128), sp)

grid_img.save(grid_path)
print(f"Saved visual smoke test grid to: {grid_path}")

