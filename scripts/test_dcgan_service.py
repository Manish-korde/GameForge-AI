import os
import sys
import io
import base64
import numpy as np
from PIL import Image

# Add backend to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "backend")))
from dcgan_service import load_dcgan_generator, generate_character_sprites, get_status, CLASS_PRESETS

print("--- TESTING DCGAN SERVICE INTEGRATION ---")
gen = load_dcgan_generator()
print("Status after load:", get_status())
assert gen is not None, "Failed to load generator"

# Test generation
result = generate_character_sprites(
    role="Archer (Ranger)",
    count=4,
    seed=42,
    pixel_art_mode=True,
    upscale_factor=8
)

print("Result status:", result.get("status"))
print("Model:", result.get("model"))
sprites = result.get("sprites", [])
print(f"Generated {len(sprites)} sprites.")
assert len(sprites) == 4, f"Expected 4 sprites, got {len(sprites)}"

artifact_dir = r"C:\Users\manis\.gemini\antigravity\brain\0e75ca3f-2ae4-4b95-8ef8-c8e5292ad012"
os.makedirs("phase_dcgan_improvement", exist_ok=True)

for idx, b64_str in enumerate(sprites):
    raw_b64 = b64_str.split(",")[1]
    img_data = base64.b64decode(raw_b64)
    img = Image.open(io.BytesIO(img_data))
    
    print(f"Sprite #{idx+1}: size={img.size}, mode={img.mode}")
    assert img.size == (512, 512), f"Expected 512x512, got {img.size}"
    assert img.mode == "RGBA", f"Expected RGBA, got {img.mode}"
    
    arr = np.array(img)
    alpha_uniques = np.unique(arr[..., 3])
    fg_mask = arr[..., 3] > 127
    fg_ratio = np.mean(fg_mask)
    n_colors = len(np.unique(arr[..., :3][fg_mask], axis=0)) if np.any(fg_mask) else 0
    
    print(f"  Alpha values: {alpha_uniques.tolist()}, FG ratio: {fg_ratio*100:.1f}%, Unique Colors: {n_colors}")
    
    out_path = f"phase_dcgan_improvement/service_test_archer_{idx+1}.png"
    img.save(out_path)
    img.save(os.path.join(artifact_dir, f"service_test_archer_{idx+1}.png"))

# Composite all 4 into a single row test preview
composite = Image.new("RGBA", (512 * 4 + 50, 512 + 40), (240, 245, 250, 255))
for idx, b64_str in enumerate(sprites):
    raw_b64 = b64_str.split(",")[1]
    img = Image.open(io.BytesIO(base64.b64decode(raw_b64)))
    composite.paste(img, (20 + idx * (512 + 10), 20), mask=img.split()[3])

composite.save("phase_dcgan_improvement/gui_service_verified_output.png")
composite.save(os.path.join(artifact_dir, "gui_service_verified_output.png"))
print("GUI Service Verification Successful! All assertions passed.")

