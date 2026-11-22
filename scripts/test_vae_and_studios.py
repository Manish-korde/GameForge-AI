import requests
import json
import base64

BASE_URL = "http://127.0.0.1:8000"

print("--- 1. Testing VAE Variation ---")
res_var = requests.post(f"{BASE_URL}/generate_variants", data={"image_url": "/alucard_dataset_image/200000", "scale": 1.0})
print("VAE Variant Status:", res_var.status_code)
if res_var.status_code == 200:
    print("  Variant image length:", len(res_var.json().get("variant", "")))

print("\n--- 2. Testing VAE Interpolate (alpha = 0.0, 0.5, 1.0) ---")
for a in [0.0, 0.5, 1.0]:
    res_int = requests.post(f"{BASE_URL}/interpolate_vae", data={
        "image_url_a": "/alucard_dataset_image/200000",
        "image_url_b": "/alucard_dataset_image/200081",
        "alpha": a
    })
    print(f"  Alpha {a} Status:", res_int.status_code, "img len:", len(res_int.json().get("interpolated", "")))

print("\n--- 3. Testing DCGAN Generation ---")
res_dcgan = requests.post(f"{BASE_URL}/dcgan/generate_character", data={"role": "🎲 Random Latent (Fresh Synthesis)", "count": 4, "pixelArtMode": True})
print("DCGAN Status:", res_dcgan.status_code)
if res_dcgan.status_code == 200:
    sprites = res_dcgan.json().get("sprites", [])
    print(f"  DCGAN returned {len(sprites)} sprites")

print("\n--- 4. Testing Action Diffusion Generation ---")
res_diff = requests.post(f"{BASE_URL}/api/action_diffusion/generate", json={"action": "walk", "pose_id": 2, "steps": 25, "seed": 42})
print("Diffusion Status:", res_diff.status_code)
if res_diff.status_code == 200:
    print("  Diffusion returned sprite b64 len:", len(res_diff.json().get("sprite_b64", "")))
