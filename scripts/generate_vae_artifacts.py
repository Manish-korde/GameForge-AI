import requests
import base64
import os
from PIL import Image
import io

BASE_URL = "http://127.0.0.1:8000"
OUT_DIR = r"C:\Users\manis\.gemini\antigravity\brain\674ad8cd-af16-4b87-91e3-66973cf97bc5"

# 1. VAE Variation
sprite_a_url = "/alucard_dataset_image/200000" # Knight
sprite_b_url = "/alucard_dataset_image/200005" # Wizard or similar

print("Generating VAE Variation...")
res_var = requests.post(f"{BASE_URL}/generate_variants", data={"image_url": sprite_a_url, "scale": 1.0})
if res_var.status_code == 200:
    b64_data = res_var.json()["variant"].split(",")[1]
    img = Image.open(io.BytesIO(base64.b64decode(b64_data)))
    img.save(os.path.join(OUT_DIR, "vae_raw_variant.png"))
    print("Saved vae_raw_variant.png")

# 2. VAE Blends: 0.0, 0.5, 1.0
for a in [0.0, 0.5, 1.0]:
    print(f"Generating VAE Blend alpha={a}...")
    res = requests.post(f"{BASE_URL}/interpolate_vae", data={
        "image_url_a": sprite_a_url,
        "image_url_b": sprite_b_url,
        "alpha": a
    })
    if res.status_code == 200:
        b64_data = res.json()["interpolated"].split(",")[1]
        img = Image.open(io.BytesIO(base64.b64decode(b64_data)))
        filename = f"vae_blend_alpha_{str(a).replace('.', '_')}.png"
        img.save(os.path.join(OUT_DIR, filename))
        print(f"Saved {filename}")

# 3. Multi-step interpolation strip
strip_imgs = []
for a in [0.0, 0.25, 0.5, 0.75, 1.0]:
    res = requests.post(f"{BASE_URL}/interpolate_vae", data={
        "image_url_a": sprite_a_url,
        "image_url_b": sprite_b_url,
        "alpha": a
    })
    if res.status_code == 200:
        b64_data = res.json()["interpolated"].split(",")[1]
        img = Image.open(io.BytesIO(base64.b64decode(b64_data))).convert("RGBA")
        strip_imgs.append((a, img))

if strip_imgs:
    w, h = strip_imgs[0][1].size
    total_w = w * len(strip_imgs) + 20 * (len(strip_imgs) - 1)
    canvas = Image.new("RGBA", (total_w, h + 30), (18, 18, 24, 255))
    x_offset = 0
    for a, img in strip_imgs:
        canvas.paste(img, (x_offset, 25), img)
        x_offset += w + 20
    canvas.save(os.path.join(OUT_DIR, "vae_interpolation_strip.png"))
    print("Saved vae_interpolation_strip.png")
