import requests
import json
import base64
from PIL import Image
import io
import os

BASE_URL = "http://127.0.0.1:8000"
OUT_DIR = r"C:\Users\manis\.gemini\antigravity\brain\674ad8cd-af16-4b87-91e3-66973cf97bc5"

res = requests.post(f"{BASE_URL}/dcgan/generate_character", data={"role": "🎲 Random Latent (Fresh Synthesis)", "count": 4, "pixelArtMode": True})
if res.status_code == 200:
    sprites = res.json().get("sprites", [])
    for idx, s in enumerate(sprites):
        b64_str = s if isinstance(s, str) else s.get("image", "")
        b64 = b64_str.split(",")[1] if "," in b64_str else b64_str
        img = Image.open(io.BytesIO(base64.b64decode(b64)))
        img.save(os.path.join(OUT_DIR, f"dcgan_sample_{idx}.png"))
    print(f"Saved {len(sprites)} DCGAN samples")
else:
    print("DCGAN error:", res.status_code, res.text)
