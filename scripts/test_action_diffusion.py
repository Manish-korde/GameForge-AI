import requests
import base64
import os
from PIL import Image
import io

BASE_URL = "http://127.0.0.1:8000"
OUT_DIR = r"C:\Users\manis\.gemini\antigravity\brain\674ad8cd-af16-4b87-91e3-66973cf97bc5"

# Fetch sample sprite
res_img = requests.get(f"{BASE_URL}/alucard_samples/alucard_0.png")
if res_img.status_code == 200:
    ref_b64 = "data:image/png;base64," + base64.b64encode(res_img.content).decode("utf-8")
    payload = {
        "action": "walk",
        "pose_id": 2,
        "reference_b64": ref_b64,
        "steps": 20,
        "seed": 42
    }
    res_diff = requests.post(f"{BASE_URL}/api/action_diffusion/generate", json=payload)
    print("Action Diffusion Status:", res_diff.status_code)
    if res_diff.status_code == 200:
        d = res_diff.json()
        b64 = d.get("image_b64", "")
        if b64:
            if "," in b64:
                b64 = b64.split(",")[1]
            img = Image.open(io.BytesIO(base64.b64decode(b64)))
            img.save(os.path.join(OUT_DIR, "diffusion_sample.png"))
            print("Saved diffusion_sample.png")
        else:
            print("No image_b64 in response")
else:
    print("Failed to fetch sample image:", res_img.status_code)
