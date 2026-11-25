import json
import urllib.request
import urllib.parse
import base64
import io
import numpy as np
from PIL import Image

SERVER_URL = "http://127.0.0.1:8000"

def call_generate(generator_mode="unconditional", class_id=0, count=4, seed=1337, quality_filter=True):
    url = f"{SERVER_URL}/dcgan/generate_character"
    data = urllib.parse.urlencode({
        "generator_mode": generator_mode,
        "class_id": class_id,
        "count": count,
        "seed": seed,
        "quality_filter": str(quality_filter).lower(),
        "role": "🎲 Random Latent (Fresh Synthesis)"
    }).encode("utf-8")
    
    req = urllib.request.Request(url, data=data, method="POST")
    req.add_header("Content-Type", "application/x-www-form-urlencoded")
    
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read().decode("utf-8"))

print("=== SMOKE TEST: Running FastAPI Endpoints ===")

# 1. Classes & Modes Endpoint
with urllib.request.urlopen(f"{SERVER_URL}/dcgan/classes", timeout=30) as resp:
    classes_info = json.loads(resp.read().decode("utf-8"))
    print("\n[GET /dcgan/classes]")
    print(f"Modes: {classes_info.get('modes')}")
    print(f"Classes: {classes_info.get('conditional_classes')}")
    print(f"DCGAN Status: {classes_info.get('dcgan_status')}")
    print(f"cDCGAN Status: {classes_info.get('cdcgan_status')}")

# 2. Test Unconditional Mode
print("\n[POST /dcgan/generate_character - Mode: Unconditional]")
uncond_res = call_generate(generator_mode="unconditional", count=4, seed=1337, quality_filter=True)
print(f"Model: {uncond_res.get('model')}")
print(f"Mode: {uncond_res.get('generator_mode')}")
print(f"Sprites Count: {len(uncond_res.get('sprites', []))}")
print(f"Filter Stats: {uncond_res.get('filter_stats')}")

# 3. Test Conditional Mode (All 4 classes)
cond_results = {}
for cid in range(4):
    cname = ["Knight (Group 0)", "Orc (Group 1)", "Ranger (Group 2)", "Mage (Group 3)"][cid]
    print(f"\n[POST /dcgan/generate_character - Mode: Conditional, Class: {cid} ({cname})]")
    res = call_generate(generator_mode="conditional_v3", class_id=cid, count=4, seed=1337, quality_filter=True)
    cond_results[cid] = res
    print(f"Model: {res.get('model')}")
    print(f"Class Name: {res.get('class_name')}")
    print(f"Class ID: {res.get('class_id')}")
    print(f"Filter Stats: {res.get('filter_stats')}")
    print(f"Filter Label: {res.get('filter_label')}")

# 4. Measure Foreground Colors & Verify Palette Differentiation
def get_mean_fg_rgb(b64_str):
    header, data = b64_str.split(",", 1)
    img = Image.open(io.BytesIO(base64.b64decode(data))).convert("RGBA")
    arr = np.array(img)
    fg = arr[..., 3] > 127
    if np.sum(fg) == 0:
        return np.array([0, 0, 0], dtype=float)
    return np.mean(arr[fg][:, :3], axis=0)

mean_colors = {}
for cid in range(4):
    rgbs = [get_mean_fg_rgb(s) for s in cond_results[cid]["sprites"]]
    mean_colors[cid] = np.mean(rgbs, axis=0)

print("\n--- Palette Discrimination Verification ---")
for cid in range(4):
    cname = ["Knight", "Orc", "Ranger", "Mage"][cid]
    print(f"Class {cid} ({cname}): Mean RGB = {np.round(mean_colors[cid], 1)}")

pairwise_distances = {}
for i in range(4):
    for j in range(i + 1, 4):
        dist = np.linalg.norm(mean_colors[i] - mean_colors[j])
        pair_key = f"{['Knight','Orc','Ranger','Mage'][i]}_vs_{['Knight','Orc','Ranger','Mage'][j]}"
        pairwise_distances[pair_key] = round(float(dist), 2)
        print(f"Distance {pair_key}: {dist:.2f}")

# 5. Output Summary JSON for Smoke Test Report
report_data = {
    "unconditional": {
        "model": uncond_res.get("model"),
        "count": len(uncond_res.get("sprites", [])),
        "filter_stats": uncond_res.get("filter_stats")
    },
    "conditional": {
        cid: {
            "name": cond_results[cid].get("class_name"),
            "class_id": cond_results[cid].get("class_id"),
            "model": cond_results[cid].get("model"),
            "filter_stats": cond_results[cid].get("filter_stats"),
            "mean_rgb": np.round(mean_colors[cid], 1).tolist()
        } for cid in range(4)
    },
    "pairwise_distances": pairwise_distances
}

with open(r"C:\Users\manis\.gemini\antigravity\brain\674ad8cd-af16-4b87-91e3-66973cf97bc5\dcgan_studio_smoke_test_results.json", "w") as f:
    json.dump(report_data, f, indent=2)

print("\nSaved smoke test results JSON.")
