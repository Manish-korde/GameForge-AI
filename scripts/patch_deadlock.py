import os
from pathlib import Path

target_file = Path("backend/multi_pose_diffusion_service.py")
with open(target_file, "r", encoding="utf-8") as f:
    code = f.read()

target = '''            if data_str.startswith("http://") or data_str.startswith("https://"):
                try:
                    import urllib.request
                    req = urllib.request.Request(data_str, headers={'User-Agent': 'Mozilla/5.0'})
                    with urllib.request.urlopen(req) as resp:
                        pil_img = Image.open(io.BytesIO(resp.read())).convert("RGBA")
                except Exception as e:
                    raise ValueError(f"Failed to load reference image from URL: {e}")'''

replacement = '''            if data_str.startswith("http://") or data_str.startswith("https://"):
                # If localhost / 127.0.0.1, read file directly to prevent single-worker deadlock
                if ":8000/" in data_str:
                    path_part = data_str.split(":8000/", 1)[1].split("?")[0]
                    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                    # Check direct path, or inside root
                    cand1 = os.path.join(root_dir, path_part.replace("/", os.sep))
                    cand2 = os.path.join(root_dir, "backend", path_part.replace("/", os.sep))
                    for c in [cand1, cand2]:
                        if os.path.exists(c):
                            try:
                                pil_img = Image.open(c).convert("RGBA")
                                break
                            except Exception:
                                pass
                if pil_img is None:
                    try:
                        import urllib.request
                        req = urllib.request.Request(data_str, headers={'User-Agent': 'Mozilla/5.0'})
                        with urllib.request.urlopen(req, timeout=4) as resp:
                            pil_img = Image.open(io.BytesIO(resp.read())).convert("RGBA")
                    except Exception as e:
                        raise ValueError(f"Failed to load reference image from URL: {e}")'''

assert target in code, "Target not found"
code = code.replace(target, replacement, 1)

with open(target_file, "w", encoding="utf-8") as f:
    f.write(code)

print("Applied deadlock prevention patch successfully!")

