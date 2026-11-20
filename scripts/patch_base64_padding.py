import os
from pathlib import Path

target_file = Path("backend/multi_pose_diffusion_service.py")
with open(target_file, "r", encoding="utf-8") as f:
    code = f.read()

target = '''        elif isinstance(reference_input, str):
            data_str = reference_input.strip()
            if data_str.startswith("data:image"):
                data_str = data_str.split(",", 1)[1]
            try:
                img_bytes = base64.b64decode(data_str)
                pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGBA")
            except Exception as e:
                raise ValueError(f"Failed to decode base64 reference image: {e}")'''

replacement = '''        elif isinstance(reference_input, str):
            data_str = reference_input.strip()
            if data_str.startswith("http://") or data_str.startswith("https://"):
                try:
                    import urllib.request
                    req = urllib.request.Request(data_str, headers={'User-Agent': 'Mozilla/5.0'})
                    with urllib.request.urlopen(req) as resp:
                        pil_img = Image.open(io.BytesIO(resp.read())).convert("RGBA")
                except Exception as e:
                    raise ValueError(f"Failed to load reference image from URL: {e}")
            elif os.path.exists(data_str):
                try:
                    pil_img = Image.open(data_str).convert("RGBA")
                except Exception as e:
                    raise ValueError(f"Failed to open image file '{data_str}': {e}")
            else:
                if data_str.startswith("data:image"):
                    data_str = data_str.split(",", 1)[1]
                # Auto-pad missing base64 padding characters ('=')
                pad_len = len(data_str) % 4
                if pad_len != 0:
                    data_str += "=" * (4 - pad_len)
                try:
                    img_bytes = base64.b64decode(data_str)
                    pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGBA")
                except Exception as e:
                    raise ValueError(f"Failed to decode base64 reference image: {e}")'''

assert target in code, "Target block not found in multi_pose_diffusion_service.py"
code = code.replace(target, replacement, 1)

with open(target_file, "w", encoding="utf-8") as f:
    f.write(code)

print("SUCCESS: Added URL fetch, local path support, and base64 auto-padding!")

