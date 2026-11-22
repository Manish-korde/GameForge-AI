import os
import subprocess
import time

chrome_path = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
out_dir = r"C:\Users\manis\.gemini\antigravity\brain\674ad8cd-af16-4b87-91e3-66973cf97bc5"
user_data = r"C:\Users\manis\.gemini\chrome_temp"

targets = [
    ("vae_variation.png", "http://127.0.0.1:5173/create?test=variation", 6000),
    ("vae_blend_alpha_0_0_ui.png", "http://127.0.0.1:5173/create?test=blend_0", 6000),
    ("vae_blend_alpha_0_5_ui.png", "http://127.0.0.1:5173/create?test=blend_05", 6000),
    ("vae_blend_alpha_1_0_ui.png", "http://127.0.0.1:5173/create?test=blend_1", 6000),
    ("vae_interpolation_ui.png", "http://127.0.0.1:5173/create?test=interpolation", 6000),
    ("dcgan_studio.png", "http://127.0.0.1:5173/dcgan?auto=true", 7000),
    ("diffusion_studio.png", "http://127.0.0.1:5173/actions?auto=true", 10000),
]

for filename, url, budget in targets:
    out_file = os.path.join(out_dir, filename)
    cmd = [
        chrome_path,
        "--headless=new",
        "--disable-gpu",
        f"--user-data-dir={user_data}",
        f"--virtual-time-budget={budget}",
        "--window-size=1440,1600",
        f"--screenshot={out_file}",
        url
    ]
    print(f"Capturing {filename} from {url}...")
    res = subprocess.run(cmd, capture_output=True, text=True)
    if os.path.exists(out_file):
        sz = os.path.getsize(out_file)
        print(f"  -> SUCCESS: {filename} ({sz:,} bytes)")
    else:
        print(f"  -> FAILED: {filename}")
        if res.stderr:
            print("     STDERR:", res.stderr[:200])
