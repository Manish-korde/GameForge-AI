import os
import subprocess
import time

chrome_path = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
out_dir = r"C:\Users\manis\.gemini\antigravity\brain\674ad8cd-af16-4b87-91e3-66973cf97bc5"
user_data = r"C:\Users\manis\.gemini\chrome_temp_dcgan_demo"

targets = [
    ("dcgan_studio_grid_filtered.png", "http://127.0.0.1:5173/dcgan?auto=true&count=16&seed=1337&quality_filter=true"),
    ("dcgan_studio_grid_raw.png", "http://127.0.0.1:5173/dcgan?auto=true&count=16&seed=1337&quality_filter=false"),
]

for filename, url in targets:
    out_file = os.path.join(out_dir, filename)
    cmd = [
        chrome_path,
        "--headless=new",
        "--disable-gpu",
        f"--user-data-dir={user_data}",
        "--virtual-time-budget=9000",
        "--window-size=1480,1800",
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
