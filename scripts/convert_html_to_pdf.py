import subprocess
import os

chrome = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
html_path = os.path.abspath("submission/GameForge_AI_Master_Demonstration_and_Training.html")
pdf_path = os.path.abspath("submission/GameForge_AI_Master_Demonstration_and_Training.pdf")
url_path = "file:///" + html_path.replace("\\", "/")

cmd = [
    chrome,
    "--headless=new",
    "--disable-gpu",
    "--no-sandbox",
    f"--print-to-pdf={pdf_path}",
    url_path
]

print("Executing Chrome headless print...")
res = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
print("Return code:", res.returncode)
if os.path.exists(pdf_path):
    print(f"✅ Success! PDF created at: {pdf_path} ({os.path.getsize(pdf_path) / 1024:.2f} KB)")
else:
    print("Failed. Stderr:", res.stderr)
