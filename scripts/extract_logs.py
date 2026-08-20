import PyPDF2
import re
import json
import os

pdf_path = r"C:\Users\manis\OneDrive\Desktop\Prompt_to_game_asset_generator\training log_of280k dataset.pdf"
logs_dir = r"C:\Users\manis\OneDrive\Desktop\Prompt_to_game_asset_generator\logs"

text = ""
with open(pdf_path, "rb") as f:
    reader = PyPDF2.PdfReader(f)
    for page in reader.pages:
        text += page.extract_text() + "\n"

# Regex to find loss and val_loss
# Example: loss: 0.0086 - val_loss: 0.0038
# Or: loss: 9.2419e-04 - val_loss: 8.8053e-04
pattern = r"loss:\s*([0-9\.\-e]+)\s*-\s*val_loss:\s*([0-9\.\-e]+)"
matches = re.findall(pattern, text)

print(f"Found {len(matches)} epochs in log.")

history_data = {
    "epoch": [],
    "training_loss": [],
    "validation_loss": []
}

for i, match in enumerate(matches):
    loss = float(match[0])
    val_loss = float(match[1])
    history_data["epoch"].append(i + 1)
    history_data["training_loss"].append(loss)
    history_data["validation_loss"].append(val_loss)

out_path = os.path.join(logs_dir, "history.json")
with open(out_path, "w") as f:
    json.dump(history_data, f, indent=4)

print(f"Saved true history to {out_path}")
