import sys, os
sys.path.insert(0, os.path.abspath("."))
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

path_v2 = r"C:\Users\manis\Downloads\gameforge_flan_t5_small_v2"
tok = AutoTokenizer.from_pretrained(path_v2)
mod = AutoModelForSeq2SeqLM.from_pretrained(path_v2)
mod.eval()

prompts = [
    "cyberpunk hacker infiltrating corporate data tower",
    "western sheriff in dusty frontier town hunting bandits",
    "space exploration with robot companion searching for energy cells"
]

for p in prompts:
    inp = tok(f"game concept: {p}", return_tensors="pt")
    with torch.no_grad():
        out = mod.generate(**inp, num_beams=1, max_new_tokens=256)
    text = tok.decode(out[0], skip_special_tokens=True)
    print(f"=== PROMPT: {p} ===")
    print("RAW OUTPUT:", text)
    print()
