import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

MODEL_PATH = "models/gameforge_t5"

device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH).to(device)
model.eval()

prompt = (
    "Create a survival game in a Dense Wilderness "
    "where a stranded survivor faces a wild predator."
)

inputs = tokenizer(
    "game concept: " + prompt,
    return_tensors="pt"
).to(device)

with torch.no_grad():
    output = model.generate(
        **inputs,
        max_new_tokens=512,
        num_beams=4,
        do_sample=False,
    )

print("RAW TOKEN IDS:")
print(output[0].tolist())

print("\nRAW TOKENS:")
print(tokenizer.convert_ids_to_tokens(output[0]))

print("\nRAW TEXT:")
print(tokenizer.decode(output[0], skip_special_tokens=False))

print("\nCLEAN TEXT:")
print(tokenizer.decode(output[0], skip_special_tokens=True))