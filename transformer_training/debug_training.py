import math
import torch

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
)

MODEL_NAME = "google/flan-t5-small"

print("=" * 60)
print("GAMEFORGE AI - TRAINING DEBUG")
print("=" * 60)

# ---------------------------------------------------------
# 1. Load tokenizer
# ---------------------------------------------------------
print("\n[1] Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

print("PAD token:", tokenizer.pad_token_id)
print("EOS token:", tokenizer.eos_token_id)
print("Vocab size:", tokenizer.vocab_size)

# ---------------------------------------------------------
# 2. Load model
# ---------------------------------------------------------
print("\n[2] Loading model...")
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

print("Device:", device)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

# ---------------------------------------------------------
# 3. Load dataset
# ---------------------------------------------------------
print("\n[3] Loading dataset...")

dataset = load_dataset(
    "json",
    data_files="transformer_training/train.jsonl"
)["train"]

print("Dataset size:", len(dataset))
print("Columns:", dataset.column_names)

# Take two examples
examples = dataset.select(range(2))

# ---------------------------------------------------------
# 4. Inspect raw examples
# ---------------------------------------------------------
print("\n[4] Raw examples")

for i in range(2):
    print("\nExample", i)
    print("INPUT TYPE:", type(examples[i]["input"]))
    print("INPUT:", examples[i]["input"])

    print("TARGET TYPE:", type(examples[i]["target"]))
    print("TARGET:", examples[i]["target"][:300])

# ---------------------------------------------------------
# 5. Tokenize individually
# ---------------------------------------------------------
print("\n[5] Tokenizing examples...")

features = []

for i in range(2):

    input_text = examples[i]["input"]
    target_text = examples[i]["target"]

    input_tokens = tokenizer(
        input_text,
        max_length=256,
        truncation=True,
    )

    target_tokens = tokenizer(
        text_target=target_text,
        max_length=512,
        truncation=True,
    )

    features.append(
        {
            "input_ids": input_tokens["input_ids"],
            "attention_mask": input_tokens["attention_mask"],
            "labels": target_tokens["input_ids"],
        }
    )

    print(
        f"Example {i}: "
        f"input_tokens={len(input_tokens['input_ids'])}, "
        f"label_tokens={len(target_tokens['input_ids'])}"
    )

# ---------------------------------------------------------
# 6. Data collator
# ---------------------------------------------------------
print("\n[6] Running DataCollatorForSeq2Seq...")

collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model,
)

batch = collator(features)

print("Batch keys:", batch.keys())
print("Input shape:", batch["input_ids"].shape)
print("Attention shape:", batch["attention_mask"].shape)
print("Label shape:", batch["labels"].shape)

# ---------------------------------------------------------
# 7. Inspect labels
# ---------------------------------------------------------
labels = batch["labels"]

print("\n[7] Label diagnostics")

print("PAD token ID:", tokenizer.pad_token_id)
print("Ignored label ID:", -100)

pad_count = (labels == tokenizer.pad_token_id).sum().item()
ignored_count = (labels == -100).sum().item()
non_ignored_count = (labels != -100).sum().item()

print("PAD count:", pad_count)
print("Ignored (-100) count:", ignored_count)
print("Non-ignored count:", non_ignored_count)
print("Total label values:", labels.numel())

print("\nFirst label sequence:")
print(labels[0][:50].tolist())

# ---------------------------------------------------------
# 8. Forward pass
# ---------------------------------------------------------
print("\n[8] Running model forward pass...")

batch = {
    key: value.to(device) if torch.is_tensor(value) else value
    for key, value in batch.items()
}

model.eval()

with torch.no_grad():
    outputs = model(**batch)

loss = outputs.loss.item()

print("MODEL LOSS:", loss)
print("LOSS FINITE:", math.isfinite(loss))

# ---------------------------------------------------------
# 9. Final verdict
# ---------------------------------------------------------
print("\n" + "=" * 60)

if non_ignored_count == 0:
    print("❌ PROBLEM: All labels are ignored.")
elif not math.isfinite(loss):
    print("❌ PROBLEM: Model forward loss is NaN/Inf.")
else:
    print("✅ MODEL + LABELS + COLLATOR ARE WORKING.")

print("=" * 60)