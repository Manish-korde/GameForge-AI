import os
import json
import re
import hashlib
import numpy as np
import pandas as pd
from collections import Counter
from PIL import Image
import matplotlib.pyplot as plt
from datasets import load_dataset

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
EVAL_DIR = os.path.join(PROJECT_ROOT, "evaluation")
PLOTS_DIR = os.path.join(EVAL_DIR, "plots")
CONTACT_DIR = os.path.join(EVAL_DIR, "contact_sheets")

for d in [EVAL_DIR, PLOTS_DIR, CONTACT_DIR]:
    os.makedirs(d, exist_ok=True)

print("=== PHASE 1: ALUCARD DATASET ANALYSIS ===")
print("Loading 'evilsocket/alucard-sprites' dataset...")

ds = load_dataset("evilsocket/alucard-sprites")
raw_data = ds["train"]
raw_count = len(raw_data)
print(f"1. Raw Dataset Row Count: {raw_count}")

# Deduplication / Verification
seen_hashes = set()
unique_indices = []
duplicate_count = 0

print("2. Verifying exact image hashes & deduplication...")
for idx in range(raw_count):
    row = raw_data[idx]
    img = row["image"].convert("RGBA")
    h = hashlib.sha256(np.asarray(img, dtype=np.uint8).tobytes()).hexdigest()
    if h in seen_hashes:
        duplicate_count += 1
    else:
        seen_hashes.add(h)
        unique_indices.append(idx)

unique_count = len(unique_indices)
print(f"   Raw Entries: {raw_count}")
print(f"   Exact Duplicates: {duplicate_count}")
print(f"   Unique Images: {unique_count}")

# Category taxonomy rules & normalization
CATEGORY_PATTERNS = [
    ("Character", r"\b(character|hero|knight|wizard|mage|spellcaster|warrior|rogue|ninja|paladin|archer|necromancer|cleric|bard|monk|priest|sorcerer|fighter|adventurer|man|woman|boy|girl|person|human|dwarf|elf|orc|goblin|skeleton|zombie|demon|vampire)\b"),
    ("Enemy / Monster", r"\b(enemy|monster|boss|slime|bat|ghost|spider|dragon|golem|beast|creature|zombie|skeleton|demon|vampire|imp|snake|wolf|rat|giant|minotaur)\b"),
    ("Weapon", r"\b(weapon|sword|blade|dagger|bow|crossbow|staff|wand|axe|hammer|spear|shield|mace|scythe|rifle|gun|pistol|shield)\b"),
    ("Item / Consumable", r"\b(item|potion|elixir|scroll|book|ring|amulet|gem|jewel|crystal|coin|gold|chest|key|food|apple|bread|meat|bottle|flask|chest)\b"),
    ("Prop / Decor", r"\b(prop|decor|furniture|table|chair|torch|banner|door|window|statue|column|pillar|barrel|crate|box|sign|fence|gate|grave|tomb)\b"),
    ("Effect / Spell", r"\b(effect|spell|magic|fire|ice|lightning|spark|explosion|aura|trail|slash|impact|beam|particle|smoke|flame)\b"),
    ("Tile / Environment", r"\b(tile|wall|floor|ground|grass|stone|brick|tree|rock|bush|water|lava|cliff|path|road|background)\b"),
]

def parse_caption(text):
    if not text:
        return {"category": "Unknown", "tags": [], "style": None, "color": None, "size": None}
    
    text_lower = text.lower()
    tags = [t.strip() for t in text_lower.split(",") if t.strip()]
    
    # Inferred Category
    category = "Unknown / Miscellaneous"
    for cat_label, pattern in CATEGORY_PATTERNS:
        if re.search(pattern, text_lower):
            category = cat_label
            break
            
    # Extract color
    colors = ["gray", "grey", "red", "blue", "green", "yellow", "purple", "black", "white", "brown", "gold", "silver", "orange", "pink"]
    found_color = next((c for c in colors if c in text_lower), None)
    
    # Extract size
    sizes = ["small", "medium", "large", "tiny", "huge", "giant"]
    found_size = next((s for s in sizes if s in text_lower), None)

    # Extract style
    found_style = "pixel art" if "pixel art" in text_lower or "pixel" in text_lower else None

    return {
        "category": category,
        "tags": tags,
        "color": found_color,
        "size": found_size,
        "style": found_style
    }

print("3. Analyzing captions and parsing semantic categories...")
category_counts = Counter()
category_samples = {cat_label: [] for cat_label, _ in CATEGORY_PATTERNS}
category_samples["Unknown / Miscellaneous"] = []

all_parsed_meta = []
image_stats = []

# Analyze a large representative sample (50,000 unique images) for image stats & category counts
sample_indices = unique_indices[:50000]

for idx in sample_indices:
    row = raw_data[idx]
    text = row.get("text", "")
    meta = parse_caption(text)
    category_counts[meta["category"]] += 1
    all_parsed_meta.append(meta)
    
    # Store index for contact sheet generation
    if len(category_samples[meta["category"]]) < 36:
        category_samples[meta["category"]].append(idx)

    # Image statistics
    img = row["image"].convert("RGBA")
    arr = np.asarray(img, dtype=np.uint8)
    alpha = arr[:, :, 3]
    non_zero_alpha = alpha > 10
    
    if np.any(non_zero_alpha):
        rows = np.any(non_zero_alpha, axis=1)
        cols = np.any(non_zero_alpha, axis=0)
        ymin, ymax = np.where(rows)[0][[0, -1]]
        xmin, xmax = np.where(cols)[0][[0, -1]]
        box_w = xmax - xmin + 1
        box_h = ymax - ymin + 1
        occupied_ratio = float((box_w * box_h) / (128 * 128))
        alpha_ratio = float(np.sum(non_zero_alpha) / (128 * 128))
        aspect_ratio = float(box_w / box_h)
    else:
        occupied_ratio = 0.0
        alpha_ratio = 0.0
        aspect_ratio = 1.0

    image_stats.append({
        "occupied_ratio": occupied_ratio,
        "alpha_ratio": alpha_ratio,
        "aspect_ratio": aspect_ratio
    })

print("\n=== CATEGORY DISTRIBUTION (50K Unique Sample) ===")
cat_rows = []
for cat, count in category_counts.most_common():
    pct = (count / len(sample_indices)) * 100
    print(f"  - {cat}: {count:,} ({pct:.2f}%)")
    cat_rows.append({"Category": cat, "Count": count, "Percentage": round(pct, 2)})

# Save CSV
df_cat = pd.DataFrame(cat_rows)
df_cat.to_csv(os.path.join(EVAL_DIR, "category_counts.csv"), index=False)

# Compute Image Statistics
df_stats = pd.DataFrame(image_stats)
mean_occupied = float(df_stats["occupied_ratio"].mean())
mean_alpha = float(df_stats["alpha_ratio"].mean())
mean_aspect = float(df_stats["aspect_ratio"].mean())

print("\n=== IMAGE CONTENT STATISTICS ===")
print(f"  - Mean Occupied Bounding Box Area: {mean_occupied*100:.2f}%")
print(f"  - Mean Non-Transparent Alpha Ratio: {mean_alpha*100:.2f}%")
print(f"  - Mean Aspect Ratio (W/H): {mean_aspect:.2f}")

# Generate Category Distribution Bar Plot
plt.figure(figsize=(10, 5))
categories = df_cat["Category"]
percentages = df_cat["Percentage"]
plt.barh(categories[::-1], percentages[::-1], color='#6366f1')
plt.title("Alucard Dataset Category Distribution (%)", fontsize=14, fontweight='bold')
plt.xlabel("Percentage of Dataset (%)")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "category_distribution.png"), dpi=200)
plt.close()

# Generate Visual Contact Sheets for each category
print("4. Generating visual contact sheets for each category...")
for cat_name, idx_list in category_samples.items():
    if not idx_list:
        continue
    fig, axes = plt.subplots(6, 6, figsize=(12, 12))
    fig.suptitle(f"Category: {cat_name} (Sample Contact Sheet)", fontsize=16, fontweight='bold')
    for ax in axes.flat:
        ax.axis('off')
    for i, idx in enumerate(idx_list[:36]):
        r, c = divmod(i, 6)
        img = raw_data[idx]["image"].convert("RGBA")
        axes[r, c].imshow(img)
        axes[r, c].set_title(f"ID #{idx}", fontsize=8)
    plt.tight_layout()
    safe_cat_name = cat_name.lower().replace(" / ", "_").replace(" ", "_")
    plt.savefig(os.path.join(CONTACT_DIR, f"{safe_cat_name}.png"), dpi=200)
    plt.close()
    print(f"   Saved contact sheet for {cat_name} -> {safe_cat_name}.png")

# Save Summary Analysis JSON
summary_json = {
    "dataset_name": "evilsocket/alucard-sprites",
    "raw_row_count": raw_count,
    "exact_duplicates_removed": duplicate_count,
    "unique_image_count": unique_count,
    "sample_analyzed": len(sample_indices),
    "image_format": "128x128 RGBA",
    "category_distribution": cat_rows,
    "image_statistics": {
        "mean_occupied_bounding_box_pct": round(mean_occupied * 100, 2),
        "mean_non_transparent_alpha_pct": round(mean_alpha * 100, 2),
        "mean_content_aspect_ratio": round(mean_aspect, 2)
    }
}

with open(os.path.join(EVAL_DIR, "dataset_analysis.json"), "w") as f:
    json.dump(summary_json, f, indent=4)

print(f"\nPhase 1 Completed Successfully! Summary saved to {os.path.join(EVAL_DIR, 'dataset_analysis.json')}")
