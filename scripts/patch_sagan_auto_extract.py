import json
import re

nb_path = "Notebook/training/03_Discrete_SAGAN_Production.ipynb"
with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

cell3_code = '''# Section 3: Universal Dataset Loader (HuggingFace Hub + Auto-Extract Front-Facing Sprites)
import os, json, re, time
import numpy as np
from PIL import Image

HF_DATASET_NAME = "evilsocket/alucard-sprites"
INDICES_FILE_NAME = "alucard_frontfacing_42k_indices.json"

# 1. Search for existing index file across all candidate locations
curated_indices = None
INDICES_CANDIDATES = [
    f"{CONFIG_DIR}/{INDICES_FILE_NAME}",
    f"{INDICES_FILE_NAME}",
    f"/kaggle/working/{INDICES_FILE_NAME}",
    f"/kaggle/input/{INDICES_FILE_NAME}",
    f"/kaggle/input/alucard-frontfacing/{INDICES_FILE_NAME}",
    f"/kaggle/input/alucard-sprites/{INDICES_FILE_NAME}",
    f"Notebook/training/{INDICES_FILE_NAME}",
    f"Notebook/{INDICES_FILE_NAME}"
]

for p in INDICES_CANDIDATES:
    if os.path.isfile(p):
        try:
            with open(p, "r") as f:
                d = json.load(f)
                curated_indices = d if isinstance(d, list) else d.get("indices", [])
            print(f"Loaded {len(curated_indices):,} indices from {p}")
            break
        except Exception:
            pass

# Check for local images directory first; fallback to HuggingFace Hub
LOCAL_CANDIDATES = [
    "/kaggle/input/alucard-sprites",
    "/kaggle/input/evilsocket-alucard-sprites",
    "../input/alucard-sprites",
    "./alucard-sprites",
    "dataset/alucard-sprites"
]
DATASET_ROOT = None
for p in LOCAL_CANDIDATES:
    if os.path.isdir(p):
        DATASET_ROOT = p
        break

if DATASET_ROOT is not None:
    print(f"Loading sprites from local directory: {DATASET_ROOT}")
    all_files = sorted([
        os.path.join(DATASET_ROOT, f) for f in os.listdir(DATASET_ROOT)
        if f.lower().endswith(('.png', '.webp'))
    ])
    if curated_indices is None:
        curated_indices = list(range(len(all_files)))
    curated_items = [all_files[i] for i in curated_indices if i < len(all_files)]
    is_local_files = True
else:
    print(f"Loading '{HF_DATASET_NAME}' from HuggingFace Hub...")
    from datasets import load_dataset
    raw_ds = load_dataset(HF_DATASET_NAME, split="train")
    
    # Auto-extract front-facing indices if file not uploaded
    if curated_indices is None:
        print("Index file not found on disk. Auto-extracting strictly front-facing sprites from dataset captions...")
        RE_FRONT = re.compile(r"\\b(front view|facing front|front facing|canonical south)\\b", re.IGNORECASE)
        RE_BACK  = re.compile(r"\\b(back view|back|rear|behind|facing away|facing north|north|walk_north)\\b", re.IGNORECASE)
        RE_SIDE  = re.compile(r"\\b(side view|profile|facing east|facing west|walk_east|walk_west|east|west)\\b", re.IGNORECASE)
        
        t_filter = time.time()
        curated_indices = [
            i for i, text in enumerate(raw_ds["text"])
            if RE_FRONT.search(text) and not RE_BACK.search(text) and not RE_SIDE.search(text)
        ]
        print(f"Auto-extracted {len(curated_indices):,} front-facing sprites in {time.time()-t_filter:.1f}s!")
        
        # Save to config for persistence
        with open(os.path.join(CONFIG_DIR, INDICES_FILE_NAME), "w") as f:
            json.dump(curated_indices, f)
            
    curated_items = raw_ds.select(curated_indices)
    is_local_files = False

N_SPRITES = len(curated_items)
print(f"Total Curated Front-Facing Sprites: {N_SPRITES:,}")

def ground_sprite_array(arr):
    # arr is (64, 64, 4) uint8
    fg = arr[..., 3] > 127
    if not np.any(fg):
        return arr
    rows = np.any(fg, axis=1)
    cols = np.any(fg, axis=0)
    ymin, ymax = np.where(rows)[0][[0, -1]]
    xmin, xmax = np.where(cols)[0][[0, -1]]
    crop = arr[ymin:ymax+1, xmin:xmax+1]
    ch, cw = crop.shape[:2]
    canvas = np.zeros((64, 64, 4), dtype=np.uint8)
    start_x = max(0, (64 - cw) // 2)
    start_y = max(0, 64 - 4 - ch)
    canvas[start_y:start_y+ch, start_x:start_x+cw] = crop[:min(ch, 64-start_y), :min(cw, 64-start_x)]
    return canvas

# Preload grounded sprites into RAM
print("Preloading and bottom-grounding sprites into RAM...")
t0 = time.time()
raw_images = np.zeros((N_SPRITES, IMG_HEIGHT, IMG_WIDTH, CHANNELS), dtype=np.uint8)

for i in range(N_SPRITES):
    if is_local_files:
        with Image.open(curated_items[i]) as im:
            im = im.convert("RGBA").resize((64, 64), Image.NEAREST)
            arr = np.array(im)
    else:
        im = curated_items[i]["image"].convert("RGBA").resize((64, 64), Image.NEAREST)
        arr = np.array(im)
        
    # Enforce strict 1-bit alpha and ground
    arr[..., 3] = np.where(arr[..., 3] > 127, 255, 0).astype(np.uint8)
    arr[arr[..., 3] == 0, :3] = 0
    raw_images[i] = ground_sprite_array(arr)
    
    if (i + 1) % 10000 == 0 or (i + 1) == N_SPRITES:
        print(f"  Grounded {i+1:,}/{N_SPRITES:,} sprites ({time.time() - t0:.1f}s)")

# Build 32-Color Discrete Global Palette
print("Computing 32-Color Global Palette from grounded sprites...")
sample_indices = np.random.choice(N_SPRITES, size=min(3000, N_SPRITES), replace=False)
sample_pixels = []
for idx in sample_indices:
    spr = raw_images[idx]
    fg = spr[..., 3] > 0
    if np.any(fg):
        sample_pixels.append(spr[..., :3][fg])

sample_pixels = np.concatenate(sample_pixels, axis=0)
from sklearn.cluster import MiniBatchKMeans
kmeans = MiniBatchKMeans(n_clusters=PALETTE_SIZE, random_state=SEED, batch_size=4096, n_init=3)
kmeans.fit(sample_pixels)
global_palette_u8 = np.clip(np.round(kmeans.cluster_centers_), 0, 255).astype(np.uint8)
np.save(os.path.join(CONFIG_DIR, "global_palette_32.npy"), global_palette_u8)
print(f"Global 32-Color Palette saved to {CONFIG_DIR}.")
'''

nb["cells"][3]["source"] = [line + "\n" for line in cell3_code.split("\n")]

with open(nb_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=2)

print("Updated Cell 3 in 03_Discrete_SAGAN_Production.ipynb with auto-extract fallback!")

