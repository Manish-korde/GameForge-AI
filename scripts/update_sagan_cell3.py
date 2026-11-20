"""
scripts/update_sagan_cell3.py
Updates Cell 3 in build_discrete_sagan_notebook.py to support:
1. Direct HuggingFace Hub loading (from datasets import load_dataset)
2. Embedded 42,212 front-facing indices (zero external file upload needed)
3. Local file fallback if attached
4. BBox crop & bottom-grounding on 64x64 canvas
5. 32-color discrete palette extraction
"""
import os
import json

with open("scripts/embedded_b64.txt", "r", encoding="utf-8") as f:
    b64_content = f.read().strip()
    if b64_content.startswith("EMBEDDED_B64 = "):
        b64_str = b64_content[len("EMBEDDED_B64 = "):].strip(' "')
    elif b64_content.startswith("    EMBEDDED_B64 = "):
        b64_str = b64_content[len("    EMBEDDED_B64 = "):].strip(' "')
    else:
        b64_str = b64_content.strip(' "')

cell3_code = f'''# Section 3: Universal Dataset Loader (HuggingFace Hub + Embedded 42K Indices + BBox Grounding)
import zlib, base64

HF_DATASET_NAME = "evilsocket/alucard-sprites"
INDICES_FILE_NAME = "alucard_frontfacing_42k_indices.json"

# Embedded 42,212 audited front-facing indices
EMBEDDED_B64 = "{b64_str}"

curated_indices = None
INDICES_CANDIDATES = [
    f"{{CONFIG_DIR}}/{{INDICES_FILE_NAME}}",
    f"Notebook/training/{{INDICES_FILE_NAME}}",
    f"Notebook/{{INDICES_FILE_NAME}}",
    f"/kaggle/input/alucard-frontfacing/{{INDICES_FILE_NAME}}",
    f"/kaggle/input/alucard-sprites/{{INDICES_FILE_NAME}}",
    f"/kaggle/working/{{INDICES_FILE_NAME}}"
]

for p in INDICES_CANDIDATES:
    if os.path.isfile(p):
        try:
            with open(p, "r") as f:
                d = json.load(f)
                curated_indices = d if isinstance(d, list) else d.get("indices", [])
            print(f"Loaded {{len(curated_indices):,}} indices from {{p}}")
            break
        except Exception:
            pass

if not curated_indices:
    print("Unpacking pre-embedded 42,212 front-facing indices...")
    curated_indices = json.loads(zlib.decompress(base64.b64decode(EMBEDDED_B64.encode('ascii'))).decode('utf-8'))
    print(f"Successfully unpacked {{len(curated_indices):,}} embedded indices!")

with open(os.path.join(CONFIG_DIR, INDICES_FILE_NAME), "w") as f:
    json.dump(curated_indices, f)

# Check for local directory first; fallback to HuggingFace Hub
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

if DATASET_ROOT is not None:
    print(f"Loading sprites from local directory: {{DATASET_ROOT}}")
    all_files = sorted([
        os.path.join(DATASET_ROOT, f) for f in os.listdir(DATASET_ROOT)
        if f.lower().endswith(('.png', '.webp'))
    ])
    curated_items = [all_files[i] for i in curated_indices if i < len(all_files)]
    is_local_files = True
else:
    print(f"No local directory found. Loading '{{HF_DATASET_NAME}}' from HuggingFace Hub...")
    from datasets import load_dataset
    raw_ds = load_dataset(HF_DATASET_NAME, split="train")
    curated_items = raw_ds.select(curated_indices)
    is_local_files = False

N_SPRITES = len(curated_items)
print(f"Total Curated Front-Facing Sprites to Process: {{N_SPRITES:,}}")

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
        print(f"  Grounded {{i+1:,}}/{{N_SPRITES:,}} sprites ({{time.time() - t0:.1f}}s)")

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
print(f"Global 32-Color Palette saved to {{CONFIG_DIR}}.")
'''

# Now let's update Notebook/training/03_Discrete_SAGAN_Production.ipynb directly
nb_path = "Notebook/training/03_Discrete_SAGAN_Production.ipynb"
with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

# Cell 3 is index 3 (after markdown title, cell 1 env, cell 2 config)
nb["cells"][3]["source"] = [line + "\n" for line in cell3_code.split("\n")]

# Also in Cell 4, idx_dataset quantization loop should use raw_images directly
cell4_code = '''# Quantize all grounded sprites into uint8 palette index array (~172 MB)
BG_INDEX = PALETTE_SIZE  # Index 32 is transparent background
idx_dataset = np.full((N_SPRITES, IMG_HEIGHT, IMG_WIDTH), BG_INDEX, dtype=np.uint8)
pal_32_f32 = global_palette_u8.astype(np.float32)

t0 = time.time()
print(f"Quantizing {N_SPRITES:,} sprites into uint8 index matrix...")
for i in range(N_SPRITES):
    spr = raw_images[i]
    fg = spr[..., 3] > 0
    if np.any(fg):
        px = spr[..., :3][fg].astype(np.float32)
        dists = np.sum((px[:, None, :] - pal_32_f32[None, :, :]) ** 2, axis=-1)
        nearest_idx = np.argmin(dists, axis=-1).astype(np.uint8)
        idx_dataset[i][fg] = nearest_idx
    if (i + 1) % 10000 == 0 or (i + 1) == N_SPRITES:
        print(f"  Quantized {i+1:,}/{N_SPRITES:,} sprites ({(time.time()-t0):.1f}s)")

import gc
del raw_images
gc.collect()

print(f"Total Dataset In-Memory Size: {idx_dataset.nbytes / (1024*1024):.1f} MB (100% GraphDef Safe).")

# RGBA Lookup Table in [-1.0, 1.0]
lut = np.zeros((PALETTE_SIZE + 1, 4), dtype=np.float32)
lut[:PALETTE_SIZE, :3] = (global_palette_u8 / 127.5) - 1.0
lut[:PALETTE_SIZE, 3] = 1.0     # Opaque foreground
lut[BG_INDEX] = -1.0           # Transparent background [-1, -1, -1, -1]
lut_tf = tf.constant(lut, dtype=tf.float32)

def expand_indices_to_rgba(idx_batch):
    return tf.gather(lut_tf, tf.cast(idx_batch, tf.int32))

tf_dataset = (
    tf.data.Dataset.from_tensor_slices(idx_dataset)
    .shuffle(10000, seed=SEED)
    .batch(GLOBAL_BATCH_SIZE, drop_remainder=True)
    .map(expand_indices_to_rgba, num_parallel_calls=tf.data.AUTOTUNE)
    .prefetch(tf.data.AUTOTUNE)
)

dist_dataset = strategy.experimental_distribute_dataset(tf_dataset)
print("Distributed tf.data pipeline created successfully.")
'''
nb["cells"][4]["source"] = [line + "\n" for line in cell4_code.split("\n")]

with open(nb_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=2)

print(f"Updated {nb_path} with universal loader and memory-efficient quantization!")

