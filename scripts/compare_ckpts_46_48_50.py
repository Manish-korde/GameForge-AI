import os
import sys
import numpy as np
import tensorflow as tf
from PIL import Image
from scipy.ndimage import label, binary_dilation

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "backend")))
from dcgan_service import build_generator, logits_to_image

MODEL_DIR = r"models\3_DCGAN_42k_v1"
WEIGHTS_H5 = os.path.join(MODEL_DIR, "game_ready_assets", "generator_weights_final.weights.h5")
PALETTE_PATH = os.path.join(MODEL_DIR, "config", "global_palette_32.npy")
FIXED_Z_PATH = os.path.join(MODEL_DIR, "config", "fixed_latent_vectors.npy")
CKPT_DIR = os.path.join(MODEL_DIR, "checkpoints")

palette_u8 = np.load(PALETTE_PATH)
palette_tf = tf.constant(palette_u8 / 255.0, dtype=tf.float32)
fixed_z = np.load(FIXED_Z_PATH)

def remove_stray_floaters_tight(rgba_img, min_standalone_size=12, touch_dilation=2):
    """
    Tightened LCC: drop pieces under ~12 px unless touching the main body (keep weapons attached to it).
    """
    alpha = rgba_img[..., 3] > 127
    if not np.any(alpha):
        return rgba_img, 0
    labeled_array, num_features = label(alpha)
    if num_features <= 1:
        return rgba_img, 0
    
    sizes = np.bincount(labeled_array.ravel())
    largest_label = np.argmax(sizes[1:]) + 1
    body_mask = (labeled_array == largest_label)
    
    # Dilate body to find anything touching/adjacent (hilts, weapons, staves, hair)
    dilated_body = binary_dilation(body_mask, iterations=touch_dilation)
    
    keep_mask = body_mask.copy()
    for i in range(1, num_features + 1):
        if i == largest_label:
            continue
        sz = sizes[i]
        comp_mask = (labeled_array == i)
        # Keep if standalone piece is >= 12 px (large detached weapon/shield) OR if it touches main body
        if sz >= min_standalone_size or np.any(comp_mask & dilated_body):
            keep_mask |= comp_mask
            
    removed_pixels = int(np.sum(alpha) - np.sum(keep_mask))
    cleaned = rgba_img.copy()
    cleaned[~keep_mask] = 0
    return cleaned, removed_pixels

print("Building generator...")
gen = build_generator()

checkpoints_to_test = {
    "ckpt-23 (Ep 46)": os.path.join(CKPT_DIR, "ckpt-23"),
    "ckpt-24 (Ep 48)": os.path.join(CKPT_DIR, "ckpt-24"),
    "ckpt-25 (Ep 50)": os.path.join(CKPT_DIR, "ckpt-25"),
    "final_h5 (Ep 50)": WEIGHTS_H5
}

results = {}
for name, path in checkpoints_to_test.items():
    if path.endswith(".h5"):
        gen.load_weights(path)
    else:
        ckpt = tf.train.Checkpoint(generator=gen)
        ckpt.restore(path).expect_partial()
        
    logits = gen(fixed_z, training=False)
    raw_imgs = logits_to_image(logits, palette_tf, hard=True, noise=False)
    
    orphan_counts = []
    color_counts = []
    fg_fractions = []
    cleaned_imgs = []
    
    for img in raw_imgs:
        cleaned, orphans = remove_stray_floaters_tight(img)
        cleaned_imgs.append(cleaned)
        orphan_counts.append(orphans)
        fg = cleaned[..., 3] > 127
        fg_fractions.append(np.mean(fg))
        if np.any(fg):
            color_counts.append(len(np.unique(cleaned[..., :3][fg], axis=0)))
            
    results[name] = {
        "mean_orphans": np.mean(orphan_counts),
        "mean_colors": np.mean(color_counts),
        "mean_fg": np.mean(fg_fractions) * 100,
        "images": cleaned_imgs
    }
    print(f"{name:18s} | Mean Stray Floaters: {np.mean(orphan_counts):4.2f} px | Mean Colors: {np.mean(color_counts):4.1f} | Mean FG: {np.mean(fg_fractions)*100:4.1f}%")

# Save a comparison grid
os.makedirs("phase_dcgan_improvement", exist_ok=True)
artifact_dir = r"C:\Users\manis\.gemini\antigravity\brain\0e75ca3f-2ae4-4b95-8ef8-c8e5292ad012"

for name, data in results.items():
    clean_name = name.split()[0].replace("-", "_")
    grid = Image.new("RGBA", (6 * 64 + 7 * 4, 6 * 64 + 7 * 4), (255, 255, 255, 255))
    for idx, im_arr in enumerate(data["images"][:36]):
        r, c = idx // 6, idx % 6
        x = 4 + c * 68
        y = 4 + r * 68
        p_im = Image.fromarray(im_arr, mode="RGBA")
        grid.paste(p_im, (x, y), mask=p_im.split()[3])
    out_p = f"phase_dcgan_improvement/checkpoint_compare_{clean_name}.png"
    grid.save(out_p)
    grid.save(os.path.join(artifact_dir, f"checkpoint_compare_{clean_name}.png"))

print("Checkpoint comparison successfully completed!")
