import os
import sys
import numpy as np
from PIL import Image
from scipy.ndimage import label, binary_dilation

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "backend")))
import dcgan_service

gen = dcgan_service.load_dcgan_generator()

def clean_and_measure_v2(rgba_img, touch_dilation=2):
    """
    Quality Filter v2:
    - Main body = largest connected component
    - Kept pieces = main body + any component touching main body within 2 px (dilation=2)
    - All other components are detached floaters
    Returns: cleaned_img, floater_count, stray_pixels
    """
    alpha = rgba_img[..., 3] > 127
    if not np.any(alpha):
        return rgba_img, 0, 0
    labeled_array, num_features = label(alpha)
    if num_features <= 1:
        return rgba_img, 0, 0
        
    sizes = np.bincount(labeled_array.ravel())
    largest_label = np.argmax(sizes[1:]) + 1
    body_mask = (labeled_array == largest_label)
    
    # 2px touch dilation around main body
    dilated_body = binary_dilation(body_mask, iterations=touch_dilation)
    
    keep_mask = body_mask.copy()
    floater_count = 0
    stray_pixels = 0
    
    for i in range(1, num_features + 1):
        if i == largest_label:
            continue
        comp_mask = (labeled_array == i)
        # Keep only if touching main body within 2px
        if np.any(comp_mask & dilated_body):
            keep_mask |= comp_mask
        else:
            floater_count += 1
            stray_pixels += int(np.sum(comp_mask))
            
    cleaned = rgba_img.copy()
    cleaned[~keep_mask] = 0
    return cleaned, floater_count, stray_pixels

print("Testing 32 candidates on seed 1337...")
# Generate 32 raw candidates
rng = np.random.RandomState(1337)
raw_z = rng.normal(0, 1.0, size=(32, dcgan_service.LATENT_DIM)).astype(np.float32)
raw_z = np.clip(raw_z, -2.0, 2.0)
logits = gen(raw_z, training=False)
imgs = dcgan_service.logits_to_image(logits, dcgan_service._palette_tf, hard=True, noise=False)

stats = []
for idx, img in enumerate(imgs):
    cleaned, floaters, strays = clean_and_measure_v2(img)
    score = dcgan_service.score_candidate(cleaned, floaters)
    # Check if candidate has leftover floaters
    # After cleaning, clean_and_measure_v2 on cleaned should have 0 floaters
    cleaned_again, f_left, s_left = clean_and_measure_v2(cleaned)
    stats.append({
        "idx": idx,
        "floaters": floaters,
        "strays": strays,
        "f_left": f_left,
        "s_left": s_left,
        "score": score
    })
    print(f"Cand #{idx:02d}: floaters={floaters}, strays={strays}, leftover={f_left}, score={score:.2f}")

zero_floaters = sum(1 for s in stats if s["floaters"] == 0)
print(f"\nCandidates with 0 floaters: {zero_floaters}/32")
clean_candidates = [s for s in stats if s["score"] < float("inf")]
print(f"Candidates passing score (< inf): {len(clean_candidates)}/32")
