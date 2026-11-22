import os
import io
import base64
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from scipy.ndimage import label, binary_dilation

# Canonical model path
MODEL_RESULTS_DIR = os.path.abspath(os.path.join(
    os.path.dirname(__file__),
    "..", "models", "3_DCGAN_42k_v1"
))
WEIGHTS_PATH = os.path.join(MODEL_RESULTS_DIR, "game_ready_assets", "generator_weights_final.weights.h5")
PALETTE_PATH = os.path.join(MODEL_RESULTS_DIR, "config", "global_palette_32.npy")
LATENT_VECTORS_PATH = os.path.join(MODEL_RESULTS_DIR, "config", "fixed_latent_vectors.npy")

LATENT_DIM = 128
PALETTE_SIZE = 32
G_CHANNELS = [512, 256, 128, 64]
G_INIT_SPATIAL = 4
UPSAMPLE_MODE = "nearest"

CLASS_PRESETS = {
    "🎲 Random Latent (Fresh Synthesis)": None,
    **{f"Latent Preset #{i+1}": i for i in range(16)}
}

_generator = None
_fixed_latents = None
_global_palette_u8 = None
_palette_tf = None
_status = "Not Loaded"

def build_generator(latent_dim=LATENT_DIM):
    """
    Builds the exact Discrete-Palette DCGAN Generator architecture (Notebook 02, Epoch 50).
    Outputs (B, 64, 64, 33) representing 32 color logits + 1 alpha logit.
    """
    z_in = layers.Input(shape=(latent_dim,), name="latent_vector")
    x = layers.Dense(G_INIT_SPATIAL * G_INIT_SPATIAL * G_CHANNELS[0], use_bias=False)(z_in)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Reshape((G_INIT_SPATIAL, G_INIT_SPATIAL, G_CHANNELS[0]))(x)
    
    # 4x4 -> 8x8
    x = layers.UpSampling2D(size=(2, 2), interpolation=UPSAMPLE_MODE)(x)
    x = layers.Conv2D(G_CHANNELS[1], (3, 3), padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)
    
    # 8x8 -> 16x16
    x = layers.UpSampling2D(size=(2, 2), interpolation=UPSAMPLE_MODE)(x)
    x = layers.Conv2D(G_CHANNELS[2], (3, 3), padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)
    
    # 16x16 -> 32x32
    x = layers.UpSampling2D(size=(2, 2), interpolation=UPSAMPLE_MODE)(x)
    x = layers.Conv2D(G_CHANNELS[3], (3, 3), padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)
    
    # 32x32 -> 64x64
    x = layers.UpSampling2D(size=(2, 2), interpolation=UPSAMPLE_MODE)(x)
    x = layers.Conv2D(64, (3, 3), padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)
    
    # Full-Resolution 64x64 Refinement Layer
    x = layers.Conv2D(64, (3, 3), padding="same")(x)
    x = layers.LeakyReLU(0.2)(x)
    
    # Discrete K+1 Logits Head (32 color logits + 1 alpha logit)
    logits = layers.Conv2D(PALETTE_SIZE + 1, (3, 3), padding="same", name="palette_logits")(x)
    return keras.Model(z_in, logits, name="Discrete_DCGAN_Generator")

def logits_to_image(logits, palette_tf, tau=1.0, hard=True, noise=False):
    """
    Transforms continuous K+1 logits into crisp indexed RGBA pixel art.
    Uses noise=False for deterministic evaluation/inference.
    """
    pl = logits[..., :PALETTE_SIZE]
    al = logits[..., PALETTE_SIZE:]
    
    if noise:
        u = tf.random.uniform(tf.shape(pl), minval=1e-6, maxval=1.0)
        gumbel_noise = -tf.math.log(-tf.math.log(u))
        pl = pl + gumbel_noise
        
    ys = tf.nn.softmax(pl / tau, axis=-1)
    if hard:
        y_hard = tf.one_hot(tf.argmax(ys, axis=-1), depth=PALETTE_SIZE, dtype=tf.float32)
        y = tf.stop_gradient(y_hard - ys) + ys
    else:
        y = ys
        
    rgb = tf.tensordot(y, palette_tf, axes=[[-1], [0]])  # (B, 64, 64, 3) in [0, 1]
    a_s = tf.sigmoid(al)
    if hard:
        a_hard = tf.cast(a_s > 0.5, dtype=tf.float32)
        a = tf.stop_gradient(a_hard - a_s) + a_s
    else:
        a = a_s
        
    rgba = tf.concat([rgb * a, a], axis=-1)
    return tf.cast(tf.clip_by_value(rgba * 255.0 + 0.5, 0.0, 255.0), tf.uint8).numpy()

import colorsys

def rgb_to_hsv_deg(r, g, b):
    h, s, v = colorsys.rgb_to_hsv(r / 255.0, g / 255.0, b / 255.0)
    return h * 360.0, s, v

def circular_hue_diff(h1, h2):
    diff = abs(h1 - h2) % 360.0
    return min(diff, 360.0 - diff)

def check_facial_hue_consistency(rgba_img: np.ndarray, max_divergence: float = 45.0):
    """
    Quality Filter v3 Facial Hue Gate:
    Checks if left and right halves of the head region diverge in hue by >= max_divergence.
    """
    alpha = rgba_img[..., 3] > 127
    if not np.any(alpha):
        return True, 0.0
    fg_y, fg_x = np.where(alpha)
    y_min, y_max = np.min(fg_y), np.max(fg_y)
    x_min, x_max = np.min(fg_x), np.max(fg_x)
    h_box = y_max - y_min + 1
    
    head_y_cutoff = y_min + int(0.35 * h_box)
    head_mask = alpha & (np.arange(64)[:, None] <= head_y_cutoff)
    x_mid = (x_min + x_max) // 2
    
    left_head = head_mask & (np.arange(64)[None, :] < x_mid)
    right_head = head_mask & (np.arange(64)[None, :] >= x_mid)
    
    if np.sum(left_head) < 8 or np.sum(right_head) < 8:
        return True, 0.0
        
    l_rgb = rgba_img[..., :3][left_head]
    r_rgb = rgba_img[..., :3][right_head]
    
    l_hues = [rgb_to_hsv_deg(r, g, b)[0] for r, g, b in l_rgb if rgb_to_hsv_deg(r, g, b)[1] > 0.18 and rgb_to_hsv_deg(r, g, b)[2] > 0.18]
    r_hues = [rgb_to_hsv_deg(r, g, b)[0] for r, g, b in r_rgb if rgb_to_hsv_deg(r, g, b)[1] > 0.18 and rgb_to_hsv_deg(r, g, b)[2] > 0.18]
    
    if len(l_hues) < 5 or len(r_hues) < 5:
        return True, 0.0
        
    l_rad = np.deg2rad(l_hues)
    r_rad = np.deg2rad(r_hues)
    l_mean_h = np.rad2deg(np.arctan2(np.mean(np.sin(l_rad)), np.mean(np.cos(l_rad)))) % 360.0
    r_mean_h = np.rad2deg(np.arctan2(np.mean(np.sin(r_rad)), np.mean(np.cos(r_rad)))) % 360.0
    
    h_diff = circular_hue_diff(l_mean_h, r_mean_h)
    return (h_diff < max_divergence), float(h_diff)

def check_head_torso_hue_agreement(rgba_img: np.ndarray, max_clash: float = 100.0):
    """
    Independent Metric: Head-vs-Torso Hue Agreement.
    Extracts chromatic hue of Head zone vs Torso zone.
    Returns (True, diff) if hues are harmonious or neutral, or (False, diff) if clashing.
    """
    alpha = rgba_img[..., 3] > 127
    if not np.any(alpha):
        return True, 0.0
    fg_y, fg_x = np.where(alpha)
    y_min, y_max = np.min(fg_y), np.max(fg_y)
    h_box = y_max - y_min + 1
    
    head_y_cutoff = y_min + int(0.35 * h_box)
    torso_y_cutoff = y_min + int(0.70 * h_box)
    
    head_mask = alpha & (np.arange(64)[:, None] <= head_y_cutoff)
    torso_mask = alpha & (np.arange(64)[:, None] > head_y_cutoff) & (np.arange(64)[:, None] <= torso_y_cutoff)
    
    if np.sum(head_mask) < 10 or np.sum(torso_mask) < 10:
        return True, 0.0
        
    head_rgb = rgba_img[..., :3][head_mask]
    torso_rgb = rgba_img[..., :3][torso_mask]
    
    head_hues = [rgb_to_hsv_deg(r, g, b)[0] for r, g, b in head_rgb if rgb_to_hsv_deg(r, g, b)[1] > 0.20 and rgb_to_hsv_deg(r, g, b)[2] > 0.20]
    torso_hues = [rgb_to_hsv_deg(r, g, b)[0] for r, g, b in torso_rgb if rgb_to_hsv_deg(r, g, b)[1] > 0.20 and rgb_to_hsv_deg(r, g, b)[2] > 0.20]
    
    if len(head_hues) < 8 or len(torso_hues) < 8:
        return True, 0.0
        
    h_rad = np.deg2rad(head_hues)
    t_rad = np.deg2rad(torso_hues)
    h_mean = np.rad2deg(np.arctan2(np.mean(np.sin(h_rad)), np.mean(np.cos(h_rad)))) % 360.0
    t_mean = np.rad2deg(np.arctan2(np.mean(np.sin(t_rad)), np.mean(np.cos(t_rad)))) % 360.0
    
    h_diff = circular_hue_diff(h_mean, t_mean)
    return (h_diff <= max_clash), float(h_diff)

def remove_detached_components_v2(rgba_img: np.ndarray, touch_dilation: int = 2):
    """
    Quality Filter v2:
    - Main body = largest connected component
    - Kept pieces = main body + any component touching main body within 2 px (dilation=2)
    - All other components are detached floaters / stray pixels that are removed
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
    
    dilated_body = binary_dilation(body_mask, iterations=touch_dilation)
    
    keep_mask = body_mask.copy()
    floater_count = 0
    stray_pixels = 0
    
    for i in range(1, num_features + 1):
        if i == largest_label:
            continue
        comp_mask = (labeled_array == i)
        if np.any(comp_mask & dilated_body):
            keep_mask |= comp_mask
        else:
            floater_count += 1
            stray_pixels += int(np.sum(comp_mask))
            
    cleaned = rgba_img.copy()
    cleaned[~keep_mask] = 0
    return cleaned, floater_count, stray_pixels

def remove_stray_floaters_tight(rgba_img: np.ndarray, min_standalone_size: int = 12, touch_dilation: int = 2):
    """Backwards compatibility adapter mapping to remove_detached_components_v2."""
    cleaned, floater_count, stray_pixels = remove_detached_components_v2(rgba_img, touch_dilation=touch_dilation)
    return cleaned, stray_pixels

def load_dcgan_generator():
    """Initializes and loads weights for the 50-epoch Discrete-Palette DCGAN generator. Fails loudly on missing files."""
    global _generator, _fixed_latents, _global_palette_u8, _palette_tf, _status
    if _generator is not None and _palette_tf is not None:
        return _generator
    
    # FAIL LOUDLY: No fallback to old continuous generator
    if not os.path.exists(WEIGHTS_PATH):
        _status = f"FATAL ERROR: Weights file not found at {WEIGHTS_PATH}"
        raise FileNotFoundError(_status)
        
    if not os.path.exists(PALETTE_PATH):
        _status = f"FATAL ERROR: Palette file not found at {PALETTE_PATH}"
        raise FileNotFoundError(_status)
    
    _status = "Loading..."
    print(f"Loading Discrete DCGAN Generator from {WEIGHTS_PATH}...")
    
    _global_palette_u8 = np.load(PALETTE_PATH)
    _palette_tf = tf.constant(_global_palette_u8 / 255.0, dtype=tf.float32)
    print(f"Loaded 32-Color Empirical Palette: {_global_palette_u8.shape}")
    
    gen = build_generator()
    gen.load_weights(WEIGHTS_PATH)
    _generator = gen
    _status = "Loaded (Discrete DCGAN 50-Epoch)"
    print("Discrete DCGAN Generator weights loaded successfully.")
    
    if os.path.exists(LATENT_VECTORS_PATH):
        _fixed_latents = np.load(LATENT_VECTORS_PATH)
        print(f"Loaded {len(_fixed_latents)} fixed latent presets.")
    
    return _generator

def get_status():
    return _status

def score_candidate(c_img: np.ndarray, orphans: int) -> float:
    """
    Rejection score:
    - Penalize stray px
    - Prefer 18-22 unique colors (dataset ground truth ~18)
    - Enforce 0.15 <= FG <= 0.55
    """
    fg_mask = c_img[..., 3] > 127
    fg_frac = np.mean(fg_mask)
    if fg_frac < 0.15 or fg_frac > 0.55:
        return float("inf")
        
    fg_rgb = c_img[..., :3][fg_mask]
    n_colors = len(np.unique(fg_rgb, axis=0)) if np.any(fg_mask) else 0
    
    # Color preference: 18 - 22 colors
    color_penalty = 0.0
    if n_colors < 18:
        color_penalty = (18 - n_colors) * 2.5
    elif n_colors > 22:
        color_penalty = (n_colors - 22) * 1.5
        
    score = (orphans * 3.0) + (abs(fg_frac - 0.32) * 120.0) + color_penalty
    return score

def generate(n: int = 16, seed: int = 1337, role: str = "🎲 Random Latent (Fresh Synthesis)", quality_filter: bool = True) -> list:
    """
    SINGLE ENTRY POINT:
    generate(n, seed, role, quality_filter) -> list of 512x512 RGBA PIL Images (8x nearest neighbor).
    quality_filter: if True (default), applies Quality Filter v2: generates 32 candidates,
                    removes detached components (keeps main body + parts within 2 px),
                    rejects candidates with leftover floaters/strays, ranks by quality score,
                    and returns the best n.
                    if False, generates n raw unfiltered sprites directly from latents.
    """
    global _generator, _palette_tf
    if _generator is None or _palette_tf is None:
        load_dcgan_generator()
        
    n = max(1, min(36, int(n)))
    rng = np.random.RandomState(seed) if seed is not None else np.random.RandomState()
    preset_idx = CLASS_PRESETS.get(role)
    
    output_images = []
    
    if quality_filter:
        # Quality Filter v3: Generate 64 candidates (or max(64, n * 4))
        n_candidates = max(64, n * 4)
        candidates_z = []
        
        if preset_idx is not None and _fixed_latents is not None and preset_idx < len(_fixed_latents):
            base_latent = _fixed_latents[preset_idx].copy()
            candidates_z.append(base_latent)
            for _ in range(n_candidates - 1):
                rand_perturb = rng.normal(0, 0.40, size=(LATENT_DIM,)).astype(np.float32)
                candidates_z.append(np.clip(base_latent + rand_perturb, -2.0, 2.0))
        else:
            for _ in range(n_candidates):
                z = rng.normal(0, 1.0, size=(LATENT_DIM,)).astype(np.float32)
                candidates_z.append(np.clip(z, -2.0, 2.0))
                
        candidates_z = np.array(candidates_z, dtype=np.float32)
        cand_logits = _generator(candidates_z, training=False)
        cand_imgs = logits_to_image(cand_logits, _palette_tf, hard=True, noise=False)
        
        scored_candidates = []
        backup_candidates = []
        for c_img in cand_imgs:
            cleaned_cand, fl_count, stray_px = remove_detached_components_v2(c_img)
            # Re-check for leftover floaters after cleaning
            _, left_fl, left_st = remove_detached_components_v2(cleaned_cand)
            
            # Facial hue consistency gate (Filter v3)
            is_consistent, h_div = check_facial_hue_consistency(cleaned_cand, max_divergence=45.0)
            
            # Head-vs-torso hue agreement (Filter v3 independent metric)
            ht_agree, ht_diff = check_head_torso_hue_agreement(cleaned_cand, max_clash=100.0)
            
            score = score_candidate(cleaned_cand, fl_count)
            if not ht_agree:
                score += (ht_diff - 100.0) * 2.0
                
            backup_candidates.append((score, cleaned_cand))
            
            # Reject candidates with leftover floaters, invalid silhouettes, or bilateral facial split
            if left_fl > 0 or left_st > 0 or score == float("inf") or not is_consistent:
                continue
                
            scored_candidates.append((score, cleaned_cand))
            
        scored_candidates.sort(key=lambda x: x[0])
        
        # If strict rejection left fewer than n candidates, supplement with best available
        if len(scored_candidates) < n:
            backup_candidates.sort(key=lambda x: x[0])
            for item in backup_candidates:
                if len(scored_candidates) >= n:
                    break
                if item not in scored_candidates:
                    scored_candidates.append(item)
                    
        selected_sprites = [x[1] for x in scored_candidates[:n]]
    else:
        # Raw discrete DCGAN generation: 1 candidate per requested sprite (unfiltered)
        raw_z = []
        if preset_idx is not None and _fixed_latents is not None and preset_idx < len(_fixed_latents):
            base_latent = _fixed_latents[preset_idx].copy()
            raw_z.append(base_latent)
            for _ in range(n - 1):
                rand_perturb = rng.normal(0, 0.20, size=(LATENT_DIM,)).astype(np.float32)
                raw_z.append(np.clip(base_latent + rand_perturb, -2.0, 2.0))
        else:
            for _ in range(n):
                z = rng.normal(0, 1.0, size=(LATENT_DIM,)).astype(np.float32)
                raw_z.append(np.clip(z, -2.0, 2.0))
                
        raw_z = np.array(raw_z, dtype=np.float32)
        cand_logits = _generator(raw_z, training=False)
        selected_sprites = logits_to_image(cand_logits, _palette_tf, hard=True, noise=False)
        
    for final_sprite_64 in selected_sprites:
        pil_img_64 = Image.fromarray(final_sprite_64, mode="RGBA")
        pil_img_512 = pil_img_64.resize((512, 512), Image.NEAREST)
        pil_img_512.native_64 = final_sprite_64
        output_images.append(pil_img_512)
        
    return output_images

def find_safe_seed(seed: int = None, max_retries: int = 5) -> int:
    """
    Finds a seed whose generated sprite foreground area fraction is >= 0.10 (10%).
    Retries up to max_retries times if a seed produces a near-empty canvas.
    """
    cur_seed = int(seed) if (seed is not None and str(seed).strip() != "") else int(np.random.randint(1, 999999))
    for _ in range(max_retries + 1):
        imgs = generate(n=1, seed=cur_seed, quality_filter=False)
        arr = getattr(imgs[0], 'native_64', np.array(imgs[0].resize((64, 64), Image.NEAREST)))
        fg_mask = arr[..., 3] > 127
        if np.mean(fg_mask) >= 0.10:
            return cur_seed
        cur_seed = int(np.random.randint(1, 999999))
    return cur_seed

def generate_character_sprites(
    role: str = "🎲 Random Latent (Fresh Synthesis)",
    count: int = 16,
    seed: int = 1337,
    pixel_art_mode: bool = True,
    quality_filter: bool = True,
    upscale_factor: int = 8
):
    """
    GUI endpoint adapter wrapping generate(n, seed, role, quality_filter) -> base64 data URLs.
    """
    images_512 = generate(n=count, seed=seed, role=role, quality_filter=quality_filter)
    sprites_b64 = []
    for pil_img in images_512:
        buffer = io.BytesIO()
        pil_img.save(buffer, format="PNG")
        b64_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
        sprites_b64.append(f"data:image/png;base64,{b64_str}")
        
    filter_label = f"best {len(images_512)} of 64 by quality & facial hue score" if quality_filter else "unfiltered (raw model)"
    return {
        "status": "success",
        "role": role,
        "count": len(images_512),
        "generator_status": _status,
        "model": "Discrete DCGAN (50 Epochs, Palette-Conditioned Head)",
        "quality_filter": quality_filter,
        "filter_label": filter_label,
        "sprites": sprites_b64
    }
