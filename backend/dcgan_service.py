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

def remove_stray_floaters_tight(rgba_img: np.ndarray, min_standalone_size: int = 12, touch_dilation: int = 2):
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

def generate(n: int = 4, seed: int = None, role: str = "🎲 Random Latent (Fresh Synthesis)") -> list:
    """
    SINGLE ENTRY POINT:
    generate(n, seed) -> list of 512x512 RGBA PIL Images (8x nearest neighbor).
    Also attaches .b64 attribute or helper if needed.
    """
    global _generator, _palette_tf
    if _generator is None or _palette_tf is None:
        load_dcgan_generator()
        
    n = max(1, min(36, int(n)))
    rng = np.random.RandomState(seed) if seed is not None else np.random.RandomState()
    preset_idx = CLASS_PRESETS.get(role)
    
    output_images = []
    
    for i in range(n):
        if preset_idx is not None and _fixed_latents is not None and preset_idx < len(_fixed_latents):
            base_latent = _fixed_latents[preset_idx].copy()
            if i == 0:
                anchor_z = base_latent
            else:
                rand_perturb = rng.normal(0, 0.2, size=(LATENT_DIM,)).astype(np.float32)
                anchor_z = base_latent + rand_perturb
        else:
            anchor_z = rng.normal(0, 1.0, size=(LATENT_DIM,)).astype(np.float32)
            
        anchor_z = np.clip(anchor_z, -2.0, 2.0)
        
        # 8 candidate variations for quality rejection sampling
        candidates_z = [anchor_z]
        for k in range(7):
            jitter = rng.normal(0, 0.80, size=(LATENT_DIM,)).astype(np.float32)
            cand = 0.80 * anchor_z + 0.20 * jitter
            candidates_z.append(np.clip(cand, -2.0, 2.0))
        candidates_z = np.array(candidates_z, dtype=np.float32)
        
        cand_logits = _generator(candidates_z, training=False)
        cand_imgs = logits_to_image(cand_logits, _palette_tf, hard=True, noise=False)
        
        best_cleaned = cand_imgs[0]
        best_score = float("inf")
        
        for c_img in cand_imgs:
            cleaned_cand, orphans = remove_stray_floaters_tight(c_img)
            score = score_candidate(cleaned_cand, orphans)
            if score < best_score:
                best_score = score
                best_cleaned = cleaned_cand
                
        # 8x nearest-neighbor upscale: 64x64 -> 512x512
        pil_img_64 = Image.fromarray(best_cleaned, mode="RGBA")
        pil_img_512 = pil_img_64.resize((512, 512), Image.NEAREST)
        # Store 64x64 native array on image object for fast pipeline reuse
        pil_img_512.native_64 = best_cleaned
        output_images.append(pil_img_512)
        
    return output_images

def generate_character_sprites(
    role: str = "🎲 Random Latent (Fresh Synthesis)",
    count: int = 4,
    seed: int = None,
    pixel_art_mode: bool = True,
    upscale_factor: int = 8
):
    """
    GUI endpoint adapter wrapping generate(n, seed) -> base64 data URLs.
    """
    images_512 = generate(n=count, seed=seed, role=role)
    sprites_b64 = []
    for pil_img in images_512:
        buffer = io.BytesIO()
        pil_img.save(buffer, format="PNG")
        b64_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
        sprites_b64.append(f"data:image/png;base64,{b64_str}")
        
    return {
        "status": "success",
        "role": role,
        "count": len(images_512),
        "generator_status": _status,
        "model": "Discrete DCGAN (50 Epochs, Palette-Conditioned Head)",
        "sprites": sprites_b64
    }
