import os
import io
import base64
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

MODEL_RESULTS_DIR = os.path.abspath(os.path.join(
    os.path.dirname(__file__),
    "..", "models", "dcgan_all_characters_50ep_results"
))
WEIGHTS_PATH = os.path.join(MODEL_RESULTS_DIR, "generator_weights_final.weights.h5")
LATENT_VECTORS_PATH = os.path.join(MODEL_RESULTS_DIR, "config", "fixed_latent_vectors.npy")

LATENT_DIM = 128
G_CHANNELS = [512, 256, 128, 64]
G_INIT_SPATIAL = 4
UPSAMPLE_MODE = "nearest"
CHANNELS = 4

# Class archetype presets (Random is first and default for 100% unique synthesis)
CLASS_PRESETS = {
    "🎲 Random Character (Fresh Synthesis)": None,
    "Hero (Knight)": 0,
    "Mage (Sorcerer)": 1,
    "Rogue (Thief)": 2,
    "Paladin (Guardian)": 3,
    "Archer (Ranger)": 4,
    "Warrior (Barbarian)": 5,
    "Cleric (Healer)": 6,
    "Necromancer (Dark Mage)": 7,
    "Goblin (Scout)": 8,
    "Skeleton (Fighter)": 9,
    "Slime (Creature)": 10,
    "Assassin (Ninja)": 11,
    "Monk (Martial)": 12,
    "Druid (Shifter)": 13,
    "Berserker (Heavy)": 14,
    "Champion (Elite)": 15,
}

_generator = None
_fixed_latents = None
_status = "Not Loaded"

def build_generator(latent_dim=LATENT_DIM):
    """Builds the exact DCGAN Generator architecture trained in dcgan_all_characters_50ep."""
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
    
    # 32x32 -> 64x64 RGBA
    x = layers.UpSampling2D(size=(2, 2), interpolation=UPSAMPLE_MODE)(x)
    img_out = layers.Conv2D(CHANNELS, (3, 3), padding="same", activation="tanh", name="rgba_output")(x)
    return keras.Model(z_in, img_out, name="DCGAN_Generator_128D")

def load_dcgan_generator():
    """Initializes and loads weights for the 50-epoch DCGAN character generator."""
    global _generator, _fixed_latents, _status
    if _generator is not None:
        return _generator
    
    if not os.path.exists(WEIGHTS_PATH):
        _status = f"Error: Weights file not found at {WEIGHTS_PATH}"
        print(_status)
        return None
    
    try:
        _status = "Loading..."
        print(f"Loading DCGAN Character Generator from {WEIGHTS_PATH}...")
        gen = build_generator()
        gen.load_weights(WEIGHTS_PATH)
        _generator = gen
        _status = "Loaded (DCGAN 169K 50-Epoch)"
        print("DCGAN 169K Character Generator weights loaded successfully.")
        
        # Load fixed latents if available
        if os.path.exists(LATENT_VECTORS_PATH):
            _fixed_latents = np.load(LATENT_VECTORS_PATH)
            print(f"Loaded {len(_fixed_latents)} fixed latent presets for archetype alignment.")
        
        return _generator
    except Exception as e:
        _status = f"Error: {e}"
        print(f"Failed to load DCGAN Character Generator: {e}")
        return None

def get_status():
    return _status

def quantize_pixel_art(rgba_img: Image.Image, num_colors: int = 20) -> Image.Image:
    """
    Clamps continuous floating-point RGB artifacts into a crisp indexed pixel art palette
    with morphological alpha despeckling and chroma restoration.
    """
    # 1. Binary alpha mask & morphological despeckling (removes floating 1-pixel noise)
    alpha = np.array(rgba_img.split()[3])
    base_mask = (alpha > 60).astype(np.uint8) * 255
    mask_im = Image.fromarray(base_mask, mode="L")
    clean_mask = mask_im.filter(ImageFilter.MinFilter(3)).filter(ImageFilter.MaxFilter(3))
    clean_alpha = np.maximum(np.array(clean_mask), (alpha > 125).astype(np.uint8) * 255)
    
    # 2. Chroma / Color Vibrancy Enhancement (+30%)
    rgb = rgba_img.convert("RGB")
    rgb_boosted = ImageEnhance.Color(rgb).enhance(1.30)
    
    # 3. Discrete Palette Quantization (Median-Cut clamping to retro indexed palette)
    quantized_rgb = rgb_boosted.quantize(colors=num_colors, method=Image.MEDIANCUT).convert("RGB")
    
    # 4. Recombine with binary alpha mask
    quantized_rgb.putalpha(Image.fromarray(clean_alpha, mode="L"))
    return quantized_rgb

def generate_character_sprites(
    role: str = "🎲 Random Character (Fresh Synthesis)",
    count: int = 4,
    seed: int = None,
    pixel_art_mode: bool = True
):
    """
    Generates transparent 64x64 RGBA character sprites using the 50-epoch DCGAN model.
    Produces truly unique random characters on every generation call.
    """
    global _generator
    if _generator is None:
        load_dcgan_generator()
    if _generator is None:
        return {"error": f"DCGAN Generator is unavailable: {_status}", "sprites": []}
    
    count = max(1, min(16, int(count)))
    rng = np.random.RandomState(seed) if seed is not None else np.random.RandomState()
    
    preset_idx = CLASS_PRESETS.get(role)
    
    # Pure Random Mode (default or when Random is selected):
    # Every sprite is drawn from an independent Gaussian latent vector
    if preset_idx is None or _fixed_latents is None or len(_fixed_latents) <= preset_idx:
        z_batch = rng.normal(0, 1.0, size=(count, LATENT_DIM)).astype(np.float32)
    else:
        # Archetype Mode: Anchor to archetype vector with substantial variation per sprite
        base_latent = _fixed_latents[preset_idx]
        base_norm = np.linalg.norm(base_latent)
        z_batch = []
        for i in range(count):
            if i == 0 and seed is not None:
                z = base_latent.copy()
            else:
                # Blend archetype direction with independent random exploration
                random_vec = rng.normal(0, 1.0, size=(LATENT_DIM,)).astype(np.float32)
                z = 0.50 * base_latent + 0.50 * (random_vec / (np.linalg.norm(random_vec) + 1e-8) * base_norm)
                z = z / (np.linalg.norm(z) + 1e-8) * base_norm
            z_batch.append(z)
        z_batch = np.array(z_batch, dtype=np.float32)
    
    preds = _generator.predict(z_batch, verbose=0)
    
    # Scale from [-1, 1] to [0, 255]
    scaled = ((preds + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
    
    sprites_b64 = []
    for i in range(count):
        raw_rgba = scaled[i]
        img = Image.fromarray(raw_rgba, mode="RGBA")
        
        if pixel_art_mode:
            img = quantize_pixel_art(img, num_colors=20)
            
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        b64_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
        sprites_b64.append(f"data:image/png;base64,{b64_str}")
        
    return {
        "status": "success",
        "role": role,
        "count": count,
        "generator_status": _status,
        "model": "DCGAN 169K Characters (50 Epochs)",
        "sprites": sprites_b64
    }
