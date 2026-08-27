import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
from dotenv import load_dotenv
load_dotenv()

import io
import base64
import requests
import numpy as np
from PIL import Image
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import tensorflow as tf
try:
    import google.generativeai as genai
    api_key = os.getenv("GEMINI_API_KEY")
    if api_key:
        genai.configure(api_key=api_key)
except Exception as e:
    print(f"Gemini init warning: {e}")

@tf.keras.utils.register_keras_serializable()
class Sampling(tf.keras.layers.Layer):
    """Uses (z_mean, z_log_var) to sample z."""
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

# Monkey patch layers for newer Keras model compat (stripping quantization_config)
layers_to_patch = [
    tf.keras.layers.Dense,
    tf.keras.layers.Conv2D,
    tf.keras.layers.Conv2DTranspose,
    tf.keras.layers.Flatten,
    tf.keras.layers.Reshape,
    tf.keras.layers.InputLayer
]
for layer_cls in layers_to_patch:
    original_init = layer_cls.__init__
    def make_patched_init(orig_init):
        def patched_init(self, *args, **kwargs):
            kwargs.pop('quantization_config', None)
            orig_init(self, *args, **kwargs)
        return patched_init
    layer_cls.__init__ = make_patched_init(original_init)

app = FastAPI()

# Mount Alucard 280k dataset samples static route
samples_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "alucard_samples"))
if os.path.exists(samples_dir):
    app.mount("/alucard_samples", StaticFiles(directory=samples_dir), name="alucard_samples")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = os.path.abspath(os.path.join(
    os.path.dirname(__file__), 
    "..", "models", "280k dataset model", "AE_280K_best.keras"
))

VAE_ENCODER_PATH = os.path.abspath(os.path.join(
    os.path.dirname(__file__), 
    "..", "models", "280k model VAE (VAE v2)", "VAE_280K_Outputs", "encoder_280k_final.keras"
))

VAE_DECODER_PATH = os.path.abspath(os.path.join(
    os.path.dirname(__file__), 
    "..", "models", "280k model VAE (VAE v2)", "VAE_280K_Outputs", "decoder_280k_final.keras"
))

def preprocess_image(image: Image.Image) -> np.ndarray:
    image = image.convert("RGBA")
    image = image.resize((128, 128))
    img_array = np.asarray(image, dtype=np.float32)
    img_array /= 255.0
    return np.expand_dims(img_array, axis=0)

def postprocess_image(img_array: np.ndarray) -> Image.Image:
    img_array = img_array[0] * 255.0
    img_array = np.clip(img_array, 0, 255).astype(np.uint8)
    return Image.fromarray(img_array, mode="RGBA")

def image_to_base64(image: Image.Image) -> str:
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{img_str}"

ALUCARD_INDEX = []

def preindex_alucard_samples():
    global ALUCARD_INDEX
    if len(ALUCARD_INDEX) > 0:
        return ALUCARD_INDEX
    if vae_encoder is None:
        return []
    try:
        samples_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "alucard_samples"))
        if not os.path.exists(samples_dir):
            return
        
        batch_tensors = []
        metadata = []
        for i in range(20):
            img_path = os.path.join(samples_dir, f"alucard_{i}.png")
            if os.path.exists(img_path):
                img = Image.open(img_path).convert("RGBA")
                tensor = preprocess_image(img)[0] # (128, 128, 4)
                batch_tensors.append(tensor)
                metadata.append({
                    "url": f"http://localhost:8000/alucard_samples/alucard_{i}.png",
                    "path": img_path
                })
        
        if batch_tensors:
            # Predict all 20 images in 1 single fast batch call!
            batch_array = np.array(batch_tensors)
            z_means, _, _ = vae_encoder.predict(batch_array, verbose=0)
            
            index = []
            for idx, meta in enumerate(metadata):
                z_vec = z_means[idx]
                norm = float(np.linalg.norm(z_vec))
                index.append({
                    "url": meta["url"],
                    "path": meta["path"],
                    "latent": z_vec,
                    "norm": norm
                })
            ALUCARD_INDEX = index
            print(f"Pre-indexed {len(ALUCARD_INDEX)} Alucard 280k dataset sprites into VAE latent space in single batch!")
    except Exception as e:
        print(f"Error indexing Alucard dataset samples: {e}")

import threading

model = None
model_status = "Loading..."

model = None
model_status = "Loaded"

vae_encoder = None
vae_decoder = None
vae_status = "Loaded"

def ensure_ae_loaded():
    global model, model_status
    if model is None and os.path.exists(MODEL_PATH):
        try:
            print(f"Loading AE model from {MODEL_PATH}...")
            model = tf.keras.models.load_model(MODEL_PATH)
            model_status = "Loaded"
            print("1. Autoencoder (AE) model loaded successfully.")
        except Exception as e:
            print(f"Error loading AE model: {e}")

def ensure_vae_loaded():
    global vae_encoder, vae_decoder, vae_status
    if vae_encoder is None and os.path.exists(VAE_ENCODER_PATH) and os.path.exists(VAE_DECODER_PATH):
        try:
            print(f"Loading VAE Encoder from {VAE_ENCODER_PATH}...")
            vae_encoder = tf.keras.models.load_model(VAE_ENCODER_PATH, custom_objects={'Sampling': Sampling})
            print(f"Loading VAE Decoder from {VAE_DECODER_PATH}...")
            vae_decoder = tf.keras.models.load_model(VAE_DECODER_PATH, custom_objects={'Sampling': Sampling})
            vae_status = "Loaded"
            print("2. Variational Autoencoder (VAE) model loaded successfully.")
        except Exception as e:
            print(f"Error loading VAE models: {e}")

@app.on_event("startup")
def startup_event():
    print("Backend API server started successfully.")

@app.get("/status")
def status():
    return {
        "status": f"Autoencoder: {model_status}",
        "vae_status": f"VAE: {vae_status}"
    }

class ReconstructRequest(BaseModel):
    image_url: str

@app.post("/reconstruct")
async def reconstruct(image_url: str = Form(None), file: UploadFile = File(None)):
    ensure_ae_loaded()
    if model is None:
        raise HTTPException(status_code=503, detail="Model unavailable")

    try:
        image = None
        if file:
            content = await file.read()
            image = Image.open(io.BytesIO(content))
        elif image_url:
            response = requests.get(image_url)
            response.raise_for_status()
            image = Image.open(io.BytesIO(response.content))
        else:
            raise HTTPException(status_code=400, detail="Must provide file or image_url")

        # 1. Preprocess
        input_tensor = preprocess_image(image)
        
        # 2. Inference
        output_tensor = model.predict(input_tensor)
        
        # 3. Metrics
        mse = float(np.mean(np.square(input_tensor - output_tensor)))
        mae = float(np.mean(np.abs(input_tensor - output_tensor)))
        psnr = float(tf.image.psnr(input_tensor, output_tensor, max_val=1.0).numpy()[0])
        ssim = float(tf.image.ssim(input_tensor[:,:,:,:3], output_tensor[:,:,:,:3], max_val=1.0).numpy()[0])
        
        # New Advanced Metrics
        threshold = 0.05
        exact_match = float(np.mean(np.abs(input_tensor - output_tensor) <= threshold) * 100)
        
        alpha_true = input_tensor[0, :, :, 3] > 0.5
        alpha_pred = output_tensor[0, :, :, 3] > 0.5
        intersection = np.logical_and(alpha_true, alpha_pred).sum()
        union = np.logical_or(alpha_true, alpha_pred).sum()
        alpha_iou = float((intersection / union) * 100) if union > 0 else 100.0
        
        rgb_true = input_tensor[0, :, :, :3]
        rgb_pred = output_tensor[0, :, :, :3]
        color_diff = np.abs(rgb_true - rgb_pred)
        if alpha_true.sum() > 0:
            color_match = float(np.mean(color_diff[alpha_true] <= threshold) * 100)
        else:
            color_match = 100.0
        
        # 4. Postprocess
        reconstructed_img = postprocess_image(output_tensor)
        original_processed_img = postprocess_image(input_tensor)
        
        return {
            "original_processed": image_to_base64(original_processed_img),
            "reconstructed": image_to_base64(reconstructed_img),
            "metrics": {
                "mse": mse,
                "mae": mae,
                "psnr": psnr,
                "ssim": ssim,
                "exact_match": exact_match,
                "alpha_iou": alpha_iou,
                "color_match": color_match
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/reconstruct_noisy")
async def reconstruct_noisy(
    image_url: str = Form(None), 
    file: UploadFile = File(None),
    noise_scale: float = Form(0.1)
):
    if model is None:
        raise HTTPException(status_code=503, detail="Model unavailable")

    try:
        image = None
        if file:
            content = await file.read()
            image = Image.open(io.BytesIO(content))
        elif image_url:
            response = requests.get(image_url)
            response.raise_for_status()
            image = Image.open(io.BytesIO(response.content))
        else:
            raise HTTPException(status_code=400, detail="Must provide file or image_url")

        # 1. Preprocess
        input_tensor = preprocess_image(image)
        
        # 2. Add Gaussian noise
        noise = tf.random.normal(shape=tf.shape(input_tensor), mean=0.0, stddev=noise_scale, dtype=tf.float32)
        noisy_tensor = input_tensor + noise
        noisy_tensor = tf.clip_by_value(noisy_tensor, 0.0, 1.0)
        
        # 3. Inference
        output_tensor = model.predict(noisy_tensor)
        
        # 4. Metrics (Compare output to ORIGINAL clean tensor to see denoising power)
        mse = float(np.mean(np.square(input_tensor - output_tensor)))
        mae = float(np.mean(np.abs(input_tensor - output_tensor)))
        psnr = float(tf.image.psnr(input_tensor, output_tensor, max_val=1.0).numpy()[0])
        ssim = float(tf.image.ssim(input_tensor[:,:,:,:3], output_tensor[:,:,:,:3], max_val=1.0).numpy()[0])
        
        threshold = 0.05
        exact_match = float(np.mean(np.abs(input_tensor - output_tensor) <= threshold) * 100)
        
        alpha_true = input_tensor[0, :, :, 3] > 0.5
        alpha_pred = output_tensor[0, :, :, 3] > 0.5
        intersection = np.logical_and(alpha_true, alpha_pred).sum()
        union = np.logical_or(alpha_true, alpha_pred).sum()
        alpha_iou = float((intersection / union) * 100) if union > 0 else 100.0
        
        rgb_true = input_tensor[0, :, :, :3]
        rgb_pred = output_tensor[0, :, :, :3]
        color_diff = np.abs(rgb_true - rgb_pred)
        if alpha_true.sum() > 0:
            color_match = float(np.mean(color_diff[alpha_true] <= threshold) * 100)
        else:
            color_match = 100.0
            
        # 5. Postprocess
        original_processed_img = postprocess_image(input_tensor)
        noisy_img = postprocess_image(noisy_tensor)
        reconstructed_img = postprocess_image(output_tensor)
        
        return {
            "original_processed": image_to_base64(original_processed_img),
            "noisy_image": image_to_base64(noisy_img),
            "reconstructed": image_to_base64(reconstructed_img),
            "metrics": {
                "mse": mse,
                "mae": mae,
                "psnr": psnr,
                "ssim": ssim,
                "exact_match": exact_match,
                "alpha_iou": alpha_iou,
                "color_match": color_match
            }
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate_variants")
async def generate_variants(
    image_url: str = Form(None), 
    file: UploadFile = File(None),
    scale: float = Form(1.0)
):
    if vae_encoder is None or vae_decoder is None:
        raise HTTPException(status_code=503, detail="VAE Model unavailable")
    
    try:
        # Resolve image
        image = None
        if file:
            image_bytes = await file.read()
            image = Image.open(io.BytesIO(image_bytes)).convert("RGBA")
        elif image_url:
            if image_url.startswith("http"):
                response = requests.get(image_url)
                image = Image.open(io.BytesIO(response.content)).convert("RGBA")
            else:
                raise HTTPException(status_code=400, detail="Invalid image_url")
        else:
            raise HTTPException(status_code=400, detail="Must provide either image_url or file")
        
        # 1. Preprocess
        input_tensor = preprocess_image(image)
        
        # 2. Encode to get latent mean
        # encoder returns [z_mean, z_log_var, z]
        z_mean, z_log_var, _ = vae_encoder.predict(input_tensor)
        
        # 3. Apply perturbation scaling based on global prior std (1.0)
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.random.normal(shape=tf.shape(z_mean))
        
        # Following mathematical resolution: z_variant = \mu + (scale \cdot \epsilon)
        z_variant = z_mean + (scale * epsilon)
        
        # 4. Decode
        output_tensor = vae_decoder.predict(z_variant)
        
        # 5. Postprocess back to PIL image
        variant_image = postprocess_image(output_tensor[0:1])
        
        buffered = io.BytesIO()
        variant_image.save(buffered, format="PNG")
        variant_b64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        
        return {
            "variant": f"data:image/png;base64,{variant_b64}",
            "scale_applied": scale
        }
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/interpolate_vae")
async def interpolate_vae(
    image_url_a: str = Form(None), 
    image_url_b: str = Form(None),
    alpha: float = Form(0.5)
):
    if vae_encoder is None or vae_decoder is None:
        raise HTTPException(status_code=503, detail="VAE Model unavailable")
    
    try:
        if not image_url_a or not image_url_b:
            raise HTTPException(status_code=400, detail="Must provide both image_url_a and image_url_b")
            
        def load_image(url):
            if url.startswith("http"):
                response = requests.get(url)
                response.raise_for_status()
                return Image.open(io.BytesIO(response.content)).convert("RGBA")
            raise HTTPException(status_code=400, detail="Invalid image_url")

        image_a = load_image(image_url_a)
        image_b = load_image(image_url_b)
        
        # 1. Preprocess
        tensor_a = preprocess_image(image_a)
        tensor_b = preprocess_image(image_b)
        
        # 2. Encode to get latent means
        z_mean_a, _, _ = vae_encoder.predict(tensor_a)
        z_mean_b, _, _ = vae_encoder.predict(tensor_b)
        
        # 3. Interpolate in latent space
        z_interpolated = (1.0 - alpha) * z_mean_a + alpha * z_mean_b
        
        # 4. Decode
        output_tensor = vae_decoder.predict(z_interpolated)
        
        # 5. Postprocess back to PIL image
        interp_image = postprocess_image(output_tensor[0:1])
        
        buffered = io.BytesIO()
        interp_image.save(buffered, format="PNG")
        interp_b64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        
        return {
            "interpolated": f"data:image/png;base64,{interp_b64}",
            "alpha": alpha
        }
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/vae/search_similar")
async def vae_search_similar(
    image_url: str = Form(...),
    top_k: int = Form(4)
):
    if vae_encoder is None:
        raise HTTPException(status_code=503, detail="VAE Model unavailable")
    try:
        # Load query image
        if image_url.startswith("http://localhost:8000/alucard_samples/"):
            filename = os.path.basename(image_url)
            local_path = os.path.join(os.path.dirname(__file__), "alucard_samples", filename)
            query_img = Image.open(local_path).convert("RGBA")
        else:
            resp = requests.get(image_url)
            resp.raise_for_status()
            query_img = Image.open(io.BytesIO(resp.content)).convert("RGBA")
            
        query_tensor = preprocess_image(query_img)
        q_z, _, _ = vae_encoder.predict(query_tensor, verbose=0)
        q_vec = q_z[0]
        norm_q = np.linalg.norm(q_vec)
        
        # Match against pre-computed Alucard dataset index (Instant 0.001s response!)
        results = []
        for item in ALUCARD_INDEX:
            c_vec = item["latent"]
            norm_c = item["norm"]
            dot_prod = np.dot(q_vec, c_vec)
            similarity = float((dot_prod / (norm_q * norm_c + 1e-8)) * 100)
            
            results.append({
                "url": item["url"],
                "similarity": round(max(0, min(100, similarity)), 2)
            })
            
        results.sort(key=lambda x: x["similarity"], reverse=True)
        return {"query_url": image_url, "matches": results[:top_k]}
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/vae/cluster_assets")
async def vae_cluster_assets(urls: str = Form(...)):
    if vae_encoder is None:
        raise HTTPException(status_code=503, detail="VAE Model unavailable")
    try:
        url_list = [u.strip() for u in urls.split(",") if u.strip()]
        if not url_list:
            raise HTTPException(status_code=400, detail="No URLs provided")
            
        latents = []
        valid_urls = []
        for url in url_list:
            try:
                if url.startswith("http://localhost:8000/alucard_samples/"):
                    filename = os.path.basename(url)
                    local_path = os.path.join(os.path.dirname(__file__), "alucard_samples", filename)
                    img = Image.open(local_path).convert("RGBA")
                else:
                    resp = requests.get(url)
                    img = Image.open(io.BytesIO(resp.content)).convert("RGBA")
                    
                tensor = preprocess_image(img)
                z_mean, _, _ = vae_encoder.predict(tensor, verbose=0)
                latents.append(z_mean[0])
                valid_urls.append(url)
            except Exception:
                continue
                
        latents = np.array(latents)
        if len(latents) == 0:
            raise HTTPException(status_code=400, detail="Could not process any image URLs")
            
        # Standardize and project 2D with PCA
        mean_vec = np.mean(latents, axis=0)
        centered = latents - mean_vec
        u, s, vt = np.linalg.svd(centered, full_matrices=False)
        coords_2d = centered @ vt[:2].T if latents.shape[0] > 1 else np.zeros((1, 2))
        
        # Simple cluster assignment based on coordinate quadrants
        items = []
        for i in range(len(valid_urls)):
            x, y = float(coords_2d[i, 0]), float(coords_2d[i, 1])
            cluster_id = 1 if x >= 0 and y >= 0 else (2 if x < 0 and y >= 0 else (3 if x < 0 and y < 0 else 4))
            items.append({
                "url": valid_urls[i],
                "x": round(x, 4),
                "y": round(y, 4),
                "cluster": cluster_id
            })
            
        return {"total": len(items), "clusters": items}
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/vae/detect_duplicate")
async def vae_detect_duplicate(
    image_url_a: str = Form(...),
    image_url_b: str = Form(...)
):
    if vae_encoder is None:
        raise HTTPException(status_code=503, detail="VAE Model unavailable")
    try:
        def load_img_local_or_remote(url):
            if url.startswith("http://localhost:8000/alucard_samples/"):
                filename = os.path.basename(url)
                local_path = os.path.join(os.path.dirname(__file__), "alucard_samples", filename)
                return Image.open(local_path).convert("RGBA")
            resp = requests.get(url)
            return Image.open(io.BytesIO(resp.content)).convert("RGBA")

        img_a = load_img_local_or_remote(image_url_a)
        tensor_a = preprocess_image(img_a)
        
        img_b = load_img_local_or_remote(image_url_b)
        tensor_b = preprocess_image(img_b)
        
        z_a, _, _ = vae_encoder.predict(tensor_a, verbose=0)
        z_b, _, _ = vae_encoder.predict(tensor_b, verbose=0)
        
        latent_dist = float(np.linalg.norm(z_a - z_b))
        pixel_mse = float(np.mean(np.square(tensor_a - tensor_b)))
        
        # Match percentage score
        similarity = float(max(0, 100 - (latent_dist * 5.0)))
        is_duplicate = bool(latent_dist < 1.5 or pixel_mse < 0.001)
        
        return {
            "image_url_a": image_url_a,
            "image_url_b": image_url_b,
            "latent_distance": round(latent_dist, 4),
            "pixel_mse": round(pixel_mse, 6),
            "similarity_score": round(similarity, 2),
            "is_duplicate": is_duplicate,
            "status": "DUPLICATE DETECTED" if is_duplicate else "UNIQUE ASSET"
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/vae/anomaly_score")
async def vae_anomaly_score(image_url: str = Form(...)):
    if vae_encoder is None or vae_decoder is None:
        raise HTTPException(status_code=503, detail="VAE Model unavailable")
    try:
        resp = requests.get(image_url)
        img = Image.open(io.BytesIO(resp.content)).convert("RGBA")
        input_tensor = preprocess_image(img)
        
        z_mean, z_log_var, _ = vae_encoder.predict(input_tensor, verbose=0)
        output_tensor = vae_decoder.predict(z_mean, verbose=0)
        
        # Reconstruction Error
        recon_mse = float(np.mean(np.square(input_tensor - output_tensor)))
        
        # KL Divergence: -0.5 * sum(1 + log_var - mean^2 - exp(log_var))
        kl_div = float(-0.5 * np.sum(1 + z_log_var - np.square(z_mean) - np.exp(z_log_var)))
        
        # Total Anomaly Score
        anomaly_score = float(recon_mse * 1000 + kl_div * 0.01)
        
        if anomaly_score < 5.0:
            classification = "Normal / High Quality"
            is_anomaly = False
        elif anomaly_score < 15.0:
            classification = "Moderate Outlier"
            is_anomaly = False
        else:
            classification = "Severe Anomaly / Out-of-Domain"
            is_anomaly = True
            
        return {
            "image_url": image_url,
            "reconstruction_mse": round(recon_mse, 6),
            "kl_divergence": round(kl_div, 4),
            "anomaly_score": round(anomaly_score, 2),
            "classification": classification,
            "is_anomaly": is_anomaly
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate_concept")
async def generate_concept(prompt: str = Form(...)):
    try:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise HTTPException(status_code=500, detail="GEMINI_API_KEY environment variable not set. Please set it in a .env file.")
        
        model = genai.GenerativeModel('gemini-2.5-flash')
        
        system_instructions = """You are an expert game designer. Generate a structured game concept document based on the user's prompt. 
You MUST return ONLY a raw JSON object with no markdown formatting, no code blocks, and no extra text.
The JSON must perfectly match this structure:
{
  "theme": "String (e.g., Dark Fantasy)",
  "environment": "String (e.g., Cursed Forest Village)",
  "characters": [
    { "name": "String", "desc": "String" }
  ],
  "weapons": [
    { "name": "String", "desc": "String" }
  ],
  "props": [
    { "name": "String", "desc": "String" }
  ],
  "visualStyle": "String (e.g., Pixel Art)",
  "rawOutput": "String containing a strict text breakdown exactly like this:\nTheme:\n[theme]\n\nEnvironment:\n[environment]\n\nCharacters:\n- [char1]\n- [char2]\n\nAssets:\n- [asset1]\n- [asset2]\n\nOriginal prompt: [prompt]"
}"""
        
        response = model.generate_content(f"{system_instructions}\n\nUser Prompt: {prompt}")
        
        raw_text = response.text.strip()
        
        # Clean markdown code blocks if the model accidentally includes them
        if raw_text.startswith("```json"):
            raw_text = raw_text.replace("```json", "", 1)
        if raw_text.startswith("```"):
            raw_text = raw_text.replace("```", "", 1)
        if raw_text.endswith("```"):
            raw_text = raw_text[:-3]
            
        json_data = json.loads(raw_text.strip())
        
        return json_data
    except json.JSONDecodeError as e:
        print("Failed to parse JSON from LLM:", response.text)
        raise HTTPException(status_code=500, detail="LLM returned invalid JSON")
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
