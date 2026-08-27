import os
import io
import base64
import requests
import numpy as np
from PIL import Image
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import tensorflow as tf
import google.generativeai as genai
from dotenv import load_dotenv
import json

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

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

model = None
model_status = "Model unavailable"

vae_encoder = None
vae_decoder = None
vae_status = "Model unavailable"

@app.on_event("startup")
async def load_model():
    global model, model_status
    global vae_encoder, vae_decoder, vae_status
    try:
        if os.path.exists(MODEL_PATH):
            print(f"Loading model from {MODEL_PATH}")
            model = tf.keras.models.load_model(MODEL_PATH)
            model_status = "Loaded"
            print("Model loaded successfully.")
        else:
            print(f"Model file not found: {MODEL_PATH}")
    except Exception as e:
        print(f"Error loading model: {e}")
        
    try:
        if os.path.exists(VAE_ENCODER_PATH) and os.path.exists(VAE_DECODER_PATH):
            print(f"Loading VAE Encoder from {VAE_ENCODER_PATH}")
            vae_encoder = tf.keras.models.load_model(VAE_ENCODER_PATH, custom_objects={'Sampling': Sampling})
            print(f"Loading VAE Decoder from {VAE_DECODER_PATH}")
            vae_decoder = tf.keras.models.load_model(VAE_DECODER_PATH, custom_objects={'Sampling': Sampling})
            vae_status = "Loaded"
            print("VAE models loaded successfully.")
        else:
            print("VAE model files not found.")
    except Exception as e:
        print(f"Error loading VAE models: {e}")

@app.get("/status")
def status():
    return {
        "status": f"Autoencoder: {model_status}",
        "vae_status": f"VAE: {vae_status}"
    }

class ReconstructRequest(BaseModel):
    image_url: str

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

@app.post("/reconstruct")
async def reconstruct(image_url: str = Form(None), file: UploadFile = File(None)):
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

@app.post("/generate_concept")
async def generate_concept(prompt: str = Form(...)):
    try:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise HTTPException(status_code=500, detail="GEMINI_API_KEY environment variable not set. Please set it in a .env file.")
        
        model = genai.GenerativeModel('gemini-1.5-flash')
        
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
  "rawOutput": "String containing a formatted text summary of the concept"
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
