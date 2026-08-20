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

model = None
model_status = "Model unavailable"

@app.on_event("startup")
async def load_model():
    global model, model_status
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

@app.get("/status")
def status():
    return {"status": f"Autoencoder: {model_status}"}

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
