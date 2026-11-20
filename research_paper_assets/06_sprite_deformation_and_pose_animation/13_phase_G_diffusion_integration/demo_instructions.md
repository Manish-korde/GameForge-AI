# GameForge AI: Multi-Pose Diffusion Studio — Demo Instructions

## Overview
This document provides instructions for running and evaluating the integrated multi-pose velocity diffusion model in GameForge AI. The application exposes both an interactive Web GUI (`ActionStudio.jsx`) and RESTful FastAPI endpoints.

---

## 1. Quick Start Guide

### Step 1: Start the Backend API Server
Open a terminal in the project root and activate the Python environment:

```powershell
# Using the project's Python virtual environment
.\backend\venv\Scripts\Activate.ps1

# Start the FastAPI server on port 8000
python -m uvicorn backend.app:app --host 127.0.0.1 --port 8000 --reload
```

The server initializes the `MultiPoseDiffusionService`, verifies the checkpoint SHA-256 (`10d33c5b...`), and confirms readiness:
```text
MultiPoseDiffusionService initialized successfully on cpu.
Loaded: phase_F_experiment_E1\checkpoint_best.pt (SHA-256: 10d33c5b2e7ad4d6f173f92fee2ee47101597bfadb1a677f6ee59d8bb7a1bc25)
Uvicorn running on http://127.0.0.1:8000
```

### Step 2: Start the Frontend UI
In a separate terminal, navigate to the `gui/` directory and start the Vite development server:

```powershell
cd gui
npm run dev
```

Open your browser to:
[http://localhost:5173/action-studio](http://localhost:5173/action-studio)

---

## 2. Using the Multi-Pose Diffusion Studio GUI

### Step 1: Select or Upload Reference Sprite
1. Click **"Choose Sample Character"** to pick one of the curated 64x64 RGBA characters (e.g., Character 00), or click **"Upload Custom Sprite"** to upload any 64x64 transparent RGBA sprite.
2. The UI automatically renders the reference sprite with a pixelated checkerboard background to confirm transparency.

### Step 2: Choose Action
Select between the two academically trained actions:
- **Idle (Action 0)**: Generates the neutral standing pose.
- **Walk (Action 1)**: Unlocks the full 8-frame walking cycle.
*(Note: Unsupported actions like Attack, Run, and Jump have been removed from the studio to ensure strict academic honesty).*

### Step 3: Select Animation Frame / Pose
If **Walk** is selected, click on any of the 8 poses:
- **F1 (Pose 0)**: Contact Point 1
- **F2 (Pose 1)**: Passing Phase (Left leg forward)
- **F3 (Pose 2)**: **Left Apex Stride** (Maximum left-foot stride extension)
- **F4 (Pose 3)**: Return Phase (Left leg returning)
- **F5 (Pose 4)**: Contact Point 2
- **F6 (Pose 5)**: Passing Phase (Right leg forward)
- **F7 (Pose 6)**: **Right Apex Stride** (Maximum right-foot stride extension)
- **F8 (Pose 7)**: Neutral Return Phase

### Step 4: Configure Generation Controls
- **Inference Steps**:
  - `25 Steps`: Fast interactive preview (~1.4s).
  - `50 Steps`: Recommended default / academic convergence (~3.3s).
  - `100 Steps`: Ultra-fine convergence (~5.9s).
- **Deterministic Seed**: Enter any integer (e.g., `42`) for bit-exact reproducibility, or check "Random Seed" for stochastic variations.

### Step 5: Generate Single Pose
1. Click **"Generate Pose"**.
2. The UI displays real-time execution latency (e.g., `2,271 ms`) and presents a side-by-side comparison of the Reference sprite and the Generated sprite.
3. Click **"Download RGBA Sprite (PNG)"** to save the generated frame.

### Step 6: Generate Full Walk-Cycle (One-Click)
1. In the "Full Walk-Cycle Generator" card, click **"Synthesize 8-Frame Walk Cycle"**.
2. The engine executes all 8 DDIM passes in ~18.9s.
3. The UI automatically displays:
   - A looping **8 FPS animated preview** of your character walking.
   - The full **horizontal 512x64 sprite sheet** containing all 8 frames.
   - **Download Animated GIF** button.
   - **Download Sprite Sheet (PNG)** button.

### Step 7: Inspect Academic Validation Metrics
Switch to the **"Academic Model Audit"** tab to view:
- Exact parameter counts and checkpoint hashes.
- Kinematic stride displacement measurements.
- Known model limitations (F7 right-leg stride compression: 38.9%, Right-Leg centroid error: 0.2087 px).

---

## 3. Headless API Testing (cURL & Python)

### 3.1 Check Model Status
```bash
curl -X GET http://localhost:8000/generate/diffusion/status
```
**Response**:
```json
{
  "status": "ready",
  "model_name": "ActionAndPoseConditionedUNet",
  "device": "cpu",
  "checkpoint_sha256": "10d33c5b2e7ad4d6f173f92fee2ee47101597bfadb1a677f6ee59d8bb7a1bc25",
  "parameter_count": 1987044,
  "supported_actions": {
    "idle": {"action_id": 0, "valid_poses": [0]},
    "walk": {"action_id": 1, "valid_poses": [0, 1, 2, 3, 4, 5, 6, 7]}
  }
}
```

### 3.2 Single Pose Synthesis via Python
```python
import base64
import requests

with open("sample_sprite.png", "rb") as f:
    ref_b64 = f"data:image/png;base64,{base64.b64encode(f.read()).decode('utf-8')}"

response = requests.post("http://localhost:8000/generate/diffusion", json={
    "reference_b64": ref_b64,
    "action": "walk",
    "pose_id": 2, # Walk F3 Left Apex
    "steps": 50,
    "seed": 42
})

result = response.json()
print("Pose Name:", result["pose_name"])
print("Latency:", result["latency_ms"], "ms")
```

### 3.3 Full Walk-Cycle Synthesis
```python
response = requests.post("http://localhost:8000/generate/diffusion/walk-cycle", json={
    "reference_b64": ref_b64,
    "steps": 50,
    "seed": 42
})

result = response.json()
print("Frames returned:", result["frame_count"])
# result['sprite_sheet_b64'] contains the 512x64 PNG
# result['gif_b64'] contains the animated GIF
```

